"""Why does SMOTE before splitting report a sensitivity close to s?

imblearn appends synthetic rows after the originals, so test positives can be labelled synthetic or
real, and reported sensitivity = s * r_syn + (1 - s) * r_real. Run for XGBoost, LR and random forest,
with and without rounding the synthetic values of integer columns, plus nearest-neighbour distances.
Data and splits as in run_robustness_experiments.py.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_robustness_experiments as rre  # noqa: E402

OUT = HERE / "outputs"
LOG = open(OUT / "identity_log.txt", "w", encoding="utf-8")
BRFSS_SEEDS = [42, 7, 123]
KAGGLE_SEEDS = list(range(20))


def say(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


def classifiers(seed):
    return {
        "XGBoost": lambda: XGBClassifier(tree_method="hist", random_state=seed, verbosity=0, n_jobs=-1),
        "Logistic regression": lambda: LogisticRegression(max_iter=3000),
        "Random forest": lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed),
    }


def integer_columns(ct, dev, feats):
    """Output columns whose values are integers in original units, with (mean, scale) to undo scaling."""
    X = rre.xform(ct, dev, feats)
    names = ct.get_feature_names_out()
    means = np.zeros(X.shape[1]); scales = np.ones(X.shape[1])
    if "num" in ct.named_transformers_:
        sc = ct.named_transformers_["num"].named_steps["sc"]
        k = len(sc.mean_); means[:k], scales[:k] = sc.mean_, sc.scale_
    orig = X * scales + means
    is_int = np.array([np.allclose(orig[:, j], np.round(orig[:, j]), atol=1e-4) for j in range(X.shape[1])])
    return is_int, means, scales, names


nn_rows = []


def nn_check(Xs, ys, synth, i_tr, i_it, X_aud, y_aud, dataset, seed, n_query=2000):
    """Distance from each kind of positive to its nearest positive in the (leaked) training part."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    tr_pos = i_tr[ys[i_tr] == 1]
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Xs[tr_pos])
    groups = {"synthetic positives, internal test": i_it[(ys[i_it] == 1) & synth[i_it]],
              "real positives, internal test": i_it[(ys[i_it] == 1) & ~synth[i_it]]}
    out = {}
    for name, ii in groups.items():
        q = rng.choice(ii, min(n_query, len(ii)), replace=False)
        out[name] = float(np.median(nn.kneighbors(Xs[q])[0]))
    for name, yy in (("positives, audit holdout", 1), ("negatives, audit holdout", 0)):
        ii = np.where(y_aud == yy)[0]
        q = rng.choice(ii, min(n_query, len(ii)), replace=False)
        out[name] = float(np.median(nn.kneighbors(X_aud[q])[0]))
    say(f"{dataset} seed={seed} median distance to nearest training positive: " +
        "; ".join(f"{k}={v:.3f}" for k, v in out.items()))
    return dict(dataset=dataset, seed=seed) | out


def run(df, dataset, seeds):
    feats = [c for c in df.columns if c != rre.TARGET]
    rows = []
    for seed in seeds:
        tr, cal, thr, audit = rre.four_way_split(df, seed)
        dev = pd.concat([tr, cal, thr], ignore_index=True)
        ct = rre.build_preprocessor(dev, feats)
        X_dev, y_dev = rre.xform(ct, dev, feats), dev[rre.TARGET].values
        X_aud, y_aud = rre.xform(ct, audit, feats), audit[rre.TARGET].values
        is_int, means, scales, _ = integer_columns(ct, dev, feats)
        Xs, ys = SMOTE(random_state=seed).fit_resample(X_dev, y_dev)
        synth = np.arange(len(ys)) >= len(y_dev)  # imblearn appends synthetic rows at the end
        assert synth.sum() == len(ys) - len(y_dev) and ys[synth].min() == 1
        Xr = Xs.copy()
        o = Xr[synth][:, is_int] * scales[is_int] + means[is_int]
        Xr[np.ix_(synth, is_int)] = (np.round(o) - means[is_int]) / scales[is_int]
        frac_nonint = float((np.abs(Xs[synth][:, is_int] * scales[is_int] + means[is_int]
                                    - np.round(Xs[synth][:, is_int] * scales[is_int] + means[is_int])) > 1e-4)
                            .any(axis=1).mean())
        idx = np.arange(len(ys))
        i_tr, i_it = train_test_split(idx, test_size=0.2, stratify=ys, random_state=seed)
        pos_it = i_it[ys[i_it] == 1]
        s = synth[pos_it].mean()
        if seed == seeds[0]:
            nn_rows.append(nn_check(Xs, ys, synth, i_tr, i_it, X_aud, y_aud, dataset, seed))
        for variant, X in (("SMOTE", Xs), ("SMOTE + rounding", Xr)):
            for cname, make in classifiers(seed).items():
                t0 = time.time()
                clf = make().fit(X[i_tr], ys[i_tr])
                p_it, p_aud = clf.predict_proba(X[i_it])[:, 1], clf.predict_proba(X_aud)[:, 1]
                hit = p_it >= 0.5
                rep_sens = hit[ys[i_it] == 1].mean()
                r_syn = hit[np.isin(i_it, pos_it) & synth[i_it]].mean()
                r_real = hit[np.isin(i_it, pos_it) & ~synth[i_it]].mean()
                neg = ys[i_it] == 0
                rows.append(dict(dataset=dataset, seed=seed, variant=variant, classifier=cname,
                                 synthetic_share_s=s, reported_sensitivity=rep_sens,
                                 sensitivity_synthetic=r_syn, sensitivity_real_internal=r_real,
                                 identity_s_plus=s + (1 - s) * r_real,
                                 reported_specificity=1 - hit[neg].mean(),
                                 reported_roc_auc=roc_auc_score(ys[i_it], p_it),
                                 actual_sensitivity=(p_aud[y_aud == 1] >= 0.5).mean(),
                                 actual_specificity=(p_aud[y_aud == 0] < 0.5).mean(),
                                 actual_roc_auc=roc_auc_score(y_aud, p_aud),
                                 synthetic_rows_with_nonint_value=frac_nonint if variant == "SMOTE" else 0.0))
                r = rows[-1]
                say(f"{dataset} seed={seed} {variant:16s} {cname:19s} s={s:.3f} rep={rep_sens:.3f} "
                    f"r_syn={r_syn:.3f} r_real={r_real:.3f} actual={r['actual_sensitivity']:.3f} "
                    f"AUC rep/act={r['reported_roc_auc']:.3f}/{r['actual_roc_auc']:.3f} ({time.time() - t0:.0f}s)")
    return rows


if __name__ == "__main__":
    rows = run(rre.load_kaggle(), "kaggle", KAGGLE_SEEDS)
    pd.DataFrame(rows).to_csv(OUT / "identity_per_run.csv", index=False)
    rows += run(rre.load_brfss(), "brfss2023", BRFSS_SEEDS)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "identity_per_run.csv", index=False)
    summ = res.groupby(["dataset", "variant", "classifier"]).mean(numeric_only=True).drop(columns="seed").round(4)
    summ.to_csv(OUT / "identity_summary.csv")
    pd.DataFrame(nn_rows).to_csv(OUT / "identity_nearest_neighbour.csv", index=False)
    say("\n" + summ.to_string())
