"""Same grid as run_identity_generalisation.py with random forest, plus XGBoost and random forest
on 900-respondent samples of the Kaggle data (10 repetitions, as the samples are small).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_robustness_experiments as rre  # noqa: E402

OUT = HERE / "outputs"
LOG = open(OUT / "identity_generalisation_ext_log.txt", "w", encoding="utf-8")
SAMPLERS = {"Random oversampling": lambda a, s: RandomOverSampler(sampling_strategy=a, random_state=s),
            "SMOTE": lambda a, s: SMOTE(sampling_strategy=a, random_state=s),
            "Borderline-SMOTE": lambda a, s: BorderlineSMOTE(sampling_strategy=a, random_state=s),
            "ADASYN": lambda a, s: ADASYN(sampling_strategy=a, random_state=s)}
CLASSIFIERS = {"XGBoost": lambda s: XGBClassifier(tree_method="hist", random_state=s, verbosity=0, n_jobs=-1),
               "Random forest": lambda s: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=s)}


def say(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


def grid(df, dataset, n, pis, alphas, reps, classifiers):
    feats = [c for c in df.columns if c != rre.TARGET]
    tr, cal, thr, audit = rre.four_way_split(df, 42)
    dev = pd.concat([tr, cal, thr], ignore_index=True)
    ct = rre.build_preprocessor(dev, feats)
    X_dev, y_dev = rre.xform(ct, dev, feats), dev[rre.TARGET].values
    X_aud, y_aud = rre.xform(ct, audit, feats), audit[rre.TARGET].values
    pos, neg = np.where(y_dev == 1)[0], np.where(y_dev == 0)[0]
    rows = []
    for pi in pis:
        for rep in reps:
            rng = np.random.default_rng(1000 * rep + int(pi * 1000))
            n1 = int(round(pi * n))
            idx = np.concatenate([rng.choice(pos, n1, replace=False), rng.choice(neg, n - n1, replace=False)])
            X, y = X_dev[idx], y_dev[idx]
            for alpha in alphas:
                for sname, make in SAMPLERS.items():
                    try:
                        Xs, ys = make(alpha, rep).fit_resample(X, y)
                    except Exception as e:
                        say(f"skip {dataset} pi={pi} alpha={alpha} {sname}: {e}")
                        continue
                    synth = np.arange(len(ys)) >= len(y)
                    n_syn = int(synth.sum())
                    i_tr, i_it = train_test_split(np.arange(len(ys)), test_size=0.2, stratify=ys, random_state=rep)
                    for cname in classifiers:
                        clf = CLASSIFIERS[cname](rep).fit(Xs[i_tr], ys[i_tr])
                        hit = clf.predict_proba(Xs[i_it])[:, 1] >= 0.5
                        yit, sit = ys[i_it], synth[i_it]
                        rows.append(dict(
                            dataset=dataset, classifier=cname, prevalence=pi, alpha=alpha, oversampler=sname, rep=rep,
                            s_counts=n_syn / (n1 + n_syn), reported_sensitivity=hit[yit == 1].mean(),
                            reported_precision=hit[yit == 1].sum() / max(hit.sum(), 1),
                            r_syn=hit[(yit == 1) & sit].mean() if ((yit == 1) & sit).any() else np.nan,
                            r_real=hit[(yit == 1) & ~sit].mean() if ((yit == 1) & ~sit).any() else np.nan,
                            actual_sensitivity=(clf.predict_proba(X_aud[y_aud == 1])[:, 1] >= 0.5).mean()))
                        r = rows[-1]
                        say(f"{dataset} {cname:13s} pi={pi:.2f} a={alpha} {sname:19s} rep={rep} s={r['s_counts']:.3f} "
                            f"rep={r['reported_sensitivity']:.3f} actual={r['actual_sensitivity']:.3f}")
    return rows


def main():
    rows = grid(rre.load_brfss(), "BRFSS 2023", 40_000, [0.01, 0.02, 0.05, 0.10, 0.20], [0.5, 1.0], [0, 1, 2],
                ["Random forest"])
    rows += grid(rre.load_kaggle(), "Kaggle", 900, [0.02, 0.05, 0.10, 0.20], [0.5, 1.0], list(range(10)),
                 ["XGBoost", "Random forest"])
    res = pd.DataFrame(rows)
    res["abs_error"] = (res.reported_sensitivity - res.s_counts).abs()
    res["a_est"] = (res.reported_sensitivity - res.s_counts) / (1 - res.s_counts)
    res.to_csv(OUT / "identity_generalisation_ext.csv", index=False)
    summ = res.groupby(["dataset", "classifier", "oversampler", "prevalence", "alpha"]).mean(numeric_only=True) \
        .drop(columns="rep").round(4)
    summ.to_csv(OUT / "identity_generalisation_ext_summary.csv")
    smote = res[res.oversampler != "Random oversampling"]
    for (ds, cl), g in smote.groupby(["dataset", "classifier"]):
        low = g[g.prevalence <= 0.05]
        hi = g[g.prevalence >= 0.10]
        say(f"{ds} / {cl} (SMOTE variants): prevalence <=5%: max |rep - s| = {low.abs_error.max():.4f}, "
            f"mean = {low.abs_error.mean():.4f}, runs = {len(low)}; prevalence >=10%: mean |a_est - actual| = "
            f"{(hi.a_est - hi.actual_sensitivity).abs().mean():.4f}")


if __name__ == "__main__":
    main()
