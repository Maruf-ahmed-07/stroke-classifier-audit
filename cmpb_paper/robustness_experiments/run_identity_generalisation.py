"""Does the synthetic-share check hold beyond 1:1 SMOTE at 4% prevalence?

Samples of 40,000 from the BRFSS 2023 development pool with prevalence 1-20% are oversampled to
alpha = 0.5 or 1 (random oversampling, SMOTE, Borderline-SMOTE, ADASYN) and passed through
pipeline C3 (split after oversampling, default XGBoost, threshold 0.5); 3 repetitions each.
Expected reported sensitivity: about s = 1 - pi / (alpha * (1 - pi)) or more.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_robustness_experiments as rre  # noqa: E402

OUT = HERE / "outputs"
LOG = open(OUT / "identity_generalisation_log.txt", "w", encoding="utf-8")
N = 40_000
PIS = [0.01, 0.02, 0.05, 0.10, 0.20]
ALPHAS = [0.5, 1.0]
REPS = [0, 1, 2]
SAMPLERS = {"Random oversampling": lambda a, s: RandomOverSampler(sampling_strategy=a, random_state=s),
            "SMOTE": lambda a, s: SMOTE(sampling_strategy=a, random_state=s),
            "Borderline-SMOTE": lambda a, s: BorderlineSMOTE(sampling_strategy=a, random_state=s),
            "ADASYN": lambda a, s: ADASYN(sampling_strategy=a, random_state=s)}


def say(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


def main():
    df = rre.load_brfss()
    feats = [c for c in df.columns if c != rre.TARGET]
    tr, cal, thr, audit = rre.four_way_split(df, 42)
    dev = pd.concat([tr, cal, thr], ignore_index=True)
    ct = rre.build_preprocessor(dev, feats)
    X_dev, y_dev = rre.xform(ct, dev, feats), dev[rre.TARGET].values
    X_aud, y_aud = rre.xform(ct, audit, feats), audit[rre.TARGET].values
    pos, neg = np.where(y_dev == 1)[0], np.where(y_dev == 0)[0]
    rows = []
    for pi in PIS:
        for rep in REPS:
            rng = np.random.default_rng(1000 * rep + int(pi * 1000))
            n1 = int(round(pi * N))
            idx = np.concatenate([rng.choice(pos, n1, replace=False), rng.choice(neg, N - n1, replace=False)])
            X, y = X_dev[idx], y_dev[idx]
            for alpha in ALPHAS:
                for name, make in SAMPLERS.items():
                    try:
                        Xs, ys = make(alpha, rep).fit_resample(X, y)
                    except Exception as e:  # ADASYN can refuse when no hard examples exist
                        say(f"skip pi={pi} alpha={alpha} {name}: {e}")
                        continue
                    synth = np.arange(len(ys)) >= len(y)  # imblearn appends new rows after the originals
                    assert ys[synth].min() == 1
                    n_syn = int(synth.sum())
                    i_tr, i_it = train_test_split(np.arange(len(ys)), test_size=0.2, stratify=ys, random_state=rep)
                    clf = XGBClassifier(tree_method="hist", random_state=rep, verbosity=0, n_jobs=-1)
                    clf.fit(Xs[i_tr], ys[i_tr])
                    hit = clf.predict_proba(Xs[i_it])[:, 1] >= 0.5
                    yit, sit = ys[i_it], synth[i_it]
                    prec = hit[yit == 1].sum() / max(hit.sum(), 1)
                    rows.append(dict(
                        prevalence=pi, alpha=alpha, oversampler=name, rep=rep,
                        s_formula=1 - pi / (alpha * (1 - pi)), s_counts=n_syn / (n1 + n_syn),
                        s_test_split=sit[yit == 1].mean(),
                        reported_sensitivity=hit[yit == 1].mean(), reported_precision=prec,
                        r_syn=hit[(yit == 1) & sit].mean(), r_real=hit[(yit == 1) & ~sit].mean(),
                        actual_sensitivity=(clf.predict_proba(X_aud[y_aud == 1])[:, 1] >= 0.5).mean()))
                    r = rows[-1]
                    say(f"pi={pi:.2f} alpha={alpha} {name:19s} rep={rep} s={r['s_counts']:.3f} "
                        f"reported={r['reported_sensitivity']:.3f} r_syn={r['r_syn']:.3f} r_real={r['r_real']:.3f} "
                        f"actual={r['actual_sensitivity']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "identity_generalisation.csv", index=False)
    res["abs_error"] = (res.reported_sensitivity - res.s_counts).abs()
    summ = res.groupby(["oversampler", "prevalence", "alpha"]).mean(numeric_only=True).drop(columns="rep").round(4)
    summ.to_csv(OUT / "identity_generalisation_summary.csv")
    say("\n" + summ.to_string())
    say(f"overall: mean |reported - s| = {res.abs_error.mean():.4f}; max = {res.abs_error.max():.4f}; "
        f"correlation = {np.corrcoef(res.reported_sensitivity, res.s_counts)[0, 1]:.4f}; "
        f"mean actual sensitivity = {res.actual_sensitivity.mean():.3f}")
    for name, g in res.groupby("oversampler"):
        say(f"{name}: mean |reported - s| = {g.abs_error.mean():.4f}, max = {g.abs_error.max():.4f}")


if __name__ == "__main__":
    main()
