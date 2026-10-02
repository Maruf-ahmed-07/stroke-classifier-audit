"""Design-based 95% CIs for the survey-weighted estimates.

PSUs (_PSU) are resampled with replacement within strata (_STSTR), and a drawn PSU brings all its
respondents with their weights. Pre-event XGBoost and LR on the 2023 test set, and the
35-predictor XGBoost on BRFSS 2024 (refitted here with the same seed).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_paper_analysis as rpa  # noqa: E402

N_BOOT_TEST, N_BOOT_2024 = 500, 200


def wstats(y, p, w, t):
    pred = p >= t
    tp = w[pred & (y == 1)].sum(); fp = w[pred & (y == 0)].sum()
    fn = w[~pred & (y == 1)].sum(); tn = w[~pred & (y == 0)].sum()
    return dict(roc_auc=roc_auc_score(y, p, sample_weight=w), pr_auc=average_precision_score(y, p, sample_weight=w),
                brier=np.average((p - y) ** 2, weights=w), observed_over_expected=np.average(y, weights=w) / np.average(p, weights=w),
                sensitivity=tp / (tp + fn), specificity=tn / (tn + fp), ppv=tp / max(tp + fp, 1e-12))


def design_boot(y, p, w, strata, psu, t, n_boot, seed=rpa.SEED):
    df = pd.DataFrame({"s": strata, "c": psu, "i": np.arange(len(y))})
    groups = {}
    for (s, c), g in df.groupby(["s", "c"], sort=False):
        groups.setdefault(s, []).append(g["i"].to_numpy())
    rng = np.random.default_rng(seed); reps = []
    for _ in range(n_boot):
        idx = []
        for s, clusters in groups.items():
            k = len(clusters)
            for j in rng.integers(0, k, k):
                idx.append(clusters[j])
        i = np.concatenate(idx)
        reps.append(wstats(y[i], p[i], w[i], t))
    return pd.DataFrame(reps)


def main():
    d23 = pd.read_parquet(rpa.CACHE / "brfss_2023_stroke_rich_clean.parquet")
    d24 = pd.read_parquet(rpa.CACHE / "brfss_2024_stroke_rich_clean.parquet")
    W23 = pd.read_parquet(rpa.OUT / "brfss_2023_weights_aligned.parquet")
    W24 = pd.read_parquet(rpa.OUT / "brfss_2024_weights_aligned.parquet")
    tr_i, ca_i, th_i, te_i = rpa.four_way_split(d23)
    pr = pd.read_parquet(rpa.OUT / "test_predictions_2023.parquet")
    thr = pd.read_csv(rpa.OUT / "primary_thresholds.csv", index_col=0)["primary_threshold"]
    y = pr["y"].values
    w, st, ps = (W23[c].values[te_i] for c in ("_LLCPWT", "_STSTR", "_PSU"))
    rows = []
    for key, label in (("xgb_pre", "XGBoost pre-event, 2023 test"), ("lr_pre", "Logistic regression pre-event, 2023 test")):
        p, t = pr[key].values, thr[key]
        point = wstats(y, p, w, t)
        reps = design_boot(y, p, w, st, ps, t, N_BOOT_TEST)
        rows += [dict(model=label, measure=k, weighted_estimate=v, design_lo=reps[k].quantile(.025),
                      design_hi=reps[k].quantile(.975)) for k, v in point.items()]
        print(label, {k: round(v, 4) for k, v in point.items()}, flush=True)

    feats = [c for c in d23.columns if c != rpa.TARGET and c not in rpa.POST and c not in rpa.ONLY_2023]
    tr, ca, th = (d23.iloc[i].reset_index(drop=True) for i in (tr_i, ca_i, th_i))
    m = rpa.Model("xgb", feats, rpa.stage2_params()).fit(tr, ca, th)
    p24, t24, y24 = m.predict(d24), m.thr["primary"], d24[rpa.TARGET].values
    w24, st24, ps24 = (W24[c].values for c in ("_LLCPWT", "_STSTR", "_PSU"))
    point = wstats(y24, p24, w24, t24)
    reps = design_boot(y24, p24, w24, st24, ps24, t24, N_BOOT_2024)
    rows += [dict(model="XGBoost pre-event (35), BRFSS 2024", measure=k, weighted_estimate=v,
                  design_lo=reps[k].quantile(.025), design_hi=reps[k].quantile(.975)) for k, v in point.items()]
    print("2024", {k: round(v, 4) for k, v in point.items()}, "unweighted ROC", round(roc_auc_score(y24, p24), 4), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(rpa.OUT / "design_ci.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
