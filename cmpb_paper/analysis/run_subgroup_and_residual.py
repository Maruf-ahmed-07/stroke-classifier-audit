"""Performance by sex and age, and the residual post-event check (pre-event model refitted
without health insurance, then also without income and home ownership).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_paper_analysis as rpa  # noqa: E402

N_BOOT = 500


def boot_auc_ci(y, p, seed=rpa.SEED):
    rng = np.random.default_rng(seed); out = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], p[i]))
    return np.percentile(out, [2.5, 97.5])


def main():
    d = pd.read_parquet(rpa.CACHE / "brfss_2023_stroke_rich_clean.parquet")
    tr_i, ca_i, th_i, te_i = rpa.four_way_split(d)
    te = d.iloc[te_i].reset_index(drop=True)
    pr = pd.read_parquet(rpa.OUT / "test_predictions_2023.parquet")
    assert np.array_equal(pr["y"].values, te[rpa.TARGET].values), "prediction rows do not match the test split"
    thr = pd.read_csv(rpa.OUT / "primary_thresholds.csv", index_col=0)["primary_threshold"]

    sex = te["sex"].map({1: "Male", 2: "Female"})
    age = pd.cut(te["age"], [17, 44, 64, 79, 80], labels=["18-44", "45-64", "65-79", "80 (top-coded)"])
    groups = [("All", pd.Series(True, index=te.index))]
    groups += [(f"Sex: {s}", sex == s) for s in ("Female", "Male")]
    groups += [(f"Age: {a}", age == a) for a in age.cat.categories]
    groups += [("Sex/age missing", sex.isna() | age.isna())]
    rows = []
    for key in ("xgb_pre", "xgb_full"):
        t = thr[key]
        for name, m in groups:
            m = m.values
            if m.sum() == 0:
                continue
            y, p = pr["y"].values[m], pr[key].values[m]
            lo, hi = boot_auc_ci(y, p) if 0 < y.sum() < len(y) else (np.nan, np.nan)
            met = rpa.metrics(y, p, t)
            rows.append(dict(model=key, group=name, n=int(m.sum()), cases=int(y.sum()), prevalence=y.mean(),
                             roc_auc=met["roc_auc"], roc_auc_lo=lo, roc_auc_hi=hi, pr_auc=met["pr_auc"],
                             observed_over_expected=y.mean() / p.mean(), ece=met["ece"],
                             sensitivity=met["sensitivity"], specificity=met["specificity"], ppv=met["ppv"],
                             fp_per_tp=met["fp_per_tp"]))
    sub = pd.DataFrame(rows)
    sub.to_csv(rpa.OUT / "subgroup_performance.csv", index=False)
    print(sub[sub.model == "xgb_pre"].round(3).to_string(index=False), flush=True)

    # residual post-event signal check
    tr, ca, th = (d.iloc[i].reset_index(drop=True) for i in (tr_i, ca_i, th_i))
    y_te = te[rpa.TARGET].values
    full = [c for c in d.columns if c != rpa.TARGET]
    pre = [c for c in full if c not in rpa.POST]
    params = rpa.stage2_params()
    variants = {"pre_event_37": pre,
                "minus_insurance": [c for c in pre if c != "health_insurance"],
                "minus_insurance_income_home": [c for c in pre if c not in ("health_insurance", "income", "home_ownership")]}
    preds, out = {}, []
    for name, feats in variants.items():
        m = rpa.Model("xgb", feats, params).fit(tr, ca, th)
        p = m.predict(te); preds[name] = p
        met = rpa.metrics(y_te, p, m.thr["primary"])
        out.append(dict(variant=name, n_features=len(feats)) | {k: met[k] for k in
                   ("threshold", "roc_auc", "pr_auc", "brier", "ece", "sensitivity", "specificity", "ppv", "fp_per_tp")})
        print(f"{name}: n={len(feats)} ROC={met['roc_auc']:.4f} PR={met['pr_auc']:.4f} "
              f"sens={met['sensitivity']:.3f} ppv={met['ppv']:.3f}", flush=True)
    rng = np.random.default_rng(rpa.SEED)
    for name in ("minus_insurance", "minus_insurance_income_home"):
        da, dp = [], []
        for _ in range(1000):
            i = rng.integers(0, len(y_te), len(y_te))
            da.append(roc_auc_score(y_te[i], preds[name][i]) - roc_auc_score(y_te[i], preds["pre_event_37"][i]))
            dp.append(average_precision_score(y_te[i], preds[name][i]) - average_precision_score(y_te[i], preds["pre_event_37"][i]))
        r = next(o for o in out if o["variant"] == name)
        r["d_roc_auc"] = r["roc_auc"] - out[0]["roc_auc"]
        r["d_roc_lo"], r["d_roc_hi"] = np.percentile(da, [2.5, 97.5])
        r["d_pr_auc"] = r["pr_auc"] - out[0]["pr_auc"]
        r["d_pr_lo"], r["d_pr_hi"] = np.percentile(dp, [2.5, 97.5])
    res = pd.DataFrame(out)
    res.to_csv(rpa.OUT / "residual_signal_check.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
