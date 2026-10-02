"""Logistic regression with the nominal items one-hot encoded instead of integer-coded.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_paper_analysis as rpa  # noqa: E402

NOMINAL = ["race_ethnicity", "employment", "marital", "home_ownership", "health_insurance", "personal_doctor",
           "recent_checkup", "current_smoking_detail", "smokeless_tobacco", "ecigarette_use"]


def main():
    d = pd.read_parquet(rpa.CACHE / "brfss_2023_stroke_rich_clean.parquet")
    for c in NOMINAL:
        d[c] = d[c].map(lambda v: np.nan if pd.isna(v) else str(int(v))).astype(object)
    full = [c for c in d.columns if c != rpa.TARGET]
    pre = [c for c in full if c not in rpa.POST]
    tr_i, ca_i, th_i, te_i = rpa.four_way_split(d)
    tr, ca, th, te = (d.iloc[i].reset_index(drop=True) for i in (tr_i, ca_i, th_i, te_i))
    y = te[rpa.TARGET].values
    rows = []
    for key, feats in (("lr_full_onehot", full), ("lr_pre_onehot", pre)):
        m = rpa.Model("lr", feats, None).fit(tr, ca, th)
        p = m.predict(te); t = m.thr["primary"]
        r = dict(model=key, n_features=len(feats), calibration=m.cal_name) | rpa.metrics(y, p, t) | rpa.boot_ci(y, p, t, rpa.N_BOOT)
        rows.append(r)
        print(f"{key}: ROC={r['roc_auc']:.4f} ({r['roc_auc_lo']:.4f}-{r['roc_auc_hi']:.4f}) PR={r['pr_auc']:.4f} "
              f"sens={r['sensitivity']:.3f} spec={r['specificity']:.3f} ppv={r['ppv']:.3f} ECE={r['ece']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(rpa.OUT / "lr_onehot_check.csv", index=False)


if __name__ == "__main__":
    main()
