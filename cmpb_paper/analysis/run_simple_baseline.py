"""Six-predictor logistic regression (age, sex, hypertension, diabetes, MI/CHD, smoking) run
through the same pipeline and compared with the pre-event XGBoost model by paired bootstrap.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_paper_analysis as rpa  # noqa: E402
from run_v2_analyses import nb  # noqa: E402

SIMPLE = ["age", "sex", "hypertension_status", "diabetes_status", "computed_michd", "smoking_status"]


def main():
    d = pd.read_parquet(rpa.CACHE / "brfss_2023_stroke_rich_clean.parquet")
    tr_i, ca_i, th_i, te_i = rpa.four_way_split(d)
    tr, ca, th, te = (d.iloc[i].reset_index(drop=True) for i in (tr_i, ca_i, th_i, te_i))
    y = te[rpa.TARGET].values
    m = rpa.Model("lr", SIMPLE, rpa.stage2_params()).fit(tr, ca, th)
    p_simple = m.predict(te)
    pr = pd.read_parquet(rpa.OUT / "test_predictions_2023.parquet")
    assert np.array_equal(pr["y"].values, y)
    p_xgb = pr["xgb_pre"].values

    def stats(yy, a, b):
        return dict(auc_simple=roc_auc_score(yy, a), auc_xgb=roc_auc_score(yy, b),
                    ap_simple=average_precision_score(yy, a), ap_xgb=average_precision_score(yy, b),
                    brier_simple=np.mean((a - yy) ** 2), brier_xgb=np.mean((b - yy) ** 2),
                    nb05_simple=nb(yy, a, 0.05), nb05_xgb=nb(yy, b, 0.05),
                    nb08_simple=nb(yy, a, 0.08), nb08_xgb=nb(yy, b, 0.08))

    point = stats(y, p_simple, p_xgb)
    rng = np.random.default_rng(rpa.SEED); reps = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        reps.append(stats(y[i], p_simple[i], p_xgb[i]))
    reps = pd.DataFrame(reps)
    rows = []
    for k in ("auc", "ap", "brier", "nb05", "nb08"):
        for who in ("simple", "xgb"):
            c = f"{k}_{who}"
            rows.append(dict(quantity=c, estimate=point[c], lo=reps[c].quantile(.025), hi=reps[c].quantile(.975)))
        dd = reps[f"{k}_xgb"] - reps[f"{k}_simple"]
        rows.append(dict(quantity=f"delta_{k}_xgb_minus_simple", estimate=point[f"{k}_xgb"] - point[f"{k}_simple"],
                         lo=dd.quantile(.025), hi=dd.quantile(.975)))
    met = rpa.metrics(y, p_simple, m.thr["primary"])
    rows += [dict(quantity=f"simple_{k}", estimate=met[k]) for k in
             ("threshold", "sensitivity", "specificity", "ppv", "fp_per_tp", "cal_slope")]
    out = pd.DataFrame(rows)
    out.to_csv(rpa.OUT / "simple_baseline.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
