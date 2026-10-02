"""
Score respondents with the released pre-event stroke-status model.

Usage
    python predict.py input.csv output.csv      # adds columns risk, flag_broad, flag_primary, flag_focused
    python predict.py --check                   # reproduces model/check_predictions.csv

The input CSV needs the 37 predictor columns listed in model/preprocessing.json, coded as in
the cleaned BRFSS 2023 file (see MODEL_CARD.md). Missing values are allowed and are imputed
with the training-set medians / most frequent categories, exactly as in the paper.
Requires only numpy, pandas and xgboost; no pickled scikit-learn objects are used.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

M = Path(__file__).resolve().parent / "model"


def load():
    spec = json.loads((M / "preprocessing.json").read_text())
    booster = xgb.Booster()
    booster.load_model(str(M / "xgb_pre_event.json"))
    meta = json.loads((M / "model_meta.json").read_text())
    cal = pd.read_csv(M / "calibration_isotonic.csv")
    thr = pd.read_csv(M / "thresholds.csv").set_index("tier")["threshold"]
    return spec, booster, meta, cal, thr


def design_matrix(df, spec):
    cols = []
    for c in spec["numeric"]:
        x = pd.to_numeric(df[c["name"]], errors="coerce").fillna(c["impute_median"]).to_numpy(float)
        cols.append(((x - c["mean"]) / c["scale"])[:, None])
    for c in spec["categorical"]:
        x = df[c["name"]].astype("object").where(df[c["name"]].notna(), c["impute_most_frequent"]).astype(str)
        cols.append(np.stack([(x == lv).to_numpy(float) for lv in c["categories"][1:]], axis=1))
    X = np.hstack(cols).astype(np.float32)
    assert X.shape[1] == len(spec["output_columns"])
    return X


def predict(df):
    spec, booster, meta, cal, thr = load()
    X = design_matrix(df, spec)
    raw = booster.predict(xgb.DMatrix(X), iteration_range=(0, meta["best_iteration"] + 1))
    risk = np.clip(np.interp(raw, cal["raw_score"], cal["calibrated_risk"]), 0, 1)
    out = pd.DataFrame({"risk": risk})
    for tier in ("broad", "primary", "focused"):
        out[f"flag_{tier}"] = (risk >= thr[tier]).astype(int)
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        ref = pd.read_csv(M / "check_predictions.csv")
        got = predict(pd.read_csv(M / "check_inputs.csv"))["risk"].to_numpy()
        err = np.abs(got - ref["xgb_pre"].to_numpy()).max()
        print(f"max absolute difference vs the paper pipeline on {len(ref):,} test rows: {err:.2e}")
        sys.exit(0 if err < 1e-5 else 1)
    inp, outp = sys.argv[1], sys.argv[2]
    df = pd.read_csv(inp)
    pd.concat([df, predict(df)], axis=1).to_csv(outp, index=False)
    print(f"scored {len(df):,} rows -> {outp}")
