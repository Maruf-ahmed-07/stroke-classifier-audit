"""Recover the BRFSS final weights (_LLCPWT) for the analytic files.

The analytic files keep the XPT rows with CVDSTRK3 in {1, 2} in their original order, so the same
filter on the XPT gives row-aligned weights. Alignment is checked on stroke, age and sex.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
XPT = PROJ / "seprate experiment"
CACHE = PROJ / "simillar reasearch test" / "rich_feature_best_4way_final_search" / "outputs" / "data_cache"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

for year in (2023, 2024):
    cols = ["CVDSTRK3", "_LLCPWT", "_AGE80", "_SEX", "_STSTR", "_PSU"]
    parts = []
    for chunk in pd.read_sas(XPT / f"LLCP{year}.XPT", format="xport", encoding="utf-8", chunksize=100_000):
        parts.append(chunk[[c for c in cols if c in chunk.columns]])
    raw = pd.concat(parts, ignore_index=True)
    raw = raw[raw["CVDSTRK3"].isin([1, 2])].reset_index(drop=True)

    rich = pd.read_parquet(CACHE / f"brfss_{year}_stroke_rich_clean.parquet", columns=["stroke", "age", "sex"])
    assert len(raw) == len(rich), f"{year}: row count {len(raw)} vs {len(rich)}"
    stroke_ok = np.array_equal((raw["CVDSTRK3"] == 1).astype(int).values, rich["stroke"].values)
    age_raw = raw["_AGE80"].where((raw["_AGE80"] >= 18) & (raw["_AGE80"] <= 80))
    age_ok = np.allclose(age_raw.fillna(-1).values, rich["age"].astype(float).fillna(-1).values)
    sex_ok = (pd.to_numeric(rich["sex"], errors="coerce").fillna(-1).values == raw["_SEX"].fillna(-1).values).mean()
    print(f"{year}: rows={len(raw):,} stroke_match={stroke_ok} age_match={age_ok} sex_match={sex_ok:.4f}")
    assert stroke_ok and age_ok and sex_ok > 0.999, "alignment failed - do not use weights"
    raw[["_LLCPWT", "_STSTR", "_PSU"]].to_parquet(OUT / f"brfss_{year}_weights_aligned.parquet", index=False)
    print(f"  wrote brfss_{year}_weights_aligned.parquet")
