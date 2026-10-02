"""Rebuild the BRFSS analytic files with corrected missing-value codes.

The original cleaning set 7/9 (77/99, ...) to missing for every item, which is wrong where these are
valid answers: INCOME3 (1-11), EMPLOY1 (7 = retired), PRIMINS1/2 (1-10, 88 = none) and the drink
counts AVEDRNK3/4 and MAXDRNKS. ALCDAY4 is also converted from its 3-digit code (1xx per week,
2xx per month, 888 none) to drinking days per month. All other columns are copied unchanged.
Output: outputs/data_cache_v2/
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
XPT = PROJ / "seprate experiment"
CACHE = PROJ / "simillar reasearch test" / "rich_feature_best_4way_final_search" / "outputs" / "data_cache"
OUT = Path(__file__).resolve().parent / "outputs"
V2 = OUT / "data_cache_v2"
V2.mkdir(parents=True, exist_ok=True)
PROCESSED = XPT / "outputs" / "data" / "processed"  # written by seprate experiment/scripts/run_separate_brfss_experiment.py


def stage2_cache(year):
    """Original processed CSV (float64 columns stored as float32); built if missing."""
    path = CACHE / f"brfss_{year}_stroke_rich_clean.parquet"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(PROCESSED / f"brfss_{year}_stroke_rich_clean.csv", low_memory=False)
        for c in df.select_dtypes(include="float64").columns:
            df[c] = df[c].astype("float32")
        df.to_parquet(path, index=False)
    return pd.read_parquet(path)

RAW = {"income": ("INCOME3",), "employment": ("EMPLOY1",), "health_insurance": ("PRIMINS1", "PRIMINS2"),
       "average_drinks": ("AVEDRNK3", "AVEDRNK4"), "max_drinks": ("MAXDRNKS",), "alcohol_days_code": ("ALCDAY4",)}


def valid(s, lo, hi, extra=()):
    s = pd.to_numeric(s, errors="coerce")
    ok = s.between(lo, hi) | s.isin(extra)
    return s.where(ok)


def clean(col, raw_name, s):
    s = pd.to_numeric(s, errors="coerce")
    if col == "income":
        return valid(s, 1, 11)
    if col == "employment":
        return valid(s, 1, 8)
    if col == "health_insurance":
        return valid(s, 1, 10, extra=(88,))
    if col in ("average_drinks", "max_drinks"):
        out = valid(s, 1, 76)
        return out.mask(s == 88, 0.0)  # 88 = none (defined for AVEDRNK; absent for MAXDRNKS)
    if col == "alcohol_days_code":
        out = pd.Series(np.nan, index=s.index)
        wk, mo = s.between(101, 107), s.between(201, 230)
        out[wk] = (s[wk] - 100) * 30.0 / 7.0
        out[mo] = s[mo] - 200
        out[s == 888] = 0.0
        return out
    raise KeyError(col)


summary = []
for year in (2023, 2024):
    old = stage2_cache(year)
    parts = []
    for chunk in pd.read_sas(XPT / f"LLCP{year}.XPT", format="xport", encoding="utf-8", chunksize=100_000):
        want = ["CVDSTRK3", "_AGE80"] + [c for v in RAW.values() for c in v]
        parts.append(chunk[[c for c in want if c in chunk.columns]])
    raw = pd.concat(parts, ignore_index=True)
    raw = raw[raw["CVDSTRK3"].isin([1, 2])].reset_index(drop=True)
    assert len(raw) == len(old)
    assert np.array_equal((raw["CVDSTRK3"] == 1).astype(int).values, old["stroke"].values)
    age = raw["_AGE80"].where(raw["_AGE80"].between(18, 80))
    assert np.allclose(age.fillna(-1).values, old["age"].astype(float).fillna(-1).values)

    new = old.copy()
    for col, cands in RAW.items():
        rn = next(c for c in cands if c in raw.columns)
        vc = raw[rn].value_counts(dropna=False).sort_index()
        print(f"{year} {rn}: raw codes " + ", ".join(f"{int(k) if k == k else 'NaN'}:{v}" for k, v in vc.items()
                                                     if not (k == k) or k < 12 or k in (77, 88, 99, 777, 888, 999)))
        new[col] = clean(col, rn, raw[rn]).astype("float32").values
        changed = int((old[col].fillna(-999).values != new[col].fillna(-999).values).sum())
        summary.append(dict(year=year, feature=col, raw_variable=rn, missing_v1=int(old[col].isna().sum()),
                            missing_v2=int(new[col].isna().sum()), values_changed=changed))
        print(f"   {col}: missing {old[col].isna().sum():,} -> {new[col].isna().sum():,}; values changed {changed:,}")
    untouched = [c for c in old.columns if c not in RAW]
    assert all(old[c].equals(new[c]) for c in untouched)
    new.to_parquet(V2 / f"brfss_{year}_stroke_rich_clean.parquet", index=False)
    print(f"wrote {V2 / f'brfss_{year}_stroke_rich_clean.parquet'}")

pd.DataFrame(summary).to_csv(OUT / "data_fix_summary.csv", index=False)
print(pd.DataFrame(summary).to_string(index=False))
