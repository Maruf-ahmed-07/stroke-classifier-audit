"""External validation in NHIS 2012 and 2014-2018 (sample adults, linked mortality to 2019).

The 18 predictors asked in both surveys are recoded to the BRFSS coding. The harmonised model is
fitted on the BRFSS 2023 splits with the paper's pipeline and applied unchanged to NHIS, for
prevalent stroke (STREV) and for cerebrovascular death (UCOD_LEADING = 005) among eligible adults
without stroke at interview. The public files give no follow-up time, so death is a binary outcome.
NHIS files are downloaded into external_data/nhis/ if missing.
"""
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_paper_analysis as rpa  # noqa: E402
from run_v2_analyses import flex_stats  # noqa: E402

EXT = Path(__file__).resolve().parent / "external_data" / "nhis"
EXT.mkdir(parents=True, exist_ok=True)
OUT = rpa.OUT
YEARS = [2012, 2014, 2015, 2016, 2017, 2018]
ASCII_YEARS = {2012, 2014, 2015}  # CSV releases start in 2016
FTP = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS"
LOGF = open(OUT / "nhis_log.txt", "w", encoding="utf-8")
FEATS = ["age", "sex", "race_ethnicity", "education", "marital", "uninsured", "hypertension",
         "high_cholesterol", "diabetes", "heart_attack", "coronary_heart_disease", "asthma_ever",
         "cancer_any", "copd", "kidney_disease", "arthritis", "bmi", "smoking_status"]
SA_VARS = ["SRVY_YR", "HHX", "FMX", "FPX", "AGE_P", "SEX", "HISPAN_I", "RACERPI2", "R_MARITL", "HYPEV",
           "CHLEV", "DIBEV", "CHDEV", "MIEV", "AASMEV", "CANEV", "COPDEV", "KIDWKYR", "ARTH1", "BMI",
           "SMKSTAT2", "STREV", "WTFA_SA"]
P_VARS = ["SRVY_YR", "HHX", "FMX", "FPX", "EDUC1", "NOTCOV"]


def say(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True); LOGF.write(line + "\n"); LOGF.flush()


# download and read
def fetch(url, dest):
    if not dest.exists() or dest.stat().st_size < 5000:
        say(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def sas_layout(path):
    import re
    txt = path.read_text(encoding="latin-1")
    body = txt[re.search(r"^\s*INPUT\s*$", txt, flags=re.M).end():]
    body = body[:body.index(";")]
    return {m.group(1): (int(m.group(2)) - 1, int(m.group(3) or m.group(2)))
            for m in re.finditer(r"([A-Z_][A-Z0-9_]*)\s+\$?\s*(\d+)\s*(?:-\s*(\d+))?", body)}


def read_file(year, kind):
    """kind: 'samadult' or 'personsx'; returns the requested columns as strings."""
    want = SA_VARS if kind == "samadult" else P_VARS
    if year in ASCII_YEARS:
        z = fetch(f"{FTP}/Datasets/NHIS/{year}/{kind}.zip", EXT / f"{kind}_{year}.zip")
        sas = fetch(f"{FTP}/Program_Code/NHIS/{year}/{kind}.sas", EXT / f"{kind}_{year}.sas")
        lay = sas_layout(sas)
        if kind == "samadult" and "DIBEV" not in lay and "DIBEV1" in lay:
            lay["DIBEV"] = lay["DIBEV1"]
        cols = [c for c in want if c in lay]
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            csv = [n for n in names if n.lower().endswith(".csv")]
            if csv:  # the 2015 archives hold both a CSV and the fixed-width file
                with zf.open(csv[0]) as fh:
                    df = pd.read_csv(fh, dtype=str, low_memory=False)
                if "DIBEV" not in df.columns and "DIBEV1" in df.columns:
                    df["DIBEV"] = df["DIBEV1"]
                df = df[[c for c in want if c in df.columns]]
            else:
                with zf.open(next(n for n in names if n.lower().endswith(".dat"))) as fh:
                    df = pd.read_fwf(fh, colspecs=[lay[c] for c in cols], names=cols, dtype=str, header=None)
    else:
        z = fetch(f"{FTP}/Datasets/NHIS/{year}/{kind}csv.zip", EXT / f"{kind}csv_{year}.zip")
        df = pd.read_csv(z, dtype=str, low_memory=False)
        if "DIBEV" not in df.columns and "DIBEV1" in df.columns:
            df["DIBEV"] = df["DIBEV1"]
        df = df[[c for c in want if c in df.columns]]
    df = df.apply(lambda s: s.str.strip())
    df["PUBLICID"] = df["SRVY_YR"] + df["HHX"].str.zfill(6) + df["FMX"].str.zfill(2) + df["FPX"].str.zfill(2)
    return df


def read_mortality(year):
    f = fetch(f"{FTP}/datalinkage/linked_mortality/NHIS_{year}_MORT_2019_PUBLIC.dat",
              EXT / f"NHIS_{year}_MORT_2019_PUBLIC.dat")
    return pd.read_fwf(f, colspecs=[(0, 14), (14, 15), (15, 16), (16, 19)],
                       names=["PUBLICID", "ELIGSTAT", "MORTSTAT", "UCOD_LEADING"], dtype=str, header=None)


# harmonisation
def num(s):
    return pd.to_numeric(s, errors="coerce")


def yn(s):
    """NHIS / BRFSS 1 = yes, 2 = no -> 1/0; anything else missing."""
    s = num(s)
    return pd.Series(np.select([s == 1, s == 2], [1.0, 0.0], np.nan), index=s.index)


def nhis_features(df):
    out = pd.DataFrame(index=df.index)
    out["age"] = num(df["AGE_P"]).clip(upper=80)
    out["sex"] = num(df["SEX"]).where(num(df["SEX"]).isin([1, 2]))
    hisp, race = num(df["HISPAN_I"]), num(df["RACERPI2"])
    out["race_ethnicity"] = np.where(hisp.between(0, 11), 5, race.map({1: 1, 2: 2, 4: 3, 3: 4, 5: 6, 6: 6}))
    ed = num(df["EDUC1"])
    out["education"] = np.select([ed == 0, ed.between(1, 8), ed.between(9, 12), ed.between(13, 14),
                                  ed.between(15, 17), ed.between(18, 21)], [1, 2, 3, 4, 5, 6], np.nan)
    out["marital"] = num(df["R_MARITL"]).map({1: 1, 2: 1, 3: 1, 4: 3, 5: 2, 6: 4, 7: 5, 8: 6})
    out["uninsured"] = yn(df["NOTCOV"])  # NOTCOV 1 = not covered
    for new, old in (("hypertension", "HYPEV"), ("high_cholesterol", "CHLEV"), ("heart_attack", "MIEV"),
                     ("coronary_heart_disease", "CHDEV"), ("asthma_ever", "AASMEV"), ("cancer_any", "CANEV"),
                     ("copd", "COPDEV"), ("kidney_disease", "KIDWKYR"), ("arthritis", "ARTH1")):
        out[new] = yn(df[old])
    dib = num(df["DIBEV"])
    out["diabetes"] = np.select([dib == 1, dib.isin([2, 3])], [1.0, 0.0], np.nan)
    b = num(df["BMI"]) / 100
    out["bmi"] = b.where(b.between(10, 80))
    out["smoking_status"] = num(df["SMKSTAT2"]).map({1: "current_every_day", 2: "current_some_days",
                                                     3: "former", 4: "never"}).astype("object")
    return out


def brfss_features(d):
    out = pd.DataFrame(index=d.index)
    for c in ("age", "sex", "race_ethnicity", "education", "marital", "bmi", "heart_attack",
              "coronary_heart_disease", "asthma_ever", "copd", "kidney_disease"):
        out[c] = d[c].astype(float)
    ins = d["health_insurance"].astype(float)
    out["uninsured"] = np.where(ins.isna(), np.nan, (ins == 88).astype(float))
    hs = d["hypertension_status"]
    out["hypertension"] = np.where(hs.isna(), np.nan, (hs == "yes").astype(float))
    out["high_cholesterol"] = yn(d["cholesterol_status"])
    ds = d["diabetes_status"]
    out["diabetes"] = np.where(ds.isna(), np.nan, (ds == "diabetes").astype(float))
    sk, oc = d["skin_cancer"].astype(float), d["other_cancer"].astype(float)
    out["cancer_any"] = np.where((sk == 1) | (oc == 1), 1.0, np.where((sk == 0) & (oc == 0), 0.0, np.nan))
    out["arthritis"] = yn(d["arthritis"])
    out["smoking_status"] = d["smoking_status"].astype("object")
    out[rpa.TARGET] = d[rpa.TARGET].values
    return out


# evaluation
def evaluate(y, p, t, label, n_boot=500):
    rng = np.random.default_rng(rpa.SEED); b = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], p[i]))
    pred = p >= t
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
    slope, citl = rpa.cal_slope_intercept(y, p)
    fs, _ = flex_stats(y, p)
    r = dict(evaluation=label, n=len(y), events=int(y.sum()), prevalence=y.mean(),
             roc_auc=roc_auc_score(y, p), roc_auc_lo=np.percentile(b, 2.5), roc_auc_hi=np.percentile(b, 97.5),
             pr_auc=average_precision_score(y, p), mean_predicted=p.mean(), observed_over_expected=y.mean() / p.mean(),
             cal_intercept_citl=citl, cal_slope=slope, ici=fs["ici"], threshold=t,
             sensitivity=tp / y.sum(), specificity=1 - fp / (y == 0).sum(), ppv=tp / max(tp + fp, 1),
             flagged_share=pred.mean())
    say(f"{label}: n={r['n']:,} events={r['events']:,} ROC={r['roc_auc']:.3f} ({r['roc_auc_lo']:.3f}-"
        f"{r['roc_auc_hi']:.3f}) PR={r['pr_auc']:.3f} O/E={r['observed_over_expected']:.2f} "
        f"slope={slope:.2f} sens={r['sensitivity']:.3f} spec={r['specificity']:.3f} ppv={r['ppv']:.3f}")
    return r


def main():
    t0 = time.time()
    frames = []
    for y in YEARS:
        sa, pe, mo = read_file(y, "samadult"), read_file(y, "personsx"), read_mortality(y)
        d = sa.merge(pe.drop(columns=["SRVY_YR", "HHX", "FMX", "FPX"]), on="PUBLICID", how="left",
                     validate="one_to_one")
        d = d.merge(mo, on="PUBLICID", how="left", validate="one_to_one")
        say(f"NHIS {y}: sample adults={len(d):,} linked={d['ELIGSTAT'].notna().mean():.4f} "
            f"eligible={(d['ELIGSTAT'] == '1').sum():,} stroke deaths={(d['UCOD_LEADING'] == '005').sum():,}")
        frames.append(d)
    nh = pd.concat(frames, ignore_index=True)
    X_nh = nhis_features(nh)
    X_nh["survey_year"] = num(nh["SRVY_YR"])
    X_nh["prevalent_stroke"] = yn(nh["STREV"])
    X_nh["eligible"] = nh["ELIGSTAT"] == "1"
    X_nh["stroke_death"] = ((nh["MORTSTAT"] == "1") & (nh["UCOD_LEADING"] == "005")).astype(int)
    X_nh["any_death"] = (nh["MORTSTAT"] == "1").astype(int)
    miss = X_nh[FEATS].isna().mean().round(4)
    say("NHIS missing share per predictor:\n" + miss.to_string())

    d23 = pd.read_parquet(rpa.CACHE / "brfss_2023_stroke_rich_clean.parquet")
    B = brfss_features(d23)
    say("BRFSS missing share per predictor:\n" + B[FEATS].isna().mean().round(4).to_string())
    tr_i, ca_i, th_i, te_i = rpa.four_way_split(d23)
    tr, ca, th, te = (B.iloc[i].reset_index(drop=True) for i in (tr_i, ca_i, th_i, te_i))
    params = rpa.stage2_params()
    models = {"XGBoost": rpa.Model("xgb", FEATS, params).fit(tr, ca, th),
              "Logistic regression": rpa.Model("lr", FEATS, params).fit(tr, ca, th)}

    rows = []
    A = X_nh[X_nh["prevalent_stroke"].notna()].reset_index(drop=True)
    yA = A["prevalent_stroke"].astype(int).values
    Bm = X_nh[X_nh["eligible"] & (X_nh["prevalent_stroke"] == 0)].reset_index(drop=True)
    yB = Bm["stroke_death"].values
    for name, m in models.items():
        t = m.thr["primary"]
        rows.append(dict(model=name, calibration=m.cal_name) |
                    evaluate(te[rpa.TARGET].values, m.predict(te), t, "BRFSS 2023 test (internal)"))
        rows.append(dict(model=name, calibration=m.cal_name) |
                    evaluate(yA, m.predict(A), t, "NHIS prevalent stroke (external)"))
        pB = m.predict(Bm)
        rows.append(dict(model=name, calibration=m.cal_name) |
                    evaluate(yB, pB, t, "NHIS cerebrovascular death, no baseline stroke (external)"))
        for yr in YEARS:
            k = (Bm["survey_year"] == yr).values
            rows.append(dict(model=name, calibration=m.cal_name, survey_year=yr) |
                        evaluate(yB[k], pB[k], t, f"NHIS cerebrovascular death, {yr} cohort"))
    # context: age alone and death from any cause
    rows.append(dict(model="Age only") | dict(evaluation="NHIS prevalent stroke (external)",
                                                roc_auc=roc_auc_score(yA, A["age"].fillna(A["age"].median()))))
    rows.append(dict(model="Age only") | dict(evaluation="NHIS cerebrovascular death, no baseline stroke (external)",
                                                roc_auc=roc_auc_score(yB, Bm["age"].fillna(Bm["age"].median()))))
    rows.append(dict(model="Age only") | dict(evaluation="BRFSS 2023 test (internal)",
                                                roc_auc=roc_auc_score(te[rpa.TARGET], te["age"].fillna(te["age"].median()))))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "nhis_external_validation.csv", index=False)
    say("\n" + res[["model", "evaluation", "n", "events", "roc_auc", "roc_auc_lo", "roc_auc_hi", "pr_auc",
                    "observed_over_expected", "cal_slope", "sensitivity", "specificity", "ppv"]].round(3).to_string())
    pd.DataFrame(dict(feature=FEATS, nhis_missing=miss.values,
                      brfss_missing=B[FEATS].isna().mean().round(4).values)).to_csv(OUT / "nhis_harmonisation.csv",
                                                                                    index=False)
    say(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
