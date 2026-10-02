"""Post-event predictor experiment (1) and evaluation-optimism audit (2).

Same split, preprocessing, tuned parameters, calibration choice and threshold rule as the thesis
code. With seed 42 and all 55 predictors it must reproduce the thesis test result (ROC-AUC 0.8326,
threshold 0.0866); this is checked.

1: refit without Tier A, A+B and A+B+C, with paired-bootstrap changes in ROC-AUC and PR-AUC.
2: common-practice pipelines (threshold 0.5, threshold tuned on the test split, SMOTE before
   splitting, SMOTE on training only) report on their own split and are then scored on the
   untouched test split. BRFSS 2023 and the Kaggle file.

    python run_robustness_experiments.py [--quick] [--exp 1|2|all]    (--quick: seed 42 only)
"""

import os
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as PlattScaler
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# config
HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
STAGE2 = PROJ / "simillar reasearch test" / "rich_feature_best_4way_final_search"
# CMPB_DATA=v1 uses the original (uncorrected) cleaning
DATA_2023 = (STAGE2 / "outputs" / "data_cache" if os.environ.get("CMPB_DATA", "v2") == "v1"
             else HERE.parent / "analysis" / "outputs" / "data_cache_v2") / "brfss_2023_stroke_rich_clean.parquet"
DATA_2023_CSV = PROJ / "seprate experiment" / "outputs" / "data" / "processed" / "brfss_2023_stroke_rich_clean.csv"
PARAMS_CSV = STAGE2 / "outputs" / "tables" / "T04_optuna_best_params.csv"
KAGGLE_RAW = PROJ / "data" / "raw" / "kaggle" / "healthcare-dataset-stroke-data.csv"
OUT = HERE / "outputs"
TARGET = "stroke"

# thesis result, for the sanity check
STAGE2_REF = dict(roc_auc=0.8326, pr_auc=0.1881, threshold=0.0866, f2=0.3779)

# Column names are the renamed ones in brfss_2023_stroke_rich_clean
# (BRFSS 2023 source variable in brackets).
TIER_A = [  # functional / sensory sequelae - strongest reverse causation
    "walking_difficulty",             # DIFFWALK
    "dressing_difficulty",            # DIFFDRES
    "independent_living_difficulty",  # DIFFALON
    "cognitive_difficulty",           # DECIDE
    "blind",                          # BLIND
]
TIER_B = [  # downstream health status, treatment, secondary prevention
    "general_health",        # GENHLTH
    "physical_health_days",  # PHYSHLTH
    "mental_health_days",    # MENTHLTH
    "poor_health_days",      # POORHLTH
    "employment",            # EMPLOY1
    "depression",            # ADDEPEV3
    "cholesterol_meds",      # CHOLMED3 (statins started after stroke)
    "exercise_any",          # EXERANY2
    "deaf",                  # DEAF (weak sequela; kept for conservatism)
]
TIER_C = [  # healthcare contact that typically increases after a stroke
    "recent_checkup",        # CHECKUP1
    "personal_doctor",       # PERSDOC3
    "flu_vaccine",           # FLUSHOT7
    "pneumonia_vaccine",     # PNEUVAC4
]
# Kept as pre-event-plausible: age, sex, race, education, income, marital,
# smoking, alcohol, BMI, diabetes, hypertension, cholesterol, MI/CHD, kidney,
# COPD, cancer, arthritis, ... Hypertension/diabetes can still be diagnosed
# after a stroke: report as a residual limitation.

SEEDS = [42, 7, 123, 2024, 99]   # 42 = primary split
KAGGLE_SEEDS = list(range(20))   # Kaggle is small: more repeats, cheap
N_BOOT = 1000

# logging
OUT.mkdir(parents=True, exist_ok=True)
if hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(OUT / "run_log.txt", mode="a", encoding="utf-8")])
log = logging.getLogger("robust")


# pipeline (as in the thesis code)
def load_brfss():
    if DATA_2023.exists():
        df = pd.read_parquet(DATA_2023)
    else:
        df = pd.read_csv(DATA_2023_CSV, low_memory=False)
    log.info(f"BRFSS 2023: {len(df):,} rows, {df.shape[1]-1} features, "
             f"prevalence {df[TARGET].mean():.4f}")
    return df


def load_kaggle():
    df = pd.read_csv(KAGGLE_RAW, na_values=["N/A"]).drop(columns=["id"])
    for c in ("hypertension", "heart_disease"):
        df[c] = df[c].astype(float)
    log.info(f"Kaggle/DataPort: {len(df):,} rows, prevalence {df[TARGET].mean():.4f}")
    return df


def stage2_params():
    p = pd.read_csv(PARAMS_CSV).set_index("label").loc["MODE_A_R0"]
    keys = ["max_depth", "learning_rate", "subsample", "colsample_bytree",
            "min_child_weight", "gamma", "reg_alpha", "reg_lambda", "scale_pos_weight"]
    out = {k: float(p[k]) for k in keys}
    out["max_depth"] = int(out["max_depth"]); out["min_child_weight"] = int(out["min_child_weight"])
    return out


def four_way_split(df, seed):
    y = df[TARGET].values
    idx = np.arange(len(df))
    idx_rest, idx_test = train_test_split(idx, test_size=0.20, stratify=y, random_state=seed)
    idx_tc, idx_thr = train_test_split(idx_rest, test_size=0.125, stratify=y[idx_rest], random_state=seed)
    idx_tr, idx_cal = train_test_split(idx_tc, test_size=1 / 7, stratify=y[idx_tc], random_state=seed)
    take = lambda i: df.iloc[i].reset_index(drop=True)
    return take(idx_tr), take(idx_cal), take(idx_thr), take(idx_test)


def build_preprocessor(df_fit, feats):
    cat = [c for c in feats if not pd.api.types.is_numeric_dtype(df_fit[c].dtype)]
    num = [c for c in feats if c not in cat]
    tr = []
    if num:
        tr.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                    ("sc", StandardScaler())]), num))
    if cat:
        tr.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                    ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first",
                                                          dtype=np.float32, sparse_output=False))]), cat))
    return ColumnTransformer(tr, remainder="drop", sparse_threshold=0).fit(df_fit[feats])


def xform(ct, df, feats):
    return ct.transform(df[feats]).astype(np.float32)


def train_xgb(X_tr, y_tr, X_cal, y_cal, params, seed):
    m = XGBClassifier(**params, n_estimators=2000, tree_method="hist", random_state=seed,
                      eval_metric="aucpr", early_stopping_rounds=50, verbosity=0, n_jobs=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)], verbose=False)
    return m


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); n = len(y); tot = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.any():
            tot += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(tot)


def select_calibration(raw_cal, y_cal):
    """Sigmoid vs isotonic on the calibration split; keep the lower Brier."""
    platt = PlattScaler(max_iter=2000, C=1.0).fit(raw_cal.reshape(-1, 1), y_cal)
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, y_cal)
    sig = lambda r: platt.predict_proba(r.reshape(-1, 1))[:, 1]
    isf = lambda r: np.clip(iso.predict(r), 0.0, 1.0)
    if brier_score_loss(y_cal, sig(raw_cal)) <= brier_score_loss(y_cal, isf(raw_cal)):
        return sig, "sigmoid"
    return isf, "isotonic"


THR_GRID = np.linspace(0.005, 0.70, 1500)


def _grid_counts(y, p):
    yp = p[np.newaxis, :] >= THR_GRID[:, np.newaxis]
    pos = float(y.sum())
    tp = yp[:, y == 1].sum(axis=1).astype(float)
    fp = yp[:, y == 0].sum(axis=1).astype(float)
    rec = tp / (pos + 1e-9); prec = tp / (tp + fp + 1e-9)
    return rec, prec


def thr_prec_at_rec(y, p, target=0.60):
    """Highest precision among grid thresholds with recall >= target."""
    rec, prec = _grid_counts(y, p)
    f2 = 5 * prec * rec / (4 * prec + rec + 1e-9)
    ok = np.where(rec >= target)[0]
    if not len(ok):
        return float(THR_GRID[np.argmax(f2)])
    return float(THR_GRID[ok[np.argmax(prec[ok])]])


def thr_max_f1(y, p):
    rec, prec = _grid_counts(y, p)
    return float(THR_GRID[np.argmax(2 * prec * rec / (prec + rec + 1e-9))])


def metrics(y, p, t):
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
    return dict(threshold=t, prevalence=y.mean(), accuracy=(tp + tn) / len(y),
                recall=rec, precision=prec, specificity=tn / max(tn + fp, 1),
                f1=2 * prec * rec / max(prec + rec, 1e-12),
                f2=5 * prec * rec / max(4 * prec + rec, 1e-12),
                roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                brier=brier_score_loss(y, p), ece=ece(y, p),
                fp_per_tp=fp / max(tp, 1), tp=tp, fp=fp, tn=tn, fn=fn)


def fit_rigorous(tr, cal, thr, feats, params, seed):
    """Paper pipeline: fit on train, calibrate on cal, prec_at_rec60 on thresh."""
    ct = build_preprocessor(tr, feats)
    Xtr, Xcal, Xthr = xform(ct, tr, feats), xform(ct, cal, feats), xform(ct, thr, feats)
    ycal = cal[TARGET].values
    model = train_xgb(Xtr, tr[TARGET].values, Xcal, ycal, params, seed)
    calib, method = select_calibration(model.predict_proba(Xcal)[:, 1], ycal)
    predict = lambda d: calib(model.predict_proba(xform(ct, d, feats))[:, 1])
    t = thr_prec_at_rec(thr[TARGET].values, predict(thr))
    return predict, t, method, model.best_iteration


def paired_bootstrap_delta(y, p_ref, p_alt, seed):
    rng = np.random.default_rng(seed); n = len(y); d_auc, d_pr = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        d_auc.append(roc_auc_score(y[i], p_alt[i]) - roc_auc_score(y[i], p_ref[i]))
        d_pr.append(average_precision_score(y[i], p_alt[i]) - average_precision_score(y[i], p_ref[i]))
    return (np.percentile(d_auc, [2.5, 97.5]), np.percentile(d_pr, [2.5, 97.5]))


# Experiment 1
def experiment_1(df, seeds):
    feats = [c for c in df.columns if c != TARGET]
    for tier in (TIER_A, TIER_B, TIER_C):
        miss = [c for c in tier if c not in feats]
        if miss:
            raise SystemExit(f"[Exp1] tier columns not in data: {miss}")
    params = stage2_params()
    drop = lambda cols: [f for f in feats if f not in cols]
    variants = {
        "full_55": feats,
        "minus_A": drop(TIER_A),
        "minus_AB": drop(TIER_A + TIER_B),
        "minus_ABC": drop(TIER_A + TIER_B + TIER_C),
    }
    rows, deltas = [], []
    for seed in seeds:
        t0 = time.time()
        tr, cal, thr, te = four_way_split(df, seed)
        y_te = te[TARGET].values; preds = {}
        for name, fs in variants.items():
            predict, t, method, n_iter = fit_rigorous(tr, cal, thr, fs, params, seed)
            p = predict(te); preds[name] = p
            m = metrics(y_te, p, t)
            rows.append(dict(seed=seed, variant=name, n_features=len(fs),
                             calibration=method, best_iteration=n_iter, **m))
            log.info(f"[Exp1] seed={seed} {name:10s} n={len(fs):2d} {method:8s} "
                     f"ROC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f} t={t:.4f} "
                     f"rec={m['recall']:.3f} prec={m['precision']:.3f} F2={m['f2']:.4f}")
            if seed == 42 and name == "full_55":
                ok = (abs(m["roc_auc"] - STAGE2_REF["roc_auc"]) < 0.002
                      and abs(m["pr_auc"] - STAGE2_REF["pr_auc"]) < 0.003)
                log.info(f"[Exp1] CHECK vs published Stage 2 on uncorrected data "
                         f"(ROC {STAGE2_REF['roc_auc']}, PR {STAGE2_REF['pr_auc']}, "
                         f"t {STAGE2_REF['threshold']}, F2 {STAGE2_REF['f2']}): "
                         f"{'within tolerance' if ok else 'differs (expected with CMPB_DATA=v2; must match with CMPB_DATA=v1)'}")
        # paired bootstrap on the primary split only; other seeds give split-to-split spread
        for name in (("minus_A", "minus_AB", "minus_ABC") if seed == 42 else ()):
            (alo, ahi), (plo, phi) = paired_bootstrap_delta(y_te, preds["full_55"], preds[name], seed)
            deltas.append(dict(seed=seed, variant=name,
                               d_roc_auc=roc_auc_score(y_te, preds[name]) - roc_auc_score(y_te, preds["full_55"]),
                               d_roc_ci_lo=alo, d_roc_ci_hi=ahi,
                               d_pr_auc=average_precision_score(y_te, preds[name]) - average_precision_score(y_te, preds["full_55"]),
                               d_pr_ci_lo=plo, d_pr_ci_hi=phi))
        log.info(f"[Exp1] seed {seed} done in {(time.time()-t0)/60:.1f} min")
        pd.DataFrame(rows).to_csv(OUT / "exp1_per_seed.csv", index=False)
        pd.DataFrame(deltas).to_csv(OUT / "exp1_delta_bootstrap.csv", index=False)

    res = pd.DataFrame(rows)
    cols = ["n_features", "roc_auc", "pr_auc", "brier", "ece", "recall", "precision",
            "specificity", "f2", "fp_per_tp"]
    summ = res.groupby("variant", sort=False)[cols].agg(["mean", "std"]).round(4)
    summ.to_csv(OUT / "exp1_summary.csv")
    log.info("\n=== Experiment 1 summary (mean, std over seeds) ===\n" + summ.to_string())
    log.info("\n=== Paired-bootstrap deltas vs full_55 ===\n"
             + pd.DataFrame(deltas).round(4).to_string(index=False))


# Experiment 2
def default_xgb(seed):
    """Untuned default XGBoost, no class weighting - typical of the literature."""
    return XGBClassifier(tree_method="hist", random_state=seed, verbosity=0, n_jobs=-1)


def experiment_2(df, seeds, dataset, rigorous_params):
    feats = [c for c in df.columns if c != TARGET]
    rows = []
    for seed in seeds:
        tr, cal, thr, audit = four_way_split(df, seed)
        y_aud = audit[TARGET].values

        predict, t, _, _ = fit_rigorous(tr, cal, thr, feats, rigorous_params, seed)
        rows.append(dict(dataset=dataset, seed=seed, pipeline="R: paper pipeline",
                         kind="actual", **metrics(y_aud, predict(audit), t)))

        dev = pd.concat([tr, cal, thr], ignore_index=True)
        ct = build_preprocessor(dev, feats)           # fitted before splitting (common)
        X_dev, y_dev = xform(ct, dev, feats), dev[TARGET].values
        X_aud = xform(ct, audit, feats)
        sm = SMOTE(random_state=seed)

        configs = [  # name, oversampling, threshold rule
            ("C1: t=0.5, no calibration", "none", "fixed"),
            ("C2: C1 + threshold tuned on test", "none", "test_f1"),
            ("C3: SMOTE before split, t=0.5", "before", "fixed"),
            ("C4: C3 + threshold tuned on test", "before", "test_f1"),
            ("C5 (control): SMOTE on train only, t=0.5", "train_only", "fixed"),
        ]
        for name, over, rule in configs:
            if over == "before":
                Xs, ys = sm.fit_resample(X_dev, y_dev)
                X_tr, X_it, y_tr, y_it = train_test_split(Xs, ys, test_size=0.2, stratify=ys, random_state=seed)
            else:
                X_tr, X_it, y_tr, y_it = train_test_split(X_dev, y_dev, test_size=0.2, stratify=y_dev, random_state=seed)
                if over == "train_only":
                    X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            clf = default_xgb(seed).fit(X_tr, y_tr)
            p_it, p_aud = clf.predict_proba(X_it)[:, 1], clf.predict_proba(X_aud)[:, 1]
            t = 0.5 if rule == "fixed" else thr_max_f1(y_it, p_it)
            rows.append(dict(dataset=dataset, seed=seed, pipeline=name, kind="reported", **metrics(y_it, p_it, t)))
            rows.append(dict(dataset=dataset, seed=seed, pipeline=name, kind="actual", **metrics(y_aud, p_aud, t)))
        log.info(f"[Exp2:{dataset}] seed {seed} done")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"exp2_{dataset}_per_seed.csv", index=False)
    cols = ["prevalence", "accuracy", "recall", "precision", "specificity", "f1", "f2",
            "roc_auc", "pr_auc", "brier", "ece"]
    summ = res.groupby(["pipeline", "kind"])[cols].mean().round(4)
    summ.to_csv(OUT / f"exp2_{dataset}_summary.csv")
    rep = summ.xs("reported", level="kind")
    act = summ.xs("actual", level="kind").loc[rep.index]
    opt = (rep - act).drop(columns="prevalence").add_prefix("optimism_")
    opt.to_csv(OUT / f"exp2_{dataset}_optimism.csv")
    log.info(f"\n=== Experiment 2 [{dataset}]: reported vs actual (mean over seeds) ===\n{summ.to_string()}")
    log.info(f"\n=== Optimism = reported - actual [{dataset}] ===\n{opt.round(4).to_string()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="seed 42 only")
    ap.add_argument("--exp", default="all", choices=["1", "2", "all"])
    a = ap.parse_args()
    seeds = [42] if a.quick else SEEDS
    log.info(f"===== run start: exp={a.exp} seeds={seeds} =====")
    if a.exp in ("1", "all"):
        experiment_1(load_brfss(), seeds)
    if a.exp in ("2", "all"):
        experiment_2(load_brfss(), seeds, "brfss2023", stage2_params())
        # Kaggle: no tuned parameters, so a neutral weighted setup for R
        kp = dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
                  scale_pos_weight=19.5)
        experiment_2(load_kaggle(), [42] if a.quick else KAGGLE_SEEDS, "kaggle", kp)
    log.info("===== run end =====")
