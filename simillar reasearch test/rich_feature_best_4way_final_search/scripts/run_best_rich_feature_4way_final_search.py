"""Rich-feature BRFSS stroke model: four-way split and Optuna search (thesis code).

Mode A: 2023 data with feature sets R0 (55), R1 (low missingness), R2 (52, also in 2024), R4, R5.
Mode B: trained on the 2023 splits and tested on all of 2024. Mode C: 2024 data, R2 features.
Split 60/10/10/20 (train / calibration / threshold / test). Platt or isotonic calibration, chosen by
Brier score on the calibration split; thresholds chosen on the threshold split, applied once to test.
"""
from __future__ import annotations

import gc
import json
import logging
import math
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as PlattScaler
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Paths
ROOT = Path(__file__).resolve().parents[1]            # rich_feature_best_4way_final_search/
PROJ = ROOT.parent.parent                              # stroke_reliability_thesis/
DATA_DIR = PROJ / "seprate experiment" / "outputs" / "data" / "processed"
DATA_2023 = DATA_DIR / "brfss_2023_stroke_rich_clean.csv"
DATA_2024 = DATA_DIR / "brfss_2024_stroke_rich_clean.csv"

OUT = ROOT / "outputs"
LOG_FILE = OUT / "logs" / "run_log.txt"

# Constants
SEED = 42
N_TRIALS_A_MAIN = 100   # R0 feature set, MODE A
N_TRIALS_A_ALT  = 60    # R1, R2 feature sets, MODE A
N_TRIALS_B      = 60    # MODE B cross-year
N_TRIALS_C      = 60    # Mode c 2024
N_BOOT          = 1000  # Bootstrap iterations
CHUNKSIZE       = 50_000

ONLY_2023 = {'hypertension_status', 'cholesterol_status', 'cholesterol_meds'}

# Cols that are numeric (not OHE-encoded)
NUMERIC_COLS = {
    'age', 'bmi', 'bmi_missing', 'children_count',
    'physical_health_days', 'mental_health_days', 'poor_health_days',
    'alcohol_days_code', 'average_drinks', 'max_drinks',
    'heart_attack', 'coronary_heart_disease', 'asthma_ever', 'skin_cancer',
    'other_cancer', 'copd', 'depression', 'kidney_disease', 'deaf', 'blind',
    'cognitive_difficulty', 'walking_difficulty', 'dressing_difficulty',
    'independent_living_difficulty', 'medical_cost_barrier', 'smoked_100',
}

THRESHOLD_POLICIES = [
    'F2_max', 'F1_max', 'Youden',
    'prec_at_rec55', 'prec_at_rec60', 'prec_at_rec65',
    'F2_at_prec12', 'F2_at_prec15', 'F2_at_prec18',
    'recall_65', 'recall_70',
]

# Previous thesis results for comparison
THESIS_2023 = dict(roc_auc=0.8310, pr_auc=0.1944, precision=0.1467, recall=0.6106, f2=0.3740)
THESIS_2024 = dict(roc_auc=0.8251, pr_auc=0.2055, precision=0.1348, recall=0.6750, f2=0.3747)

# Logging
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
# Force UTF-8 stdout so Unicode chars don't cause logging errors on Windows cp1252
if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
    ],
)
log = logging.getLogger('rich4way')

# Ensure output dirs
for subdir in ('tables', 'figures', 'logs', 'models', 'reports', 'data_cache'):
    (OUT / subdir).mkdir(parents=True, exist_ok=True)


# Data loading

def load_rich_csv(path: Path) -> pd.DataFrame:
    log.info(f"Loading {path.name} ...")
    cache = OUT / 'data_cache' / (path.stem + '.parquet')
    if cache.exists():
        df = pd.read_parquet(cache)
        log.info(f"  From cache: {len(df):,} rows x {df.shape[1]} cols")
    else:
        chunks = []
        for chunk in pd.read_csv(path, chunksize=CHUNKSIZE, low_memory=False):
            for c in chunk.select_dtypes(include='float64').columns:
                chunk[c] = chunk[c].astype('float32')
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        del chunks; gc.collect()
        df.to_parquet(cache, index=False)
    prevalence = df['stroke'].mean()
    log.info(f"  {len(df):,} rows | {df.shape[1]-1} features | prevalence={prevalence:.4f}")
    return df


# Four-way split

def four_way_split(df: pd.DataFrame, seed: int = SEED):
    """Stratified 60/10/10/20 split. Returns (train, cal, thresh, test) DataFrames."""
    y = df['stroke'].values
    idx = np.arange(len(df))

    idx_rest, idx_test = train_test_split(idx, test_size=0.20, stratify=y, random_state=seed)
    idx_tc, idx_thresh = train_test_split(idx_rest, test_size=0.125, stratify=y[idx_rest], random_state=seed)
    idx_train, idx_cal = train_test_split(idx_tc, test_size=1/7, stratify=y[idx_tc], random_state=seed)

    splits = {
        'train': df.iloc[idx_train].reset_index(drop=True),
        'cal':   df.iloc[idx_cal].reset_index(drop=True),
        'thresh':df.iloc[idx_thresh].reset_index(drop=True),
        'test':  df.iloc[idx_test].reset_index(drop=True),
    }
    for name, sdf in splits.items():
        log.info(f"  {name}: {len(sdf):,} rows | prevalence={sdf['stroke'].mean():.4f}")
    return splits


# Feature sets

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != 'stroke']

def define_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Define R0-R2 feature sets from the dataframe."""
    all_f = get_feature_cols(df)
    miss = df[all_f].isnull().mean()

    r0 = all_f
    r1 = [f for f in all_f if miss[f] <= 0.40]
    r2 = [f for f in all_f if f not in ONLY_2023]

    log.info(f"Feature sets: R0={len(r0)}, R1={len(r1)}, R2={len(r2)}")
    return {'R0': r0, 'R1': r1, 'R2': r2}

def derive_importance_sets(model: XGBClassifier, preprocessor, cat_cols, num_cols,
                            base_features: list[str]) -> dict[str, list[str]]:
    """After training, derive R4 (top 50%) and R5 (top 20) by XGB importance."""
    importances = model.feature_importances_
    n_total = len(importances)

    # map processed feature index back to original feature name
    # Numeric features come first, then OHE'd categoricals
    # We aggregate by original feature: sum importance of all OHE columns for that feature
    ohe = preprocessor.named_transformers_.get('cat')
    if ohe is not None:
        ohe_step = ohe.named_steps['ohe']
        cat_feature_names = list(ohe_step.get_feature_names_out(cat_cols))
    else:
        cat_feature_names = []

    processed_feature_names = list(num_cols) + cat_feature_names
    n_proc = len(processed_feature_names)
    if n_proc != n_total:
        log.warning(f"Feature name mismatch: {n_proc} processed vs {n_total} importances")

    # Sum importances back to original features
    orig_imp = {f: 0.0 for f in base_features}
    for i, pname in enumerate(processed_feature_names[:n_total]):
        if i < len(num_cols):
            orig = num_cols[i]
        else:
            # cat feature: "featurename_value"
            orig = pname.split('_')[0]
            # find original feature whose name is a prefix
            for f in cat_cols:
                if pname.startswith(f + '_') or pname == f:
                    orig = f
                    break
        if orig in orig_imp:
            orig_imp[orig] += importances[i]

    sorted_features = sorted(orig_imp.items(), key=lambda x: x[1], reverse=True)
    all_sorted = [f for f, _ in sorted_features if f in base_features]

    n_half = max(10, len(all_sorted) // 2)
    r4 = all_sorted[:n_half]
    r5 = all_sorted[:min(20, len(all_sorted))]

    log.info(f"Feature sets: R4={len(r4)}, R5={len(r5)} (from XGB importance on R0)")
    return {'R4': r4, 'R5': r5}


# Preprocessing

def build_preprocessor(df_train: pd.DataFrame, feature_cols: list[str]):
    """Fit ColumnTransformer: numeric -> impute+scale, categorical -> impute+OHE."""
    # Use is_numeric_dtype - handles float32/64, int*, bool; rejects object and StringDtype
    cat_cols = [c for c in feature_cols
                if not pd.api.types.is_numeric_dtype(df_train[c].dtype)]
    num_cols = [c for c in feature_cols
                if pd.api.types.is_numeric_dtype(df_train[c].dtype)]

    transformers = []
    if num_cols:
        transformers.append(('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first',
                                  dtype=np.float32, sparse_output=False)),
        ]), cat_cols))

    ct = ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)
    ct.fit(df_train[feature_cols])
    n_out = ct.transform(df_train.iloc[:1][feature_cols]).shape[1]
    log.info(f"  Preprocessor: {len(num_cols)} num, {len(cat_cols)} cat -> {n_out} processed features")
    return ct, cat_cols, num_cols

def xform(ct, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return ct.transform(df[feature_cols]).astype(np.float32)


# Optuna search

def composite_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """0.35*PR-AUC + 0.30*F2-max + 0.15*prec@rec>=0.60 + 0.10*ROC-AUC + 0.10*(1-Brier*10)."""
    try:
        pr_auc = average_precision_score(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        brier = brier_score_loss(y_true, y_proba)
    except Exception:
        return 0.0

    # Best F2 over dense threshold sweep
    thr = np.linspace(0.01, 0.50, 300)
    yp = y_proba[np.newaxis, :] >= thr[:, np.newaxis]
    pos = y_true.sum()
    tp = yp[:, y_true == 1].sum(axis=1).astype(float)
    fp = yp[:, y_true == 0].sum(axis=1).astype(float)
    prec_arr = tp / (tp + fp + 1e-9)
    rec_arr  = tp / (pos + 1e-9)
    f2_arr   = 5 * prec_arr * rec_arr / (4 * prec_arr + rec_arr + 1e-9)
    best_f2 = float(f2_arr.max())

    # prec@rec>=0.60 - highest precision where recall >= 0.60
    mask60 = rec_arr >= 0.60
    prec_at_rec60 = float(prec_arr[mask60].max()) if mask60.any() else 0.0

    return (0.35 * pr_auc + 0.30 * best_f2 + 0.15 * prec_at_rec60
            + 0.10 * roc_auc + 0.10 * max(0.0, 1.0 - brier * 10))

def make_xgb_objective(X_train: np.ndarray, y_train: np.ndarray,
                        X_cal: np.ndarray, y_cal: np.ndarray):
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            max_depth        = trial.suggest_int('max_depth', 3, 10),
            learning_rate    = trial.suggest_float('learning_rate', 0.005, 0.30, log=True),
            subsample        = trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.3, 1.0),
            min_child_weight = trial.suggest_int('min_child_weight', 1, 20),
            gamma            = trial.suggest_float('gamma', 0.0, 5.0),
            reg_alpha        = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            reg_lambda       = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            scale_pos_weight = trial.suggest_float('scale_pos_weight', 5.0, 30.0),
            n_estimators     = 2000,
            tree_method      = 'hist',
            random_state     = SEED,
            eval_metric      = 'aucpr',
            early_stopping_rounds = 50,
            verbosity        = 0,
        )
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)
        proba = model.predict_proba(X_cal)[:, 1]
        return composite_score(y_cal, proba)
    return objective

def run_optuna_search(X_train, y_train, X_cal, y_cal, n_trials: int, label: str) -> dict:
    t0 = time.time()
    log.info(f"  Optuna [{label}]: {n_trials} trials ...")
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=max(10, n_trials // 5)),
    )
    study.optimize(
        make_xgb_objective(X_train, y_train, X_cal, y_cal),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=False,
    )
    elapsed = time.time() - t0
    log.info(f"  [{label}] best={study.best_value:.4f} in {elapsed/60:.1f} min | "
             f"params={study.best_params}")
    return study.best_params, study.best_value


# Model training

def train_xgb(X_train, y_train, X_cal, y_cal, best_params: dict) -> XGBClassifier:
    params = {
        **best_params,
        'n_estimators': 2000,
        'tree_method': 'hist',
        'random_state': SEED,
        'eval_metric': 'aucpr',
        'early_stopping_rounds': 50,
        'verbosity': 0,
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)
    log.info(f"  Trained XGB: best_iteration={model.best_iteration}")
    return model


# Calibration

class _CalibratedModel:
    """Wraps an XGBClassifier + calibrator with a predict_proba interface."""

    def __init__(self, base: XGBClassifier, calibrator, method: str):
        self.base = base
        self.calibrator = calibrator
        self.method = method

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base.predict_proba(X)[:, 1]
        if self.method == 'sigmoid':
            cal = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            cal = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])


def select_calibration(model: XGBClassifier, X_cal: np.ndarray, y_cal: np.ndarray):
    """Fit sigmoid (Platt) and isotonic; return best by Brier on cal split."""
    raw = model.predict_proba(X_cal)[:, 1]
    cal_results = {}
    best_model, best_method, best_brier = None, '', np.inf

    # Sigmoid / Platt scaling
    platt = PlattScaler(max_iter=2000, C=1.0)
    platt.fit(raw.reshape(-1, 1), y_cal)
    sig_proba = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
    sig_brier = brier_score_loss(y_cal, sig_proba)
    sig_ece   = compute_ece(y_cal, sig_proba)
    cal_results['sigmoid'] = {'brier': sig_brier, 'ece': sig_ece}
    log.info(f"  Cal sigmoid:  Brier={sig_brier:.6f}, ECE={sig_ece:.6f}")

    # Isotonic regression
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw, y_cal)
    iso_proba = np.clip(iso.predict(raw), 0.0, 1.0)
    iso_brier = brier_score_loss(y_cal, iso_proba)
    iso_ece   = compute_ece(y_cal, iso_proba)
    cal_results['isotonic'] = {'brier': iso_brier, 'ece': iso_ece}
    log.info(f"  Cal isotonic: Brier={iso_brier:.6f}, ECE={iso_ece:.6f}")

    if sig_brier <= iso_brier:
        best_method = 'sigmoid'
        best_model  = _CalibratedModel(model, platt, 'sigmoid')
    else:
        best_method = 'isotonic'
        best_model  = _CalibratedModel(model, iso, 'isotonic')

    log.info(f"  Selected: {best_method}")
    return best_model, best_method, cal_results

def compute_ece(y_true, proba, n_bins=10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        mask = (proba >= bin_edges[i]) & (proba < bin_edges[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() / n * abs(y_true[mask].mean() - proba[mask].mean())
    return float(ece)


# Threshold policies

def compute_threshold_policies(cal_model, X_thresh, y_thresh) -> dict[str, float]:
    """Sweep 11 threshold policies on thresh split (leakage-free)."""
    proba = cal_model.predict_proba(X_thresh)[:, 1]
    thr = np.linspace(0.005, 0.70, 1500)

    # Vectorised confusion-matrix components
    yp = proba[np.newaxis, :] >= thr[:, np.newaxis]   # (n_thr, n_samples)
    pos = float(y_thresh.sum())
    neg = float(len(y_thresh) - pos)

    tp = yp[:, y_thresh == 1].sum(axis=1).astype(float)
    fp = yp[:, y_thresh == 0].sum(axis=1).astype(float)
    fn = pos - tp
    tn = neg - fp

    recall = tp / (pos + 1e-9)
    prec   = tp / (tp + fp + 1e-9)
    spec   = tn / (neg + 1e-9)
    f2     = 5 * prec * recall / (4 * prec + recall + 1e-9)
    f1     = 2 * prec * recall / (prec + recall + 1e-9)
    youden = recall + spec - 1.0

    def best_thr(score_arr):
        return float(thr[np.argmax(score_arr)])

    def highest_thr_meeting(cond_arr, score_arr):
        """Highest threshold where condition is met AND score is maximised."""
        mask = cond_arr
        if not mask.any():
            return best_thr(f2)
        valid = np.where(mask)[0]
        return float(thr[valid[np.argmax(score_arr[valid])]])

    def highest_thr_at_recall(min_recall):
        """Highest threshold (max precision) where recall >= min_recall."""
        mask = recall >= min_recall
        if not mask.any():
            return float(thr[0])
        return float(thr[np.where(mask)[0][-1]])

    policies = {
        'F2_max':        best_thr(f2),
        'F1_max':        best_thr(f1),
        'Youden':        best_thr(youden),
        'prec_at_rec55': highest_thr_meeting(recall >= 0.55, prec),
        'prec_at_rec60': highest_thr_meeting(recall >= 0.60, prec),
        'prec_at_rec65': highest_thr_meeting(recall >= 0.65, prec),
        'F2_at_prec12':  highest_thr_meeting(prec >= 0.12, f2),
        'F2_at_prec15':  highest_thr_meeting(prec >= 0.15, f2),
        'F2_at_prec18':  highest_thr_meeting(prec >= 0.18, f2),
        'recall_65':     highest_thr_at_recall(0.65),
        'recall_70':     highest_thr_at_recall(0.70),
    }
    for k, v in policies.items():
        log.info(f"    {k}: {v:.4f}")
    return policies


# Evaluation

def evaluate_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    ypred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, ypred, labels=[0, 1]).ravel()
    return {
        'threshold': float(threshold),
        'recall':    float(recall_score(y_true, ypred, zero_division=0)),
        'precision': float(precision_score(y_true, ypred, zero_division=0)),
        'f1':        float(f1_score(y_true, ypred, zero_division=0)),
        'f2':        float(fbeta_score(y_true, ypred, beta=2, zero_division=0)),
        'roc_auc':   float(roc_auc_score(y_true, proba)),
        'pr_auc':    float(average_precision_score(y_true, proba)),
        'brier':     float(brier_score_loss(y_true, proba)),
        'ece':       compute_ece(y_true, proba),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }

def bootstrap_ci(y_true: np.ndarray, proba: np.ndarray,
                  threshold: float, n_boot: int = N_BOOT) -> dict:
    rng = np.random.RandomState(SEED)
    n = len(y_true)
    m: dict[str, list] = {k: [] for k in ('recall', 'precision', 'f2', 'roc_auc', 'pr_auc', 'brier')}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], proba[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        ypred = (yp >= threshold).astype(int)
        m['recall'].append(recall_score(yt, ypred, zero_division=0))
        m['precision'].append(precision_score(yt, ypred, zero_division=0))
        m['f2'].append(fbeta_score(yt, ypred, beta=2, zero_division=0))
        m['roc_auc'].append(roc_auc_score(yt, yp))
        m['pr_auc'].append(average_precision_score(yt, yp))
        m['brier'].append(brier_score_loss(yt, yp))
    ci = {}
    for k, vals in m.items():
        if vals:
            ci[f'{k}_ci_lo'] = float(np.percentile(vals, 2.5))
            ci[f'{k}_ci_hi'] = float(np.percentile(vals, 97.5))
    return ci


# CORE EXPERIMENT RUNNER (one feature set, one dataset split)

def run_experiment(
    splits: dict,
    feature_cols: list[str],
    n_trials: int,
    label: str,
    test_df: pd.DataFrame | None = None,   # override test split (for MODE B cross-year)
    prefit_params: dict | None = None,     # skip Optuna, use given params
) -> dict:
    """
    Full pipeline: preprocess -> Optuna -> train -> calibrate -> threshold -> test -> bootstrap.
    Returns result dict. test_df overrides splits['test'] for cross-year evaluation.
    """
    log.info(f"\n{'-'*60}")
    log.info(f"  Experiment: {label} | features: {len(feature_cols)}")

    df_train  = splits['train']
    df_cal    = splits['cal']
    df_thresh = splits['thresh']
    df_test   = test_df if test_df is not None else splits['test']

    y_train  = df_train['stroke'].values
    y_cal    = df_cal['stroke'].values
    y_thresh = df_thresh['stroke'].values
    y_test   = df_test['stroke'].values

    # Preprocessing (fit on train only)
    ct, cat_cols, num_cols = build_preprocessor(df_train, feature_cols)
    X_train  = xform(ct, df_train,  feature_cols)
    X_cal_m  = xform(ct, df_cal,    feature_cols)
    X_thresh = xform(ct, df_thresh, feature_cols)
    X_test   = xform(ct, df_test,   feature_cols)
    log.info(f"  X_train={X_train.shape}, X_test={X_test.shape}")

    # Optuna or use prefit params
    if prefit_params is not None:
        best_params = prefit_params
        best_score  = float('nan')
        log.info(f"  Skipping Optuna — using prefit params")
    else:
        best_params, best_score = run_optuna_search(
            X_train, y_train, X_cal_m, y_cal, n_trials, label
        )

    # Train final model
    model = train_xgb(X_train, y_train, X_cal_m, y_cal, best_params)

    # Calibration
    cal_model, cal_method, cal_results = select_calibration(model, X_cal_m, y_cal)

    # Threshold policies on thresh split
    log.info(f"  Computing threshold policies on thresh split ...")
    policies = compute_threshold_policies(cal_model, X_thresh, y_thresh)

    # Evaluate all threshold policies on test set
    proba_test = cal_model.predict_proba(X_test)[:, 1]
    test_results = []
    for policy_name, thr_val in policies.items():
        metrics = evaluate_at_threshold(y_test, proba_test, thr_val)
        metrics['policy'] = policy_name
        metrics['label'] = label
        test_results.append(metrics)
        log.info(f"  [TEST] {policy_name}: thr={thr_val:.4f} "
                 f"rec={metrics['recall']:.4f} prec={metrics['precision']:.4f} "
                 f"F2={metrics['f2']:.4f} PR-AUC={metrics['pr_auc']:.4f}")

    # Bootstrap CI on best policy (F2_max)
    f2_thr = policies['F2_max']
    ci = bootstrap_ci(y_test, proba_test, f2_thr)

    # Composite score on test (for ranking)
    test_composite = composite_score(y_test, proba_test)

    return {
        'label': label,
        'feature_set': label.split('_')[-1] if '_' in label else label,
        'n_features': len(feature_cols),
        'n_processed': X_train.shape[1],
        'best_optuna_score': best_score,
        'best_params': best_params,
        'cal_method': cal_method,
        'cal_results': cal_results,
        'policies': policies,
        'test_results': test_results,
        'bootstrap_ci': ci,
        'test_composite': test_composite,
        'proba_test': proba_test,
        'y_test': y_test,
        'model': model,
        'cal_model': cal_model,
        'preprocessor': ct,
        'feature_cols': feature_cols,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
    }


# Figure generation

def plot_pr_roc(results_list: list[dict], suffix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for res in results_list:
        y = res['y_test']
        p = res['proba_test']
        lbl = res['label']
        prec, rec, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        pr_auc = average_precision_score(y, p)
        roc_auc = roc_auc_score(y, p)

        axes[0].plot(rec, prec, label=f"{lbl} (AUC={pr_auc:.4f})", linewidth=1.5)
        axes[1].plot(fpr, tpr, label=f"{lbl} (AUC={roc_auc:.4f})", linewidth=1.5)

    axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision-Recall Curve'); axes[0].legend(fontsize=8)
    axes[0].axhline(y=results_list[0]['y_test'].mean(), color='gray', linestyle='--', label='Chance')

    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
    axes[1].set_title('ROC Curve'); axes[1].legend(fontsize=8)

    plt.tight_layout()
    path = OUT / 'figures' / f'F01_pr_roc_{suffix}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path.name}")

def plot_calibration(results_list: list[dict], suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    for res in results_list:
        frac_pos, mean_pred = calibration_curve(res['y_test'], res['proba_test'], n_bins=15)
        brier = brier_score_loss(res['y_test'], res['proba_test'])
        ax.plot(mean_pred, frac_pos, marker='o', markersize=4,
                label=f"{res['label']} (Brier={brier:.4f})")
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Plot')
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = OUT / 'figures' / f'F02_calibration_{suffix}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path.name}")

def plot_threshold_sweep(res: dict, suffix: str) -> None:
    proba = res['proba_test']
    y = res['y_test']
    thresholds = np.linspace(0.005, 0.60, 400)

    yp = proba[np.newaxis, :] >= thresholds[:, np.newaxis]
    pos = y.sum()
    tp = yp[:, y == 1].sum(axis=1).astype(float)
    fp = yp[:, y == 0].sum(axis=1).astype(float)
    fn = pos - tp
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (pos + 1e-9)
    f2   = 5 * prec * rec / (4 * prec + rec + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, rec,  label='Recall', color='tab:blue')
    ax.plot(thresholds, prec, label='Precision', color='tab:orange')
    ax.plot(thresholds, f2,   label='F2', color='tab:green', linewidth=2)

    best_thr = res['policies']['F2_max']
    ax.axvline(best_thr, color='gray', linestyle='--', label=f'F2-max thr={best_thr:.3f}')

    ax.set_xlabel('Decision threshold'); ax.set_ylabel('Score')
    ax.set_title(f'Threshold Sweep — {res["label"]}')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUT / 'figures' / f'F03_threshold_sweep_{suffix}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path.name}")

def plot_feature_importance(res: dict, top_n: int = 25) -> None:
    model = res['model']
    importances = model.feature_importances_
    cat_cols = res['cat_cols']
    num_cols = res['num_cols']
    ct = res['preprocessor']

    # Get processed feature names
    ohe_step = None
    for name, tr, cols in ct.transformers_:
        if name == 'cat':
            ohe_step = tr.named_steps['ohe']
    cat_names = list(ohe_step.get_feature_names_out(cat_cols)) if ohe_step else []
    all_names = list(num_cols) + cat_names

    n = min(len(all_names), len(importances))
    pairs = sorted(zip(all_names[:n], importances[:n]), key=lambda x: x[1], reverse=True)[:top_n]
    names, vals = zip(*pairs)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(names[::-1], vals[::-1], color='steelblue')
    ax.set_xlabel('Feature Importance (gain)')
    ax.set_title(f'Top {top_n} Feature Importances — {res["label"]}')
    plt.tight_layout()
    path = OUT / 'figures' / f'F04_feature_importance_{res["label"]}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path.name}")

def plot_mode_comparison(all_results: list[dict]) -> None:
    labels, f2s, pr_aucs, recalls, precs = [], [], [], [], []
    for res in all_results:
        f2_res = next((r for r in res['test_results'] if r['policy'] == 'F2_max'), None)
        if f2_res:
            labels.append(res['label'])
            f2s.append(f2_res['f2'])
            pr_aucs.append(f2_res['pr_auc'])
            recalls.append(f2_res['recall'])
            precs.append(f2_res['precision'])

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))
    ax.bar(x - 1.5*width, f2s,     width, label='F2',        color='tab:green')
    ax.bar(x - 0.5*width, pr_aucs, width, label='PR-AUC',    color='tab:blue')
    ax.bar(x + 0.5*width, recalls, width, label='Recall',     color='tab:orange')
    ax.bar(x + 1.5*width, precs,   width, label='Precision',  color='tab:red')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylabel('Score'); ax.set_title('Mode/Feature-Set Comparison (F2-max threshold)')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = OUT / 'figures' / 'F05_mode_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path.name}")

def plot_bootstrap_distributions(res: dict) -> None:
    """Histogram of bootstrap F2 and PR-AUC for the champion result."""
    proba = res['proba_test']
    y = res['y_test']
    thr = res['policies']['F2_max']
    rng = np.random.RandomState(SEED)
    n = len(y)
    f2s, prauc = [], []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        yt, yp = y[idx], proba[idx]
        if yt.sum() == 0 or yt.sum() == n: continue
        ypred = (yp >= thr).astype(int)
        f2s.append(fbeta_score(yt, ypred, beta=2, zero_division=0))
        prauc.append(average_precision_score(yt, yp))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(f2s, bins=50, color='tab:green', edgecolor='white', alpha=0.8)
    axes[0].axvline(np.percentile(f2s, 2.5), color='red', linestyle='--')
    axes[0].axvline(np.percentile(f2s, 97.5), color='red', linestyle='--')
    axes[0].set_title(f'Bootstrap F2 (thr={thr:.4f})\n95% CI [{np.percentile(f2s,2.5):.4f}, {np.percentile(f2s,97.5):.4f}]')
    axes[0].set_xlabel('F2 Score')

    axes[1].hist(prauc, bins=50, color='tab:blue', edgecolor='white', alpha=0.8)
    axes[1].axvline(np.percentile(prauc, 2.5), color='red', linestyle='--')
    axes[1].axvline(np.percentile(prauc, 97.5), color='red', linestyle='--')
    axes[1].set_title(f'Bootstrap PR-AUC\n95% CI [{np.percentile(prauc,2.5):.4f}, {np.percentile(prauc,97.5):.4f}]')
    axes[1].set_xlabel('PR-AUC')

    plt.suptitle(f'Bootstrap Distributions — {res["label"]}', fontsize=13)
    plt.tight_layout()
    path = OUT / 'figures' / f'F06_bootstrap_{res["label"]}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Saved: {path.name}")


# Table generation

def save_tables(all_results: list[dict], feature_sets: dict, splits: dict,
                splits_2024: dict | None) -> None:
    tbl = OUT / 'tables'

    # T01 Dataset stats
    rows = [
        {'dataset': '2023', 'total_rows': 431849, 'n_stroke': 18350,
         'prevalence': 0.0425, 'n_features_R0': 55, 'n_features_R2': 52},
        {'dataset': '2024', 'total_rows': 456218, 'n_stroke': 20664,
         'prevalence': 0.0453, 'n_features_R2': 52},
    ]
    pd.DataFrame(rows).to_csv(tbl / 'T01_dataset_stats.csv', index=False)

    # T02 Feature sets
    fs_rows = []
    for name, cols in feature_sets.items():
        fs_rows.append({'set': name, 'n_features': len(cols), 'features': '|'.join(cols)})
    pd.DataFrame(fs_rows).to_csv(tbl / 'T02_feature_sets.csv', index=False)

    # T03 Split summary
    split_rows = []
    for name, sdf in splits.items():
        split_rows.append({'split': f'2023_{name}', 'n_rows': len(sdf),
                           'n_stroke': sdf['stroke'].sum(), 'prevalence': sdf['stroke'].mean()})
    if splits_2024:
        for name, sdf in splits_2024.items():
            split_rows.append({'split': f'2024_{name}', 'n_rows': len(sdf),
                               'n_stroke': sdf['stroke'].sum(), 'prevalence': sdf['stroke'].mean()})
    pd.DataFrame(split_rows).to_csv(tbl / 'T03_split_summary.csv', index=False)

    # T04 Optuna best params
    param_rows = []
    for res in all_results:
        row = {'label': res['label'], 'optuna_score': res['best_optuna_score']}
        row.update(res['best_params'])
        param_rows.append(row)
    pd.DataFrame(param_rows).to_csv(tbl / 'T04_optuna_best_params.csv', index=False)

    # T05 Calibration comparison
    cal_rows = []
    for res in all_results:
        for method, cres in res['cal_results'].items():
            cal_rows.append({'label': res['label'], 'method': method,
                             'brier': cres['brier'], 'ece': cres['ece'],
                             'selected': method == res['cal_method']})
    pd.DataFrame(cal_rows).to_csv(tbl / 'T05_calibration_comparison.csv', index=False)

    # T06 Threshold sweep - all policies all experiments
    thr_rows = []
    for res in all_results:
        for policy, thr_val in res['policies'].items():
            thr_rows.append({'label': res['label'], 'policy': policy, 'threshold': thr_val})
    pd.DataFrame(thr_rows).to_csv(tbl / 'T06_threshold_policies.csv', index=False)

    # T07-T08 Test results (all policies, all experiments)
    test_rows = []
    for res in all_results:
        for tr in res['test_results']:
            test_rows.append(tr)
    df_test = pd.DataFrame(test_rows)
    df_test.to_csv(tbl / 'T07_test_results_all.csv', index=False)

    # T08 F2_max results only (cleaner summary)
    f2_rows = [r for r in test_rows if r['policy'] == 'F2_max']
    pd.DataFrame(f2_rows).to_csv(tbl / 'T08_test_results_F2max.csv', index=False)

    # T09 Bootstrap CIs
    ci_rows = []
    for res in all_results:
        row = {'label': res['label'], 'policy': 'F2_max',
               'threshold': res['policies']['F2_max']}
        row.update(res['bootstrap_ci'])
        ci_rows.append(row)
    pd.DataFrame(ci_rows).to_csv(tbl / 'T09_bootstrap_ci.csv', index=False)

    # T10 Mode comparison - best F2 per experiment
    cmp_rows = []
    for res in all_results:
        f2_res = next((r for r in res['test_results'] if r['policy'] == 'F2_max'), {})
        if f2_res:
            cmp_rows.append({
                'label': res['label'],
                'n_features': res['n_features'],
                'n_processed': res['n_processed'],
                'cal_method': res['cal_method'],
                **{k: v for k, v in f2_res.items() if k != 'policy' and k != 'label'},
                'test_composite': res['test_composite'],
            })
    pd.DataFrame(cmp_rows).to_csv(tbl / 'T10_mode_comparison.csv', index=False)

    # T11 Feature importance for MODE_A_R0
    r0_res = next((r for r in all_results if 'A_R0' in r['label']), None)
    if r0_res:
        model = r0_res['model']
        imp = model.feature_importances_
        ct = r0_res['preprocessor']
        cat_cols = r0_res['cat_cols']
        num_cols = r0_res['num_cols']
        ohe_step = None
        for name, tr, cols in ct.transformers_:
            if name == 'cat':
                ohe_step = tr.named_steps['ohe']
        cat_names = list(ohe_step.get_feature_names_out(cat_cols)) if ohe_step else []
        all_names = list(num_cols) + cat_names
        n = min(len(all_names), len(imp))
        imp_df = pd.DataFrame({'feature': all_names[:n], 'importance': imp[:n]})
        imp_df = imp_df.sort_values('importance', ascending=False)
        imp_df.to_csv(tbl / 'T11_feature_importance_A_R0.csv', index=False)

    # T12 vs thesis comparison
    cmp_thesis_rows = []
    for res in all_results:
        f2_res = next((r for r in res['test_results'] if r['policy'] == 'F2_max'), {})
        if f2_res:
            ref = THESIS_2023 if '2023' in res['label'] or 'B_' in res['label'] else THESIS_2024
            cmp_thesis_rows.append({
                'label': res['label'],
                'this_roc_auc': f2_res.get('roc_auc'),
                'this_pr_auc': f2_res.get('pr_auc'),
                'this_recall': f2_res.get('recall'),
                'this_precision': f2_res.get('precision'),
                'this_f2': f2_res.get('f2'),
                'thesis_roc_auc': ref['roc_auc'],
                'thesis_pr_auc': ref['pr_auc'],
                'thesis_recall': ref['recall'],
                'thesis_precision': ref['precision'],
                'thesis_f2': ref['f2'],
                'delta_f2': f2_res.get('f2', 0) - ref['f2'],
                'delta_pr_auc': f2_res.get('pr_auc', 0) - ref['pr_auc'],
            })
    pd.DataFrame(cmp_thesis_rows).to_csv(tbl / 'T12_vs_thesis_comparison.csv', index=False)

    # T13 Champion result (highest test_composite)
    best = max(all_results, key=lambda r: r['test_composite'])
    best_f2 = next((r for r in best['test_results'] if r['policy'] == 'F2_max'), {})
    champion = {
        'label': best['label'],
        'n_features': best['n_features'],
        'cal_method': best['cal_method'],
        'test_composite_score': best['test_composite'],
        **best_f2,
        **best['bootstrap_ci'],
    }
    pd.DataFrame([champion]).to_csv(tbl / 'T13_champion_result.csv', index=False)
    log.info(f"\nChampion: {best['label']} | composite={best['test_composite']:.4f}")
    log.info(f"  F2={best_f2.get('f2'):.4f} | recall={best_f2.get('recall'):.4f} | "
             f"prec={best_f2.get('precision'):.4f} | PR-AUC={best_f2.get('pr_auc'):.4f}")

    log.info("Tables saved.")


# Report generation

def write_report(all_results: list[dict]) -> None:
    best = max(all_results, key=lambda r: r['test_composite'])
    best_f2 = next((r for r in best['test_results'] if r['policy'] == 'F2_max'), {})
    ci = best['bootstrap_ci']

    lines = [
        "# Rich-Feature BRFSS Stroke Prediction — Final Search Report",
        "",
        f"**Run date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Experiment Summary",
        "",
        "| Mode | Feature Set | PR-AUC | Recall | Precision | F2 | Composite |",
        "|------|-------------|--------|--------|-----------|-----|-----------|",
    ]
    for res in all_results:
        f2r = next((r for r in res['test_results'] if r['policy'] == 'F2_max'), {})
        if f2r:
            lines.append(
                f"| {res['label']} | {res['n_features']}f | "
                f"{f2r.get('pr_auc',0):.4f} | {f2r.get('recall',0):.4f} | "
                f"{f2r.get('precision',0):.4f} | {f2r.get('f2',0):.4f} | "
                f"{res['test_composite']:.4f} |"
            )

    lines += [
        "",
        "## Champion Result",
        "",
        f"**Experiment:** `{best['label']}`",
        f"**Feature set:** {best['n_features']} features (processed: {best['n_processed']})",
        f"**Calibration:** {best['cal_method']}",
        f"**Threshold policy:** F2-max = {best['policies']['F2_max']:.4f}",
        "",
        "### Test-set Metrics",
        "",
        f"| Metric | Value | 95% CI |",
        f"|--------|-------|--------|",
        f"| ROC-AUC | {best_f2.get('roc_auc',0):.4f} | — |",
        f"| PR-AUC | {best_f2.get('pr_auc',0):.4f} | [{ci.get('pr_auc_ci_lo',0):.4f}, {ci.get('pr_auc_ci_hi',0):.4f}] |",
        f"| Recall | {best_f2.get('recall',0):.4f} | [{ci.get('recall_ci_lo',0):.4f}, {ci.get('recall_ci_hi',0):.4f}] |",
        f"| Precision | {best_f2.get('precision',0):.4f} | [{ci.get('precision_ci_lo',0):.4f}, {ci.get('precision_ci_hi',0):.4f}] |",
        f"| F2 | {best_f2.get('f2',0):.4f} | [{ci.get('f2_ci_lo',0):.4f}, {ci.get('f2_ci_hi',0):.4f}] |",
        f"| Brier | {best_f2.get('brier',0):.6f} | — |",
        f"| ECE | {best_f2.get('ece',0):.6f} | — |",
        "",
        "### vs Thesis Benchmark (2023 rich-feature)",
        "",
        f"| Metric | This experiment | Thesis | Delta |",
        f"|--------|----------------|--------|-------|",
    ]
    for k in ('roc_auc', 'pr_auc', 'recall', 'precision', 'f2'):
        this_v = best_f2.get(k, 0)
        ref_v  = THESIS_2023.get(k, 0)
        delta  = this_v - ref_v
        sign   = '+' if delta >= 0 else ''
        lines.append(f"| {k} | {this_v:.4f} | {ref_v:.4f} | {sign}{delta:.4f} |")

    lines += [
        "",
        "### Best Params (XGBoost)",
        "",
        "```json",
        json.dumps(best['best_params'], indent=2),
        "```",
        "",
        "## All Threshold Policies (Champion Experiment — Test Set)",
        "",
        "| Policy | Threshold | Recall | Precision | F2 | PR-AUC |",
        "|--------|-----------|--------|-----------|-----|--------|",
    ]
    for tr in best['test_results']:
        lines.append(
            f"| {tr['policy']} | {tr['threshold']:.4f} | "
            f"{tr['recall']:.4f} | {tr['precision']:.4f} | "
            f"{tr['f2']:.4f} | {tr['pr_auc']:.4f} |"
        )

    lines += ["", "## Output Files", "",
              "### Tables", ""]
    for f in sorted((OUT / 'tables').glob('*.csv')):
        lines.append(f"- `tables/{f.name}`")
    lines += ["", "### Figures", ""]
    for f in sorted((OUT / 'figures').glob('*.png')):
        lines.append(f"- `figures/{f.name}`")

    report_path = OUT / 'reports' / 'REPORT.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    log.info(f"\nReport written: {report_path}")


# Main

def main() -> None:
    t_start = time.time()
    log.info("=" * 72)
    log.info("Rich-Feature BRFSS Stroke — Aggressive 4-Way Final Search")
    log.info("=" * 72)

    # Load data
    log.info("\n--- Loading datasets ---")
    df_2023 = load_rich_csv(DATA_2023)
    df_2024 = load_rich_csv(DATA_2024)

    # 2023 four-way split
    log.info("\n--- 2023 Four-way split ---")
    splits_2023 = four_way_split(df_2023)

    # 2024 four-way split
    log.info("\n--- 2024 Four-way split ---")
    splits_2024 = four_way_split(df_2024)

    # Feature sets
    log.info("\n--- Defining feature sets ---")
    fsets_2023 = define_feature_sets(df_2023)
    fsets_2024 = define_feature_sets(df_2024)
    # R2 for 2024 (only common features)
    r2_for_2024 = fsets_2024['R2']   # already excludes ONLY_2023

    all_results = []

    # MODE A - 2023 data, aggressive search, multiple feature sets
    log.info("\n" + "=" * 72)
    log.info("MODE A — BRFSS 2023, Aggressive XGBoost Search")
    log.info("=" * 72)

    # A_R0: Main feature set, 100 trials
    log.info("\n[MODE A / R0] 55 features, 100 Optuna trials")
    res_A_R0 = run_experiment(
        splits=splits_2023,
        feature_cols=fsets_2023['R0'],
        n_trials=N_TRIALS_A_MAIN,
        label='MODE_A_R0',
    )
    all_results.append(res_A_R0)

    # Derive R4, R5 from R0 model
    log.info("\n[MODE A] Deriving R4, R5 from R0 XGB importances ...")
    imp_sets = derive_importance_sets(
        res_A_R0['model'],
        res_A_R0['preprocessor'],
        res_A_R0['cat_cols'],
        res_A_R0['num_cols'],
        fsets_2023['R0'],
    )
    fsets_2023.update(imp_sets)

    # A_R1: Low-missing features, 60 trials
    log.info("\n[MODE A / R1] Low-miss features, 60 Optuna trials")
    res_A_R1 = run_experiment(
        splits=splits_2023,
        feature_cols=fsets_2023['R1'],
        n_trials=N_TRIALS_A_ALT,
        label='MODE_A_R1',
    )
    all_results.append(res_A_R1)

    # A_R2: Common features (also in 2024), 60 trials
    log.info("\n[MODE A / R2] 52 common features, 60 Optuna trials")
    res_A_R2 = run_experiment(
        splits=splits_2023,
        feature_cols=fsets_2023['R2'],
        n_trials=N_TRIALS_A_ALT,
        label='MODE_A_R2',
    )
    all_results.append(res_A_R2)

    # A_R4: XGB importance top-50%, use best R0 params
    log.info("\n[MODE A / R4] Top-50%% XGB features, prefit params from R0")
    res_A_R4 = run_experiment(
        splits=splits_2023,
        feature_cols=fsets_2023['R4'],
        n_trials=0,  # unused when prefit_params given
        label='MODE_A_R4',
        prefit_params=res_A_R0['best_params'],
    )
    all_results.append(res_A_R4)

    # A_R5: Top-20 features, use best R0 params
    log.info("\n[MODE A / R5] Top-20 XGB features, prefit params from R0")
    res_A_R5 = run_experiment(
        splits=splits_2023,
        feature_cols=fsets_2023['R5'],
        n_trials=0,
        label='MODE_A_R5',
        prefit_params=res_A_R0['best_params'],
    )
    all_results.append(res_A_R5)

    # MODE B - 2023 train/cal/thresh -> full 2024 as test
    log.info("\n" + "=" * 72)
    log.info("MODE B — Cross-Year Transfer: 2023 train → 2024 test")
    log.info("=" * 72)

    # Only use R2 features (common to both years)
    r2_cols = fsets_2023['R2']

    # For MODE B: use 2023 splits for train/cal/thresh, full df_2024 as test
    # We need df_2024 to have the same feature columns as r2_cols
    common_r2 = [c for c in r2_cols if c in df_2024.columns]
    log.info(f"  Common R2 features in 2024: {len(common_r2)}/{len(r2_cols)}")

    # Use best R2 params from MODE A for efficiency, then fine-tune
    log.info("\n[MODE B / R2] 2023 train → 2024 test, 60 Optuna trials")
    res_B = run_experiment(
        splits=splits_2023,
        feature_cols=common_r2,
        n_trials=N_TRIALS_B,
        label='MODE_B_R2',
        test_df=df_2024,  # entire 2024 as test set
    )
    all_results.append(res_B)

    # MODE C - 2024 data, same-year, R2 features
    log.info("\n" + "=" * 72)
    log.info("MODE C — BRFSS 2024, Same-Year, R2 Features")
    log.info("=" * 72)

    r2_2024 = [c for c in fsets_2024['R2'] if c in df_2024.columns]
    log.info(f"\n[MODE C / R2] 2024 data, {len(r2_2024)} features, 60 Optuna trials")
    res_C = run_experiment(
        splits=splits_2024,
        feature_cols=r2_2024,
        n_trials=N_TRIALS_C,
        label='MODE_C_R2',
    )
    all_results.append(res_C)

    # Figures
    log.info("\n--- Generating figures ---")

    # F01: PR/ROC for Mode A feature sets
    mode_a_results = [r for r in all_results if r['label'].startswith('MODE_A')]
    plot_pr_roc(mode_a_results, 'mode_A')

    # F02: Calibration for Mode A R0
    plot_calibration([res_A_R0], 'mode_A_R0')

    # F03: Threshold sweep for champion
    champion_res = max(all_results, key=lambda r: r['test_composite'])
    plot_threshold_sweep(champion_res, champion_res['label'])

    # F04: Feature importance for Mode A R0
    plot_feature_importance(res_A_R0)

    # F05: Mode comparison
    plot_mode_comparison(all_results)

    # F06: Bootstrap distributions for champion
    plot_bootstrap_distributions(champion_res)

    # Tables & report
    log.info("\n--- Saving tables ---")
    save_tables(all_results, fsets_2023, splits_2023, splits_2024)

    log.info("\n--- Writing REPORT.md ---")
    write_report(all_results)

    elapsed = time.time() - t_start
    log.info(f"\n{'=' * 72}")
    log.info(f"DONE in {elapsed/60:.1f} min ({elapsed:.0f}s)")
    log.info(f"Outputs: {OUT}")
    log.info(f"Report:  {OUT / 'reports' / 'REPORT.md'}")

    best = champion_res
    best_f2 = next((r for r in best['test_results'] if r['policy'] == 'F2_max'), {})
    log.info(f"\nBest result: {best['label']}")
    log.info(f"  F2={best_f2.get('f2',0):.4f} | recall={best_f2.get('recall',0):.4f} | "
             f"prec={best_f2.get('precision',0):.4f} | PR-AUC={best_f2.get('pr_auc',0):.4f} | "
             f"ROC-AUC={best_f2.get('roc_auc',0):.4f}")
    log.info(f"  Brier={best_f2.get('brier',0):.6f} | ECE={best_f2.get('ece',0):.6f}")


if __name__ == '__main__':
    main()
