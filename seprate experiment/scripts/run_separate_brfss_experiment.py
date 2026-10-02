"""Run an isolated rich-feature BRFSS stroke experiment.

Everything written by this script stays inside the "seprate experiment" folder.
It does not read or write the accepted thesis pipeline outputs.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
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


ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = {
    "BRFSS 2023": ROOT / "LLCP2023.XPT",
    "BRFSS 2024": ROOT / "LLCP2024.XPT",
}
OUT = ROOT / "outputs"
DIRS = {
    "processed": OUT / "data" / "processed",
    "tables": OUT / "tables",
    "figures": OUT / "figures",
    "models": OUT / "models",
    "logs": OUT / "logs",
}

RANDOM_STATE = 42
TEST_SIZE = 0.20
CAL_SIZE_WITHIN_TRAINCAL = 0.25  # 60/20/20 train/cal/test
CHUNK_SIZE = 50_000
EPS = 1e-6


@dataclass(frozen=True)
class FeatureSpec:
    output: str
    candidates: tuple[str, ...]
    kind: str
    description: str


FEATURE_SPECS: list[FeatureSpec] = [
    FeatureSpec("age", ("_AGE80",), "numeric_age", "Age, top-coded at 80 in BRFSS."),
    FeatureSpec("sex", ("SEXVAR", "_SEX"), "categorical", "Sex variable."),
    FeatureSpec("race_ethnicity", ("_IMPRACE",), "categorical", "Imputed race/ethnicity grouping."),
    FeatureSpec("education", ("EDUCA", "_EDUCAG"), "categorical", "Education category."),
    FeatureSpec("income", ("INCOME3", "_INCOMG1"), "categorical", "Income category."),
    FeatureSpec("employment", ("EMPLOY1",), "categorical", "Employment status."),
    FeatureSpec("marital", ("MARITAL",), "categorical", "Marital status."),
    FeatureSpec("home_ownership", ("RENTHOM1",), "categorical", "Home ownership/rent status."),
    FeatureSpec("children_count", ("CHILDREN", "_CHLDCNT"), "numeric_children", "Number of children in household."),
    FeatureSpec("general_health", ("GENHLTH", "_RFHLTH"), "categorical", "General self-rated health."),
    FeatureSpec("physical_health_days", ("PHYSHLTH",), "numeric_days", "Poor physical-health days in past 30 days."),
    FeatureSpec("mental_health_days", ("MENTHLTH",), "numeric_days", "Poor mental-health days in past 30 days."),
    FeatureSpec("poor_health_days", ("POORHLTH",), "numeric_days", "Activity-limited days because of poor health."),
    FeatureSpec("health_insurance", ("PRIMINS1", "PRIMINS2", "_HLTHPL1", "_HLTHPL2"), "categorical", "Health insurance indicator/category."),
    FeatureSpec("personal_doctor", ("PERSDOC3",), "categorical", "Has personal doctor/healthcare provider."),
    FeatureSpec("medical_cost_barrier", ("MEDCOST1",), "yes_no", "Could not see doctor because of cost."),
    FeatureSpec("recent_checkup", ("CHECKUP1",), "categorical", "Time since last routine checkup."),
    FeatureSpec("exercise_any", ("EXERANY2", "_TOTINDA"), "categorical", "Any recent physical activity/exercise indicator."),
    FeatureSpec("hypertension_status", ("BPHIGH6", "_RFHYPE6"), "bp_status", "High blood-pressure history/status when available."),
    FeatureSpec("cholesterol_status", ("TOLDHI3", "_RFCHOL3"), "categorical", "High cholesterol history/status when available."),
    FeatureSpec("cholesterol_meds", ("CHOLMED3",), "yes_no", "Currently taking cholesterol medication."),
    FeatureSpec("heart_attack", ("CVDINFR4",), "yes_no", "Ever told had myocardial infarction/heart attack."),
    FeatureSpec("coronary_heart_disease", ("CVDCRHD4",), "yes_no", "Ever told had angina/coronary heart disease."),
    FeatureSpec("computed_michd", ("_MICHD",), "categorical", "Computed myocardial infarction/coronary heart disease indicator."),
    FeatureSpec("asthma_ever", ("ASTHMA3",), "yes_no", "Ever told had asthma."),
    FeatureSpec("asthma_current", ("ASTHNOW", "_CASTHM1"), "categorical", "Current asthma status."),
    FeatureSpec("skin_cancer", ("CHCSCNC1",), "yes_no", "Ever told had skin cancer."),
    FeatureSpec("other_cancer", ("CHCOCNC1",), "yes_no", "Ever told had other cancer."),
    FeatureSpec("copd", ("CHCCOPD3",), "yes_no", "Ever told had COPD/emphysema/chronic bronchitis."),
    FeatureSpec("depression", ("ADDEPEV3",), "yes_no", "Ever told had depressive disorder."),
    FeatureSpec("kidney_disease", ("CHCKDNY2",), "yes_no", "Ever told had kidney disease."),
    FeatureSpec("arthritis", ("HAVARTH4", "_DRDXAR2"), "categorical", "Arthritis/rheumatoid/gout/lupus/fibromyalgia indicator."),
    FeatureSpec("diabetes_status", ("DIABETE4",), "diabetes_status", "Diabetes / prediabetes / gestational diabetes status."),
    FeatureSpec("deaf", ("DEAF",), "yes_no", "Deaf or serious difficulty hearing."),
    FeatureSpec("blind", ("BLIND",), "yes_no", "Blind or serious difficulty seeing."),
    FeatureSpec("cognitive_difficulty", ("DECIDE",), "yes_no", "Difficulty concentrating, remembering, or making decisions."),
    FeatureSpec("walking_difficulty", ("DIFFWALK",), "yes_no", "Serious difficulty walking or climbing stairs."),
    FeatureSpec("dressing_difficulty", ("DIFFDRES",), "yes_no", "Difficulty dressing or bathing."),
    FeatureSpec("independent_living_difficulty", ("DIFFALON",), "yes_no", "Difficulty doing errands alone."),
    FeatureSpec("bmi", ("_BMI5",), "numeric_bmi", "BMI scaled by 100 in BRFSS."),
    FeatureSpec("bmi_category", ("_BMI5CAT", "_RFBMI5"), "categorical", "BMI category/obesity indicator."),
    FeatureSpec("smoking_status", ("_SMOKER3",), "smoking_status", "Computed smoking status."),
    FeatureSpec("smoked_100", ("SMOKE100",), "yes_no", "Smoked at least 100 cigarettes."),
    FeatureSpec("current_smoking_detail", ("SMOKDAY2", "_RFSMOK3"), "categorical", "Current smoking frequency/status."),
    FeatureSpec("smokeless_tobacco", ("USENOW3",), "categorical", "Current smokeless tobacco use."),
    FeatureSpec("ecigarette_use", ("ECIGNOW2", "ECIGNOW3", "_CURECI2", "_CURECI3"), "categorical", "Current e-cigarette use/status."),
    FeatureSpec("alcohol_any", ("DRNKANY6",), "categorical", "Any alcohol use in past 30 days."),
    FeatureSpec("alcohol_days_code", ("ALCDAY4",), "numeric_code", "BRFSS alcohol days code."),
    FeatureSpec("average_drinks", ("AVEDRNK3", "AVEDRNK4"), "numeric_code", "Average drinks on drinking days."),
    FeatureSpec("binge_drinking", ("_RFBING6", "DRNK3GE5"), "categorical", "Binge drinking indicator/code."),
    FeatureSpec("max_drinks", ("MAXDRNKS",), "numeric_code", "Maximum drinks on one occasion."),
    FeatureSpec("heavy_drinking", ("_RFDRHV8", "_RFDRHV9"), "categorical", "Heavy drinking computed indicator."),
    FeatureSpec("flu_vaccine", ("FLUSHOT7", "_FLSHOT7"), "categorical", "Flu vaccination status."),
    FeatureSpec("pneumonia_vaccine", ("PNEUVAC4", "_PNEUMO3"), "categorical", "Pneumonia vaccination status."),
]


def log(lines: list[str], message: str) -> None:
    print(message, flush=True)
    lines.append(message)


def ensure_dirs() -> None:
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    (DIRS["figures"] / "roc_pr").mkdir(parents=True, exist_ok=True)
    (DIRS["figures"] / "calibration").mkdir(parents=True, exist_ok=True)
    (DIRS["figures"] / "thresholding").mkdir(parents=True, exist_ok=True)


def inspect_columns(path: Path) -> list[str]:
    reader = pd.read_sas(path, format="xport", encoding="utf-8", chunksize=1)
    first = next(reader)
    return list(first.columns)


def selected_raw_columns(columns: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    available = set(columns)
    raw_cols = {"CVDSTRK3"}
    inventory: list[dict[str, Any]] = []
    for spec in FEATURE_SPECS:
        chosen = next((col for col in spec.candidates if col in available), None)
        if chosen is not None:
            raw_cols.add(chosen)
        inventory.append(
            {
                "output_feature": spec.output,
                "chosen_raw_variable": chosen or "",
                "available": chosen is not None,
                "candidate_variables": "|".join(spec.candidates),
                "kind": spec.kind,
                "description": spec.description,
            }
        )
    return sorted(raw_cols), inventory


def read_xpt_selected(path: Path, raw_cols: list[str], lines: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    total = 0
    log(lines, f"Reading {path.name} in chunks; selected raw columns: {len(raw_cols)}")
    for i, chunk in enumerate(pd.read_sas(path, format="xport", encoding="utf-8", chunksize=CHUNK_SIZE), start=1):
        cols = [col for col in raw_cols if col in chunk.columns]
        frames.append(chunk[cols].copy())
        total += len(chunk)
        if i == 1 or i % 5 == 0:
            log(lines, f"  chunk {i}: rows read so far = {total:,}")
    df = pd.concat(frames, ignore_index=True)
    log(lines, f"Finished reading {path.name}: selected shape = {df.shape}")
    return df


def missing_codes_to_nan(series: pd.Series, codes: set[float]) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    out = out.mask(out.isin(codes), np.nan)
    return out


def clean_yes_no(series: pd.Series) -> pd.Series:
    raw = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[raw == 1] = 1.0
    out.loc[raw == 2] = 0.0
    return out


def clean_days(series: pd.Series) -> pd.Series:
    raw = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[raw == 88] = 0.0
    out.loc[(raw >= 1) & (raw <= 30)] = raw.loc[(raw >= 1) & (raw <= 30)]
    return out


def clean_children(series: pd.Series) -> pd.Series:
    raw = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out.loc[raw == 88] = 0.0
    out.loc[(raw >= 1) & (raw <= 87)] = raw.loc[(raw >= 1) & (raw <= 87)]
    return out


def clean_code(series: pd.Series) -> pd.Series:
    raw = missing_codes_to_nan(series, {7, 9, 77, 99, 777, 999, 7777, 9999})
    return raw.astype("float64")


def clean_categorical(series: pd.Series) -> pd.Series:
    raw = missing_codes_to_nan(series, {7, 9, 77, 99, 777, 999, 7777, 9999})
    return raw.astype("Int64").astype("string").replace("<NA>", pd.NA)


def clean_bp_status(series: pd.Series) -> pd.Series:
    raw = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    # BPHIGH6 detailed coding when available.
    out.loc[raw == 1] = "yes"
    out.loc[raw == 2] = "pregnancy_only"
    out.loc[raw == 3] = "no"
    out.loc[raw == 4] = "borderline"
    # _RFHYPE6, if used, is less detailed. Keep its code as a category.
    other_valid = raw.notna() & ~raw.isin([1, 2, 3, 4, 7, 9, 77, 99])
    out.loc[other_valid] = "code_" + raw.loc[other_valid].astype("Int64").astype(str)
    return out


def clean_diabetes_status(series: pd.Series) -> pd.Series:
    raw = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    out.loc[raw == 1] = "diabetes"
    out.loc[raw == 2] = "gestational_only"
    out.loc[raw == 3] = "no"
    out.loc[raw == 4] = "prediabetes"
    return out


def clean_smoking_status(series: pd.Series) -> pd.Series:
    raw = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    out.loc[raw == 1] = "current_every_day"
    out.loc[raw == 2] = "current_some_days"
    out.loc[raw == 3] = "former"
    out.loc[raw == 4] = "never"
    return out


def clean_feature(raw_df: pd.DataFrame, spec: FeatureSpec, raw_col: str) -> pd.Series:
    series = raw_df[raw_col]
    if spec.kind == "numeric_age":
        out = missing_codes_to_nan(series, {98, 99})
        return out.where((out >= 18) & (out <= 80), np.nan)
    if spec.kind == "numeric_bmi":
        out = missing_codes_to_nan(series, {9999}) / 100.0
        return out.where((out >= 10) & (out <= 80), np.nan)
    if spec.kind == "numeric_days":
        return clean_days(series)
    if spec.kind == "numeric_children":
        return clean_children(series)
    if spec.kind == "numeric_code":
        return clean_code(series)
    if spec.kind == "yes_no":
        return clean_yes_no(series)
    if spec.kind == "bp_status":
        return clean_bp_status(series)
    if spec.kind == "diabetes_status":
        return clean_diabetes_status(series)
    if spec.kind == "smoking_status":
        return clean_smoking_status(series)
    return clean_categorical(series)


def clean_dataset(dataset: str, path: Path, lines: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = inspect_columns(path)
    raw_cols, inventory = selected_raw_columns(columns)
    raw_df = read_xpt_selected(path, raw_cols, lines)

    inv_df = pd.DataFrame(inventory)
    chosen = {row["output_feature"]: row["chosen_raw_variable"] for row in inventory if row["chosen_raw_variable"]}

    target_raw = pd.to_numeric(raw_df["CVDSTRK3"], errors="coerce")
    clean = pd.DataFrame(index=raw_df.index)
    clean["stroke"] = np.nan
    clean.loc[target_raw == 1, "stroke"] = 1.0
    clean.loc[target_raw == 2, "stroke"] = 0.0

    for spec in FEATURE_SPECS:
        raw_col = chosen.get(spec.output)
        if not raw_col:
            continue
        clean[spec.output] = clean_feature(raw_df, spec, raw_col)

    if "bmi" in clean.columns:
        clean["bmi_missing"] = clean["bmi"].isna().astype(int)

    before = len(clean)
    clean = clean[clean["stroke"].notna()].copy()
    clean["stroke"] = clean["stroke"].astype(int)

    # Keep only features with at least two non-missing values/classes after target filtering.
    kept = ["stroke"]
    for col in clean.columns:
        if col == "stroke":
            continue
        non_missing = clean[col].notna().sum()
        unique = clean[col].dropna().nunique()
        if non_missing >= 2 and unique >= 2:
            kept.append(col)
    clean = clean[kept].copy()

    inv_df.insert(0, "dataset", dataset)
    inv_df["used_in_cleaned_file"] = inv_df["output_feature"].isin(clean.columns)
    inv_df["cleaned_file_note"] = np.where(
        inv_df["used_in_cleaned_file"],
        "available and retained",
        "missing from XPT or dropped because it had insufficient variation",
    )

    log(lines, f"{dataset}: removed {before - len(clean):,} rows with missing/refused target")
    log(lines, f"{dataset}: cleaned shape = {clean.shape}; stroke prevalence = {clean['stroke'].mean():.4f}")
    return clean, inv_df


def split_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_cal, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["stroke"],
        random_state=RANDOM_STATE,
    )
    train, cal = train_test_split(
        train_cal,
        test_size=CAL_SIZE_WITHIN_TRAINCAL,
        stratify=train_cal["stroke"],
        random_state=RANDOM_STATE,
    )
    return {
        "train": train.reset_index(drop=True),
        "calibration": cal.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def make_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    features = [c for c in df.columns if c != "stroke"]
    numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_string_dtype(df[c])]
    categorical_cols = [c for c in features if c not in numeric_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        sparse_threshold=0.3,
    )
    return preprocessor, numeric_cols, categorical_cols


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(y_prob, bins[1:-1], right=True)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            continue
        ece += (mask.sum() / n) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(y_prob, EPS, 1 - EPS)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    lr = LogisticRegression(solver="lbfgs", max_iter=500)
    lr.fit(logit, y_true)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, prevalence_brier: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    brier = brier_score_loss(y_true, y_prob)
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    pr = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    slope, intercept = calibration_slope_intercept(y_true, y_prob)
    return {
        "threshold": threshold,
        "roc_auc": roc,
        "pr_auc": pr,
        "brier": brier,
        "brier_skill_score": 1 - (brier / prevalence_brier) if prevalence_brier > 0 else np.nan,
        "ece_10": expected_calibration_error(y_true, y_prob, 10),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_rate": float(y_pred.mean()),
    }


class SigmoidProbabilityCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(solver="lbfgs", max_iter=500)

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "SigmoidProbabilityCalibrator":
        clipped = np.clip(y_prob, EPS, 1 - EPS)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        self.model.fit(logit, y_true)
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        clipped = np.clip(y_prob, EPS, 1 - EPS)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        return self.model.predict_proba(logit)[:, 1]


class IsotonicProbabilityCalibrator:
    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "IsotonicProbabilityCalibrator":
        self.model.fit(y_prob, y_true)
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(y_prob), dtype=float)


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 2.0) -> tuple[float, float]:
    thresholds = np.linspace(0.001, 0.999, 999)
    best = {"threshold": 0.5, "score": -1.0, "recall": -1.0, "precision": -1.0}
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        score = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        prec = precision_score(y_true, pred, zero_division=0)
        if (
            score > best["score"]
            or (math.isclose(score, best["score"]) and rec > best["recall"])
            or (math.isclose(score, best["score"]) and math.isclose(rec, best["recall"]) and prec > best["precision"])
        ):
            best = {"threshold": float(t), "score": float(score), "recall": float(rec), "precision": float(prec)}
    return best["threshold"], best["score"]


def choose_calibration(cal_rows: list[dict[str, Any]]) -> str:
    candidates = [r for r in cal_rows if r["calibration_method"] in {"uncalibrated", "sigmoid", "isotonic"}]
    best = min(candidates, key=lambda r: (r["brier"], r["ece_10"]))
    # Prefer sigmoid over isotonic when calibration split gains are tiny; this avoids
    # selecting isotonic only because it fitted the calibration split too closely.
    sigmoid = next((r for r in candidates if r["calibration_method"] == "sigmoid"), None)
    if best["calibration_method"] == "isotonic" and sigmoid is not None:
        if (sigmoid["brier"] - best["brier"]) <= 0.0005:
            return "sigmoid"
    return str(best["calibration_method"])


def get_models(pos_weight: float) -> dict[str, Any]:
    return {
        "lr_unbalanced": LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1),
        "lr_balanced": LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1, class_weight="balanced"),
        "xgb_unbalanced": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=180,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgb_weighted": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=180,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def plot_curves(dataset_slug: str, curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (_, test_prob, _) in curves.items():
        fpr, tpr, _ = roc_curve(y_test, test_prob)
        ax.plot(fpr, tpr, label=f"{label} AUC={roc_auc_score(y_test, test_prob):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{dataset_slug} ROC Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "roc_pr" / f"{dataset_slug}_roc_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    prevalence = float(y_test.mean())
    ax.axhline(prevalence, linestyle="--", color="gray", label=f"Prevalence={prevalence:.3f}")
    for label, (_, test_prob, _) in curves.items():
        precision, recall, _ = precision_recall_curve(y_test, test_prob)
        ax.plot(recall, precision, label=f"{label} AP={average_precision_score(y_test, test_prob):.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{dataset_slug} Precision-Recall Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DIRS["figures"] / "roc_pr" / f"{dataset_slug}_pr_curves.png", dpi=180)
    plt.close(fig)


def run_modeling(dataset: str, df: pd.DataFrame, lines: list[str]) -> dict[str, pd.DataFrame]:
    slug = dataset.lower().replace(" ", "_")
    df = df.copy()
    # sklearn imputers/encoders expect np.nan rather than pandas.NA.
    df = df.replace({pd.NA: np.nan})
    for col in df.columns:
        if col == "stroke":
            continue
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    splits = split_dataset(df)
    y_train = splits["train"]["stroke"].to_numpy()
    y_cal = splits["calibration"]["stroke"].to_numpy()
    y_test = splits["test"]["stroke"].to_numpy()
    x_train = splits["train"].drop(columns=["stroke"])
    x_cal = splits["calibration"].drop(columns=["stroke"])
    x_test = splits["test"].drop(columns=["stroke"])

    preprocessor, numeric_cols, categorical_cols = make_preprocessor(splits["train"])
    x_train_p = preprocessor.fit_transform(x_train)
    x_cal_p = preprocessor.transform(x_cal)
    x_test_p = preprocessor.transform(x_test)

    train_pos = int(y_train.sum())
    train_neg = int(len(y_train) - train_pos)
    pos_weight = train_neg / train_pos if train_pos else 1.0
    prevalence_brier = brier_score_loss(y_test, np.repeat(y_train.mean(), len(y_test)))

    log(lines, f"{dataset}: train/cal/test = {len(y_train):,}/{len(y_cal):,}/{len(y_test):,}; train prevalence = {y_train.mean():.4f}")
    log(lines, f"{dataset}: numeric features = {len(numeric_cols)}, categorical features = {len(categorical_cols)}, scale_pos_weight = {pos_weight:.3f}")

    dataset_summary = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "split": split,
                "n": len(part),
                "stroke_cases": int(part["stroke"].sum()),
                "non_stroke_cases": int((part["stroke"] == 0).sum()),
                "stroke_prevalence": float(part["stroke"].mean()),
                "features_used": len(df.columns) - 1,
                "numeric_features": len(numeric_cols),
                "categorical_features": len(categorical_cols),
            }
            for split, part in splits.items()
        ]
    )

    baseline_rows: list[dict[str, Any]] = []
    majority_prob = np.zeros_like(y_test, dtype=float)
    prevalence_prob = np.repeat(y_train.mean(), len(y_test))
    for baseline_name, prob in [("majority_no_stroke", majority_prob), ("train_prevalence_probability", prevalence_prob)]:
        row = {
            "dataset": dataset,
            "model_id": baseline_name,
            "model_family": "baseline",
            "calibration_method": "not_applicable",
            "threshold_source": "fixed_0_5",
            **classification_metrics(y_test, prob, 0.5, prevalence_brier),
        }
        baseline_rows.append(row)

    model_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    selected_artifacts: list[dict[str, Any]] = []
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    models = get_models(pos_weight)
    for model_id, model in models.items():
        log(lines, f"{dataset}: training {model_id}")
        model.fit(x_train_p, y_train)
        cal_raw = np.asarray(model.predict_proba(x_cal_p)[:, 1], dtype=float)
        test_raw = np.asarray(model.predict_proba(x_test_p)[:, 1], dtype=float)

        calibrators: dict[str, Any] = {
            "uncalibrated": None,
            "sigmoid": SigmoidProbabilityCalibrator().fit(cal_raw, y_cal),
            "isotonic": IsotonicProbabilityCalibrator().fit(cal_raw, y_cal),
        }

        cal_method_rows: list[dict[str, Any]] = []
        method_probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for method, calibrator in calibrators.items():
            cal_prob = cal_raw if calibrator is None else calibrator.predict(cal_raw)
            test_prob = test_raw if calibrator is None else calibrator.predict(test_raw)
            method_probs[method] = (cal_prob, test_prob)

            cal_metrics = classification_metrics(y_cal, cal_prob, 0.5, brier_score_loss(y_cal, np.repeat(y_train.mean(), len(y_cal))))
            cal_row = {
                "dataset": dataset,
                "split": "calibration",
                "model_id": model_id,
                "model_family": "Logistic Regression" if model_id.startswith("lr") else "XGBoost",
                "calibration_method": method,
                **cal_metrics,
            }
            calibration_rows.append(cal_row)
            cal_method_rows.append(cal_row)

            test_metrics_05 = classification_metrics(y_test, test_prob, 0.5, prevalence_brier)
            model_rows.append(
                {
                    "dataset": dataset,
                    "split": "test",
                    "model_id": model_id,
                    "model_family": "Logistic Regression" if model_id.startswith("lr") else "XGBoost",
                    "calibration_method": method,
                    "threshold_source": "fixed_0_5",
                    **test_metrics_05,
                }
            )

        selected_method = choose_calibration(cal_method_rows)
        selected_cal_prob, selected_test_prob = method_probs[selected_method]
        selected_threshold, _ = optimize_threshold(y_cal, selected_cal_prob, beta=2.0)
        threshold_rows.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "model_family": "Logistic Regression" if model_id.startswith("lr") else "XGBoost",
                "selected_calibration_method": selected_method,
                "threshold_source": "calibration_argmax_f2",
                **classification_metrics(y_test, selected_test_prob, selected_threshold, prevalence_brier),
            }
        )

        final_rows.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "model_family": "Logistic Regression" if model_id.startswith("lr") else "XGBoost",
                "selected_calibration_method": selected_method,
                "selected_threshold": selected_threshold,
                **classification_metrics(y_test, selected_test_prob, selected_threshold, prevalence_brier),
            }
        )

        curves[f"{model_id}/{selected_method}"] = (selected_cal_prob, selected_test_prob, np.array([selected_threshold]))
        selected_artifacts.append(
            {
                "model_id": model_id,
                "model": model,
                "calibrator": calibrators[selected_method],
                "calibration_method": selected_method,
                "threshold": selected_threshold,
                "preprocessor": preprocessor,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
            }
        )

    final_df = pd.DataFrame(final_rows)
    final_df["selection_rank_score"] = (
        final_df["pr_auc"].rank(ascending=False, method="min") * 100
        + final_df["f2"].rank(ascending=False, method="min") * 10
        + final_df["brier"].rank(ascending=True, method="min")
    )
    best_row = final_df.sort_values(["pr_auc", "f2", "brier"], ascending=[False, False, True]).iloc[0]
    final_df["final_selection_status"] = np.where(final_df["model_id"] == best_row["model_id"], "selected_final", "not_selected")
    final_df["selection_reason"] = np.where(
        final_df["model_id"] == best_row["model_id"],
        "Selected by PR-AUC first, then F2, then Brier score; accuracy was not primary.",
        "Not selected in this side experiment because another model had a stronger reliability-aware test profile.",
    )

    best_artifact = next(a for a in selected_artifacts if a["model_id"] == best_row["model_id"])
    joblib.dump(best_artifact, DIRS["models"] / f"{slug}_selected_final_model.joblib")
    plot_curves(slug, curves, y_test)

    return {
        "dataset_summary": dataset_summary,
        "baseline": pd.DataFrame(baseline_rows),
        "model_comparison": pd.DataFrame(model_rows),
        "calibration": pd.DataFrame(calibration_rows),
        "threshold": pd.DataFrame(threshold_rows),
        "final": final_df.drop(columns=["selection_rank_score"]),
    }


def write_report(
    dataset_summaries: pd.DataFrame,
    feature_inventory: pd.DataFrame,
    final_comparison: pd.DataFrame,
    selected_models: pd.DataFrame,
) -> None:
    def simple_markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        display = df.copy()
        for col in display.columns:
            if pd.api.types.is_numeric_dtype(display[col]):
                display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}" if isinstance(x, float) else str(x))
            else:
                display[col] = display[col].fillna("").astype(str)
        columns = list(display.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for _, row in display.iterrows():
            lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in columns) + " |")
        return "\n".join(lines)

    lines = [
        "# Separate BRFSS Rich-Feature Experiment Report",
        "",
        "This is an isolated exploratory experiment using only files inside `seprate experiment/`.",
        "It does not replace the accepted thesis pipeline.",
        "",
        "## Datasets",
        "",
        simple_markdown_table(dataset_summaries),
        "",
        "## Feature Handling",
        "",
        f"Candidate semantic features considered: {feature_inventory['output_feature'].nunique()}",
        f"Raw variables differ by year, so the feature inventory records which variables were available and retained.",
        "",
        "## Selected Final Models",
        "",
        simple_markdown_table(selected_models),
        "",
        "## Final Model Comparison",
        "",
        simple_markdown_table(final_comparison[
            [
                "dataset",
                "model_id",
                "selected_calibration_method",
                "selected_threshold",
                "roc_auc",
                "pr_auc",
                "brier",
                "brier_skill_score",
                "ece_10",
                "recall",
                "precision",
                "specificity",
                "f2",
                "final_selection_status",
            ]
        ].round(4)),
        "",
        "## Interpretation Guardrails",
        "",
        "- This side experiment uses richer year-specific BRFSS features and is not the same as the thesis common-core model.",
        "- Calibration is selected using the calibration split.",
        "- F2 thresholds are selected using the calibration split and evaluated on the test split.",
        "- Accuracy is not used as the primary model-selection metric.",
        "- Because 2023 and 2024 questionnaires differ, feature availability is documented explicitly.",
    ]
    (OUT / "SEPARATE_EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    start = time.time()
    lines: list[str] = ["Separate BRFSS rich-feature stroke experiment", "=" * 48]
    all_clean: list[pd.DataFrame] = []
    all_inventory: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    baselines: list[pd.DataFrame] = []
    model_tables: list[pd.DataFrame] = []
    calibration_tables: list[pd.DataFrame] = []
    threshold_tables: list[pd.DataFrame] = []
    final_tables: list[pd.DataFrame] = []

    for dataset, path in RAW_FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        slug = dataset.lower().replace(" ", "_")
        clean_path = DIRS["processed"] / f"{slug}_stroke_rich_clean.csv"
        inventory_path = DIRS["tables"] / f"{slug}_feature_inventory.csv"
        if clean_path.exists() and inventory_path.exists():
            log(lines, f"\n=== Loading cached cleaned {dataset} ===")
            clean = pd.read_csv(clean_path)
            inventory = pd.read_csv(inventory_path)
            log(lines, f"{dataset}: loaded cached cleaned shape = {clean.shape}")
        else:
            log(lines, f"\n=== Cleaning {dataset} ===")
            clean, inventory = clean_dataset(dataset, path, lines)
            clean.to_csv(clean_path, index=False)
            log(lines, f"{dataset}: saved cleaned data to {clean_path}")
            inventory.to_csv(inventory_path, index=False)
        all_clean.append(clean.assign(dataset=dataset))
        all_inventory.append(inventory)

        log(lines, f"\n=== Modeling {dataset} ===")
        outputs = run_modeling(dataset, clean, lines)
        summaries.append(outputs["dataset_summary"])
        baselines.append(outputs["baseline"])
        model_tables.append(outputs["model_comparison"])
        calibration_tables.append(outputs["calibration"])
        threshold_tables.append(outputs["threshold"])
        final_tables.append(outputs["final"])

    dataset_summary = pd.concat(summaries, ignore_index=True)
    feature_inventory = pd.concat(all_inventory, ignore_index=True)
    baseline_results = pd.concat(baselines, ignore_index=True)
    model_comparison = pd.concat(model_tables, ignore_index=True)
    calibration_results = pd.concat(calibration_tables, ignore_index=True)
    threshold_results = pd.concat(threshold_tables, ignore_index=True)
    final_comparison = pd.concat(final_tables, ignore_index=True)
    selected_models = final_comparison[final_comparison["final_selection_status"] == "selected_final"].copy()

    dataset_summary.to_csv(DIRS["tables"] / "table_01_dataset_summary.csv", index=False)
    feature_inventory.to_csv(DIRS["tables"] / "table_02_feature_inventory.csv", index=False)
    baseline_results.to_csv(DIRS["tables"] / "table_03_baseline_results.csv", index=False)
    model_comparison.to_csv(DIRS["tables"] / "table_04_model_comparison_threshold_0_5.csv", index=False)
    calibration_results.to_csv(DIRS["tables"] / "table_05_calibration_results.csv", index=False)
    threshold_results.to_csv(DIRS["tables"] / "table_06_threshold_results.csv", index=False)
    final_comparison.to_csv(DIRS["tables"] / "table_07_final_model_comparison.csv", index=False)
    selected_models.to_csv(DIRS["tables"] / "table_08_selected_final_models.csv", index=False)

    write_report(dataset_summary, feature_inventory, final_comparison, selected_models)

    manifest = {
        "generated_seconds": round(time.time() - start, 2),
        "root": str(ROOT),
        "tables": sorted(p.name for p in DIRS["tables"].glob("*.csv")),
        "figures": sorted(str(p.relative_to(ROOT)) for p in DIRS["figures"].rglob("*") if p.is_file()),
        "models": sorted(p.name for p in DIRS["models"].glob("*.joblib")),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log(lines, f"\nCompleted in {time.time() - start:.2f} seconds")
    log(lines, "All outputs were written inside the separate experiment folder.")
    (DIRS["logs"] / "separate_experiment_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
