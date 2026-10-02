"""Main analysis: the numbers, tables and data figures of the main text.

Seed-42 stratified 60/10/10/20 split; XGBoost and logistic regression on the full (55) and
pre-event (37) predictor sets; bootstrap CIs, survey-weighted estimates, threshold tiers,
decision curves, the 2024 temporal validation and SHAP.
Needs outputs/brfss_20XX_weights_aligned.parquet (extract_survey_weights.py).
"""
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJ = ROOT.parent
STAGE2 = PROJ / "simillar reasearch test" / "rich_feature_best_4way_final_search"
OUT = HERE / "outputs"
# Default: corrected files (rebuild_clean_data.py). CMPB_DATA=v1 uses the original
# cleaning and reproduces the thesis result.
DATA_VERSION = os.environ.get("CMPB_DATA", "v2")
CACHE = STAGE2 / "outputs" / "data_cache" if DATA_VERSION == "v1" else OUT / "data_cache_v2"
FIG = ROOT / "manuscript_cmpb" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
if hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SEED = 42
N_BOOT = 1000
N_BOOT_TEMPORAL = 500
TARGET = "stroke"
ONLY_2023 = ["hypertension_status", "cholesterol_status", "cholesterol_meds"]
TIER_A = ["walking_difficulty", "dressing_difficulty", "independent_living_difficulty",
          "cognitive_difficulty", "blind"]
TIER_B = ["general_health", "physical_health_days", "mental_health_days", "poor_health_days",
          "employment", "depression", "cholesterol_meds", "exercise_any", "deaf"]
TIER_C = ["recent_checkup", "personal_doctor", "flu_vaccine", "pneumonia_vaccine"]
POST = TIER_A + TIER_B + TIER_C
TIERS = {"broad": 0.80, "primary": 0.60, "focused": 0.40}

# plot colours
C_FULL, C_PRE, C_LR = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 300})


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# pipeline
def stage2_params():
    p = pd.read_csv(STAGE2 / "outputs" / "tables" / "T04_optuna_best_params.csv").set_index("label").loc["MODE_A_R0"]
    keys = ["max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight",
            "gamma", "reg_alpha", "reg_lambda", "scale_pos_weight"]
    d = {k: float(p[k]) for k in keys}
    d["max_depth"], d["min_child_weight"] = int(d["max_depth"]), int(d["min_child_weight"])
    return d


def four_way_split(df):
    y = df[TARGET].values; idx = np.arange(len(df))
    rest, te = train_test_split(idx, test_size=0.20, stratify=y, random_state=SEED)
    tc, th = train_test_split(rest, test_size=0.125, stratify=y[rest], random_state=SEED)
    tr, ca = train_test_split(tc, test_size=1 / 7, stratify=y[tc], random_state=SEED)
    return tr, ca, th, te


def preprocessor(df, feats):
    cat = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c].dtype)]
    num = [c for c in feats if c not in cat]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first",
                                                dtype=np.float32, sparse_output=False))]), cat),
    ], remainder="drop", sparse_threshold=0).fit(df[feats])


THR_GRID = np.linspace(0.005, 0.70, 1500)


def thr_at_recall(y, p, target):
    yp = p[None, :] >= THR_GRID[:, None]
    tp = yp[:, y == 1].sum(1).astype(float); fp = yp[:, y == 0].sum(1).astype(float)
    rec = tp / y.sum(); prec = tp / np.maximum(tp + fp, 1e-9)
    ok = np.where(rec >= target)[0]
    return float(THR_GRID[ok[np.argmax(prec[ok])]])


class Model:
    """Fit on train, early-stop/calibrate on cal (lower Brier of sigmoid/isotonic), thresholds on thresh."""

    def __init__(self, kind, feats, params):
        self.kind, self.feats, self.params = kind, feats, params

    def fit(self, tr, ca, th):
        self.ct = preprocessor(tr, self.feats)
        X = lambda d: self.ct.transform(d[self.feats]).astype(np.float32)
        Xtr, Xca = X(tr), X(ca)
        ytr, yca = tr[TARGET].values, ca[TARGET].values
        if self.kind == "xgb":
            self.m = XGBClassifier(**self.params, n_estimators=2000, tree_method="hist", random_state=SEED,
                                   eval_metric="aucpr", early_stopping_rounds=50, verbosity=0, n_jobs=-1)
            self.m.fit(Xtr, ytr, eval_set=[(Xca, yca)], verbose=False)
        else:
            self.m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(Xtr, ytr)
        raw = self.m.predict_proba(Xca)[:, 1]
        platt = LogisticRegression(C=1.0, max_iter=2000).fit(raw.reshape(-1, 1), yca)
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw, yca)
        sig = lambda r: platt.predict_proba(r.reshape(-1, 1))[:, 1]
        isf = lambda r: np.clip(iso.predict(r), 0, 1)
        if brier_score_loss(yca, sig(raw)) <= brier_score_loss(yca, isf(raw)):
            self.cal, self.cal_name, self.cal_obj = sig, "sigmoid", platt
        else:
            self.cal, self.cal_name, self.cal_obj = isf, "isotonic", iso
        self.X = X
        pth = self.predict(th)
        self.thr = {k: thr_at_recall(th[TARGET].values, pth, r) for k, r in TIERS.items()}
        return self

    def predict(self, d):
        return self.cal(self.m.predict_proba(self.X(d))[:, 1])


# metrics
def ece(y, p, w=None, bins=10):
    w = np.ones_like(p) if w is None else w
    edges = np.linspace(0, 1, bins + 1); tot = 0.0; W = w.sum()
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1 else (p >= edges[i])
        if m.any():
            tot += w[m].sum() / W * abs(np.average(y[m], weights=w[m]) - np.average(p[m], weights=w[m]))
    return tot


def cal_slope_intercept(y, p, w=None):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    slope = LogisticRegression(C=1e6, max_iter=2000).fit(lp.reshape(-1, 1), y, sample_weight=w).coef_[0, 0]
    # calibration-in-the-large: intercept with the slope fixed at 1 (offset), via 1-D Newton
    a = 0.0
    ww = np.ones_like(p) if w is None else w
    for _ in range(50):
        q = 1 / (1 + np.exp(-(lp + a)))
        g = np.sum(ww * (y - q)); h = np.sum(ww * q * (1 - q))
        a += g / h
        if abs(g / h) < 1e-10:
            break
    return float(slope), float(a)


def metrics(y, p, t, w=None):
    ww = np.ones_like(p) if w is None else w
    pred = p >= t
    tp = ww[pred & (y == 1)].sum(); fp = ww[pred & (y == 0)].sum()
    fn = ww[~pred & (y == 1)].sum(); tn = ww[~pred & (y == 0)].sum()
    sens = tp / (tp + fn); spec = tn / (tn + fp); ppv = tp / max(tp + fp, 1e-12); npv = tn / max(tn + fn, 1e-12)
    slope, icpt = cal_slope_intercept(y, p, w)
    return dict(threshold=t, roc_auc=roc_auc_score(y, p, sample_weight=w),
                pr_auc=average_precision_score(y, p, sample_weight=w),
                brier=brier_score_loss(y, p, sample_weight=w), ece=ece(y, p, w),
                cal_slope=slope, cal_intercept=icpt,
                sensitivity=sens, specificity=spec, ppv=ppv, npv=npv,
                f2=5 * ppv * sens / max(4 * ppv + sens, 1e-12),
                fp_per_tp=fp / max(tp, 1e-12), flagged_per_1000=1000 * (tp + fp) / ww.sum(),
                prevalence=np.average(y, weights=ww))


BOOT_KEYS = ["roc_auc", "pr_auc", "brier", "ece", "sensitivity", "specificity", "ppv", "npv", "f2", "fp_per_tp"]


def boot_ci(y, p, t, n_boot, seed=SEED):
    rng = np.random.default_rng(seed); n = len(y); acc = {k: [] for k in BOOT_KEYS}
    for _ in range(n_boot):
        i = rng.integers(0, n, n); yi, pi = y[i], p[i]
        pred = pi >= t
        tp = np.sum(pred & (yi == 1)); fp = np.sum(pred & (yi == 0))
        fn = np.sum(~pred & (yi == 1)); tn = np.sum(~pred & (yi == 0))
        sens, spec, ppv = tp / (tp + fn), tn / (tn + fp), tp / max(tp + fp, 1)
        acc["roc_auc"].append(roc_auc_score(yi, pi)); acc["pr_auc"].append(average_precision_score(yi, pi))
        acc["brier"].append(np.mean((pi - yi) ** 2)); acc["ece"].append(ece(yi, pi))
        acc["sensitivity"].append(sens); acc["specificity"].append(spec); acc["ppv"].append(ppv)
        acc["npv"].append(tn / max(tn + fn, 1)); acc["f2"].append(5 * ppv * sens / max(4 * ppv + sens, 1e-12))
        acc["fp_per_tp"].append(fp / max(tp, 1))
    return {f"{k}_lo": float(np.percentile(v, 2.5)) for k, v in acc.items()} | \
           {f"{k}_hi": float(np.percentile(v, 97.5)) for k, v in acc.items()}


def net_benefit(y, p, pts):
    n = len(y); out = []
    for pt in pts:
        pred = p >= pt
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        out.append(tp / n - fp / n * pt / (1 - pt))
    return np.array(out)


# run
def main():
    t0 = time.time()
    params = stage2_params()
    d23 = pd.read_parquet(CACHE / "brfss_2023_stroke_rich_clean.parquet")
    d24 = pd.read_parquet(CACHE / "brfss_2024_stroke_rich_clean.parquet")
    w23 = pd.read_parquet(OUT / "brfss_2023_weights_aligned.parquet")["_LLCPWT"].values
    w24 = pd.read_parquet(OUT / "brfss_2024_weights_aligned.parquet")["_LLCPWT"].values
    full = [c for c in d23.columns if c != TARGET]
    pre = [c for c in full if c not in POST]
    full_t = [c for c in full if c not in ONLY_2023]
    pre_t = [c for c in pre if c not in ONLY_2023]
    assert len(full) == 55 and len(pre) == 37, (len(full), len(pre))
    tr_i, ca_i, th_i, te_i = four_way_split(d23)
    tr, ca, th, te = (d23.iloc[i].reset_index(drop=True) for i in (tr_i, ca_i, th_i, te_i))
    y_te, w_te = te[TARGET].values, w23[te_i]
    y24 = d24[TARGET].values

    numbers = {"n_2023": len(d23), "cases_2023": int(d23[TARGET].sum()), "prev_2023": float(d23[TARGET].mean()),
               "n_2024": len(d24), "cases_2024": int(y24.sum()), "prev_2024": float(y24.mean()),
               "weighted_prev_2023": float(np.average(d23[TARGET], weights=w23)),
               "weighted_prev_2024": float(np.average(y24, weights=w24)),
               "n_features": {"full": len(full), "pre": len(pre), "full_2024": len(full_t), "pre_2024": len(pre_t)},
               "pre_event_removed": POST, "pre_event_kept": pre}
    split_rows = []
    for name, d in zip(("train", "calibration", "threshold", "test"), (tr, ca, th, te)):
        split_rows.append(dict(split=name, n=len(d), cases=int(d[TARGET].sum()), prevalence=d[TARGET].mean()))
    split_rows.append(dict(split="BRFSS 2024 (temporal)", n=len(d24), cases=int(y24.sum()), prevalence=y24.mean()))
    pd.DataFrame(split_rows).to_csv(OUT / "table1_splits.csv", index=False)

    specs = {"xgb_full": ("xgb", full), "xgb_pre": ("xgb", pre), "lr_full": ("lr", full), "lr_pre": ("lr", pre)}
    models, preds, rows = {}, {}, []
    for key, (kind, feats) in specs.items():
        m = Model(kind, feats, params).fit(tr, ca, th); models[key] = m
        p = m.predict(te); preds[key] = p
        t = m.thr["primary"]
        base = dict(model=key, n_features=len(feats), calibration=m.cal_name)
        r = base | {"eval": "test_2023"} | metrics(y_te, p, t) | boot_ci(y_te, p, t, N_BOOT)
        rows.append(r)
        rows.append(base | {"eval": "test_2023_weighted"} | metrics(y_te, p, t, w_te))
        log(f"{key}: n={len(feats)} {m.cal_name} ROC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} "
            f"t={t:.4f} sens={r['sensitivity']:.3f} ppv={r['ppv']:.3f} F2={r['f2']:.4f}")
    xf = [r for r in rows if r["model"] == "xgb_full" and r["eval"] == "test_2023"][0]
    if DATA_VERSION == "v1":
        assert abs(xf["roc_auc"] - 0.8326) < 0.002 and abs(xf["threshold"] - 0.0866) < 0.001, \
            "full XGBoost does not reproduce the published Stage 2 result"
        log("sanity check: published Stage 2 result reproduced")
    else:
        log(f"corrected data: full XGBoost ROC={xf['roc_auc']:.4f} t={xf['threshold']:.4f} "
            f"(Stage 2 on uncorrected data: 0.8326, 0.0866)")

    # temporal transfer
    for key, feats in (("xgb_full_2024feats", full_t), ("xgb_pre_2024feats", pre_t)):
        m = Model("xgb", feats, params).fit(tr, ca, th); models[key] = m
        t = m.thr["primary"]
        p23 = m.predict(te); p24 = m.predict(d24); preds[key + "_2024"] = p24
        base = dict(model=key, n_features=len(feats), calibration=m.cal_name)
        rows.append(base | {"eval": "test_2023"} | metrics(y_te, p23, t))
        r = base | {"eval": "temporal_2024"} | metrics(y24, p24, t) | boot_ci(y24, p24, t, N_BOOT_TEMPORAL)
        rows.append(r)
        rows.append(base | {"eval": "temporal_2024_weighted"} | metrics(y24, p24, t, w24))
        log(f"{key}: 2024 ROC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} sens={r['sensitivity']:.3f} "
            f"ppv={r['ppv']:.3f} ECE={r['ece']:.4f} slope={r['cal_slope']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "table2_performance.csv", index=False)

    # threshold tiers
    tier_rows = []
    for key in ("xgb_full", "xgb_pre"):
        for tier, target in TIERS.items():
            t = models[key].thr[tier]
            m = metrics(y_te, preds[key], t)
            tier_rows.append(dict(model=key, tier=tier, recall_target=target) | m |
                                 {"tp_per_1000": 1000 * np.sum((preds[key] >= t) & (y_te == 1)) / len(y_te),
                                  "fp_per_1000": 1000 * np.sum((preds[key] >= t) & (y_te == 0)) / len(y_te)})
    pd.DataFrame(tier_rows).to_csv(OUT / "table3_tiers.csv", index=False)

    # DCA
    pts = np.round(np.arange(0.01, 0.301, 0.005), 3)
    prev = y_te.mean()
    dca = pd.DataFrame({"pt": pts, "treat_all": prev - (1 - prev) * pts / (1 - pts), "treat_none": 0.0})
    for key in ("xgb_full", "xgb_pre", "lr_pre"):
        dca[key] = net_benefit(y_te, preds[key], pts)
    dca.to_csv(OUT / "dca.csv", index=False)
    for key in ("xgb_full", "xgb_pre"):
        t = models[key].thr["primary"]
        numbers[f"nb_{key}_at_threshold"] = float(net_benefit(y_te, preds[key], [t])[0])
        numbers[f"nb_treat_all_at_{key}_threshold"] = float(prev - (1 - prev) * t / (1 - t))

    # SHAP (raw XGBoost margin, 5,000 random test rows)
    rng = np.random.default_rng(SEED); samp = rng.choice(len(te), 5000, replace=False)
    shap_tabs = {}
    for key in ("xgb_full", "xgb_pre"):
        m = models[key]; Xs = m.X(te.iloc[samp])
        sv = shap.TreeExplainer(m.m).shap_values(Xs)
        names = m.ct.get_feature_names_out()
        agg = {}
        for j, nm in enumerate(names):
            base_nm = nm.split("__", 1)[1]
            orig = next((f for f in sorted(m.feats, key=len, reverse=True)
                         if base_nm == f or base_nm.startswith(f + "_")), base_nm)
            agg[orig] = agg.get(orig, 0.0) + np.abs(sv[:, j]).mean()
        s = pd.Series(agg).sort_values(ascending=False); shap_tabs[key] = s
        s.rename("mean_abs_shap").to_csv(OUT / f"shap_{key}.csv")

    numbers["models"] = {r["model"] + "|" + r["eval"]: {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                                        for k, v in r.items()} for r in rows}
    (OUT / "paper_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))

    # Test-set predictions for the figure scripts (make_figures.py draws Figures 2-3 and 5)
    pd.DataFrame({"y": y_te} | {k: preds[k] for k in ("xgb_full", "xgb_pre", "lr_full", "lr_pre")}) \
        .to_parquet(OUT / "test_predictions_2023.parquet", index=False)
    pd.Series({k: m.thr["primary"] for k, m in models.items()}, name="primary_threshold") \
        .to_csv(OUT / "primary_thresholds.csv")

    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
