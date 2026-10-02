"""Repeat the original Optuna search (TPE, seed 42, 100 trials, same ranges and objective) on the
corrected data, refit both XGBoost models with the new parameters, and compare them with the
reported models by paired bootstrap.
"""
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_paper_analysis as rpa  # noqa: E402

N_TRIALS = 100
optuna.logging.set_verbosity(optuna.logging.WARNING)


def composite_score(y_true, y_proba):
    """Same objective as the original search."""
    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    thr = np.linspace(0.01, 0.50, 300)
    yp = y_proba[np.newaxis, :] >= thr[:, np.newaxis]
    pos = y_true.sum()
    tp = yp[:, y_true == 1].sum(axis=1).astype(float)
    fp = yp[:, y_true == 0].sum(axis=1).astype(float)
    prec = tp / (tp + fp + 1e-9); rec = tp / (pos + 1e-9)
    f2 = 5 * prec * rec / (4 * prec + rec + 1e-9)
    mask60 = rec >= 0.60
    p60 = float(prec[mask60].max()) if mask60.any() else 0.0
    return 0.35 * pr_auc + 0.30 * float(f2.max()) + 0.15 * p60 + 0.10 * roc_auc + 0.10 * max(0.0, 1.0 - brier * 10)


def main():
    t0 = time.time()
    d = pd.read_parquet(rpa.CACHE / "brfss_2023_stroke_rich_clean.parquet")
    full = [c for c in d.columns if c != rpa.TARGET]
    pre = [c for c in full if c not in rpa.POST]
    tr_i, ca_i, th_i, te_i = rpa.four_way_split(d)
    tr, ca, th, te = (d.iloc[i].reset_index(drop=True) for i in (tr_i, ca_i, th_i, te_i))
    ct = rpa.preprocessor(tr, full)
    Xtr, Xca = (ct.transform(x[full]).astype(np.float32) for x in (tr, ca))
    ytr, yca = tr[rpa.TARGET].values, ca[rpa.TARGET].values

    def objective(trial):
        params = dict(max_depth=trial.suggest_int("max_depth", 3, 10),
                      learning_rate=trial.suggest_float("learning_rate", 0.005, 0.30, log=True),
                      subsample=trial.suggest_float("subsample", 0.5, 1.0),
                      colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                      min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
                      gamma=trial.suggest_float("gamma", 0.0, 5.0),
                      reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                      reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                      scale_pos_weight=trial.suggest_float("scale_pos_weight", 5.0, 30.0))
        m = XGBClassifier(**params, n_estimators=2000, tree_method="hist", random_state=rpa.SEED, eval_metric="aucpr",
                          early_stopping_rounds=50, verbosity=0)
        m.fit(Xtr, ytr, eval_set=[(Xca, yca)], verbose=False)
        return composite_score(yca, m.predict_proba(Xca)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=rpa.SEED, n_startup_trials=max(10, N_TRIALS // 5)))
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=False)
    new = dict(study.best_params)
    old = rpa.stage2_params()
    rpa.log(f"search done in {(time.time() - t0) / 60:.1f} min; best composite {study.best_value:.4f}")
    pd.DataFrame({"original_T04": pd.Series(old), "retuned_corrected_data": pd.Series(new)}) \
        .to_csv(rpa.OUT / "retune_params.csv")

    pr = pd.read_parquet(rpa.OUT / "test_predictions_2023.parquet")
    y = te[rpa.TARGET].values
    assert np.array_equal(pr["y"].values, y)
    rows = []
    for key, feats in (("xgb_pre", pre), ("xgb_full", full)):
        m = rpa.Model("xgb", feats, new).fit(tr, ca, th)
        p_new, p_old = m.predict(te), pr[key].values
        met = rpa.metrics(y, p_new, m.thr["primary"])
        rng = np.random.default_rng(rpa.SEED); d_auc, d_ap = [], []
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            d_auc.append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_old[i]))
            d_ap.append(average_precision_score(y[i], p_new[i]) - average_precision_score(y[i], p_old[i]))
        rows.append(dict(model=key, n_features=len(feats), calibration=m.cal_name,
                         roc_auc_original=roc_auc_score(y, p_old), roc_auc_retuned=met["roc_auc"],
                         delta_roc_auc=met["roc_auc"] - roc_auc_score(y, p_old),
                         delta_roc_lo=np.percentile(d_auc, 2.5), delta_roc_hi=np.percentile(d_auc, 97.5),
                         pr_auc_original=average_precision_score(y, p_old), pr_auc_retuned=met["pr_auc"],
                         delta_pr_lo=np.percentile(d_ap, 2.5), delta_pr_hi=np.percentile(d_ap, 97.5),
                         threshold_retuned=met["threshold"], sensitivity_retuned=met["sensitivity"],
                         specificity_retuned=met["specificity"], ppv_retuned=met["ppv"], fp_per_tp_retuned=met["fp_per_tp"],
                         cal_slope_retuned=met["cal_slope"], brier_retuned=met["brier"]))
        rpa.log(f"{key}: ROC original {rows[-1]['roc_auc_original']:.4f} -> retuned {met['roc_auc']:.4f} "
                f"(delta {rows[-1]['delta_roc_auc']:+.4f}, {rows[-1]['delta_roc_lo']:+.4f} to {rows[-1]['delta_roc_hi']:+.4f}); "
                f"sens {met['sensitivity']:.3f} ppv {met['ppv']:.3f}")
    pd.DataFrame(rows).to_csv(rpa.OUT / "retune_comparison.csv", index=False)
    rpa.log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
