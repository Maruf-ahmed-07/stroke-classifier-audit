"""Write LaTeX tables for the supplementary material directly from the analysis CSVs."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent
A = ROOT / "analysis" / "outputs"
R = ROOT / "robustness_experiments" / "outputs"
DST = ROOT / "manuscript_cmpb" / "supp_tables.tex"
DST.parent.mkdir(parents=True, exist_ok=True)
CACHE = Path(__file__).resolve().parent / "outputs" / "data_cache_v2"  # corrected codes (rebuild_clean_data.py)

TIER_A = ["walking_difficulty", "dressing_difficulty", "independent_living_difficulty", "cognitive_difficulty", "blind"]
TIER_B = ["general_health", "physical_health_days", "mental_health_days", "poor_health_days", "employment",
          "depression", "cholesterol_meds", "exercise_any", "deaf"]
TIER_C = ["recent_checkup", "personal_doctor", "flu_vaccine", "pneumonia_vaccine"]
ONLY_2023 = ["hypertension_status", "cholesterol_status", "cholesterol_meds"]


def tex(x):
    return str(x).replace("R: paper pipeline", "Proposed pipeline").replace("_", r"\_\allowbreak{}").replace("%", r"\%").replace("&", r"\&").replace("/", r"/\allowbreak{}")


def f(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


out = []

# S1 predictors
inv = pd.read_csv(PROJ / "seprate experiment" / "outputs" / "tables" / "table_02_feature_inventory.csv")
inv = inv[inv.dataset == "BRFSS 2023"].set_index("output_feature")
cols = pd.read_parquet(CACHE / "brfss_2023_stroke_rich_clean.parquet").drop(columns="stroke")
rows = []
for c in cols.columns:
    tier = "A" if c in TIER_A else "B" if c in TIER_B else "C" if c in TIER_C else "Pre-event"
    nlev = cols[c].nunique()
    kind = ("one-hot" if not pd.api.types.is_numeric_dtype(cols[c].dtype)
            else "binary" if nlev == 2 else "coded" if nlev <= 12 else "continuous")
    raw = inv.loc[c, "chosen_raw_variable"] if c in inv.index else "derived (BMI missing)"
    desc = inv.loc[c, "description"] if c in inv.index else "Indicator that BMI was missing."
    rows.append(f"{tex(c)} & {tex(raw)} & {tex(desc)} & {kind} & {tier} & {'no' if c in ONLY_2023 else 'yes'} \\\\")
out.append(r"""{\small\setlength{\tabcolsep}{4pt}\begin{longtable}{p{3.9cm}p{2.1cm}p{4.4cm}p{1.8cm}p{1.5cm}p{0.9cm}}
\caption{Predictors (BRFSS 2023). Set: Pre-event = retained in the 37-predictor model; A, B, C = post-event tiers removed. In 2024: available in the BRFSS 2024 core. Type: continuous, binary (0/1), coded (BRFSS integer code used as numeric) or one-hot.}\label{tab:s1}\\
\toprule Predictor & BRFSS variable & Description & Type & Set & In 2024 \\ \midrule \endfirsthead
\toprule Predictor & BRFSS variable & Description & Type & Set & In 2024 \\ \midrule \endhead
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}}\n")

# S2 hyperparameters
p = pd.read_csv(PROJ / "simillar reasearch test" / "rich_feature_best_4way_final_search" / "outputs" / "tables"
                / "T04_optuna_best_params.csv").set_index("label").loc["MODE_A_R0"]
hp = [("max\\_depth", f"{int(p.max_depth)}"), ("learning\\_rate", f"{p.learning_rate:.4f}"),
      ("subsample", f"{p.subsample:.3f}"), ("colsample\\_bytree", f"{p.colsample_bytree:.3f}"),
      ("min\\_child\\_weight", f"{int(p.min_child_weight)}"), ("gamma", f"{p.gamma:.3f}"),
      ("reg\\_alpha", f"{p.reg_alpha:.3f}"), ("reg\\_lambda", f"{p.reg_lambda:.2e}"),
      ("scale\\_pos\\_weight", f"{p.scale_pos_weight:.3f}"),
      ("n\\_estimators", "up to 2,000 (early stopping, 50 rounds, calibration-set PR-AUC)"),
      ("tree\\_method", "hist")]
out.append(r"""\begin{table}[h]\centering
\caption{XGBoost hyperparameters (Optuna TPE, 100 trials, seed 42). Objective on the calibration set: $0.35\,\text{PR-AUC} + 0.30\,\max F_2 + 0.15\,\text{precision at 60\% sensitivity} + 0.10\,\text{ROC-AUC} + 0.10\,(1 - 10\,\text{Brier})$, where F$_2$ is the F-score weighting sensitivity twice as much as precision; TPE, tree-structured Parzen estimator. Search ranges: max\_depth 3--10; learning\_rate 0.005--0.3 (log); subsample 0.5--1; colsample\_bytree 0.3--1; min\_child\_weight 1--20; gamma 0--5; reg\_alpha and reg\_lambda $10^{-8}$--10 (log); scale\_pos\_weight 5--30.}\label{tab:s2}
\begin{tabular}{ll}\toprule Hyperparameter & Value \\ \midrule
""" + "\n".join(f"{k} & {v} \\\\" for k, v in hp) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

# S3 all performance rows
t = pd.read_csv(A / "table2_performance.csv")
name = {"xgb_full": "XGBoost, full (55)", "xgb_pre": "XGBoost, pre-event (37)", "lr_full": "LR, full (55)",
        "lr_pre": "LR, pre-event (37)", "xgb_full_2024feats": "XGBoost, full (52)",
        "xgb_pre_2024feats": "XGBoost, pre-event (35)"}
ev = {"test_2023": "2023 test", "test_2023_weighted": "2023 test, weighted", "temporal_2024": "2024",
      "temporal_2024_weighted": "2024, weighted"}
assert (t.calibration == "isotonic").all()  # stated in the caption instead of a column
rows_a, rows_b = [], []
for _, r in t.iterrows():
    rows_a.append(f"{name[r.model]} & {ev[r['eval']]} & {f(r.roc_auc)} & {f(r.pr_auc)} & {f(r.brier, 4)} & "
                  f"{f(r.ece, 4)} & {f(r.cal_slope, 2)} & {f(r.cal_intercept, 3)} \\\\")
    rows_b.append(f"{name[r.model]} & {ev[r['eval']]} & {f(r.threshold, 4)} & {f(r.sensitivity)} & {f(r.specificity)} & "
                  f"{f(r.ppv)} & {f(r.npv)} & {f(r.f2)} & {f(r.fp_per_tp, 2)} & {r.flagged_per_1000:.0f} \\\\")
# portrait: two stacked panels instead of one 17-column landscape table
out.append(r"""\begin{table}\centering\footnotesize\setlength{\tabcolsep}{2.5pt}
\caption{All performance estimates, unweighted and survey-weighted (BRFSS final weight \_LLCPWT applied to frozen predictions). The 2024 models use the predictors available in both years and were fitted on 2023 data only. Isotonic calibration was selected for every model. Int.: calibration-in-the-large (logit offset with slope fixed at 1). ECE, expected calibration error; Sens., sensitivity; Spec., specificity; PPV and NPV, positive and negative predictive value; F$_2$, F-score weighting sensitivity twice as much as precision; FP/TP, false positives per true positive. LR, logistic regression.}\label{tab:s3}
\begin{tabular}{llrrrrrr}\toprule
\multicolumn{8}{l}{\textit{(a) Discrimination and calibration}} \\
Model & Evaluation & ROC-AUC & PR-AUC & Brier & ECE & Slope & Int. \\ \midrule
""" + "\n".join(rows_a) + r"""
\bottomrule\end{tabular}

\bigskip
\begin{tabular}{llrrrrrrrr}\toprule
\multicolumn{10}{l}{\textit{(b) Performance at the primary threshold}} \\
Model & Evaluation & Threshold & Sens. & Spec. & PPV & NPV & F$_2$ & FP/TP & Flagged/1,000 \\ \midrule
""" + "\n".join(rows_b) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

# S4 Experiment 1 per seed
e1 = pd.read_csv(R / "exp1_per_seed.csv")
vn = {"full_55": "Full", "minus_A": "$-$A", "minus_AB": "$-$A,B", "minus_ABC": "$-$A--C"}
rows = [f"{r.seed} & {vn[r.variant]} & {r.n_features} & {f(r.roc_auc)} & {f(r.pr_auc)} & {f(r.brier, 4)} & "
        f"{f(r.ece, 4)} & {f(r.threshold, 4)} & {f(r.recall)} & {f(r.precision)} & {f(r.f2)} & {f(r.fp_per_tp, 2)} \\\\"
        for _, r in e1.iterrows()]
out.append(r"""\begin{table}[h]\centering\footnotesize
\caption{Effect of removing post-event predictors: results for each random four-way split (seed 42 is the primary split). ECE, expected calibration error; Sens., sensitivity; PPV, positive predictive value; F$_2$, F-score weighting sensitivity twice as much as precision; FP/TP, false positives per true positive.}\label{tab:s4}
\begin{tabular}{rlrrrrrrrrrr}\toprule
Seed & Set & $n$ & ROC-AUC & PR-AUC & Brier & ECE & Threshold & Sens. & PPV & F$_2$ & FP/TP \\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

# S5 Experiment 2 full
blocks = []
for ds, label in (("brfss2023", "BRFSS 2023 (5 splits)"), ("kaggle", "Kaggle (20 splits)")):
    s = pd.read_csv(R / f"exp2_{ds}_summary.csv")
    blocks.append(f"\\midrule\\multicolumn{{11}}{{l}}{{\\textit{{{label}}}}} \\\\")
    for _, r in s.iterrows():
        blocks.append(f"{tex(r.pipeline)} & {r.kind} & {f(r.prevalence)} & {f(r.accuracy)} & {f(r.recall)} & "
                      f"{f(r.specificity)} & {f(r.precision)} & {f(r.f1)} & {f(r.roc_auc)} & {f(r.pr_auc)} & {f(r.brier, 4)} \\\\")
out.append(r"""\begin{table}[h]\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{Evaluation-optimism audit, all pipelines and metrics (means over splits). Prevalence is that of the split on which the metric was computed; after SMOTE before splitting, the pipeline's own test split is balanced (0.50). Prev., prevalence; Sens., sensitivity; Spec., specificity; PPV, positive predictive value; F$_1$, harmonic mean of precision and sensitivity.}\label{tab:s5}
\begin{tabular}{p{4.6cm}lrrrrrrrrr}\toprule
Pipeline & Kind & Prev. & Accuracy & Sens. & Spec. & PPV & F$_1$ & ROC-AUC & PR-AUC & Brier \\
""" + "\n".join(blocks) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

DST.write_text("\n".join(out), encoding="utf-8")
print("wrote", DST)

# S7 subgroup performance and S8 residual post-event signal (run_subgroup_and_residual.py)
extra = []
rc = pd.read_csv(A / "residual_signal_check.csv")
vn = {"pre_event_37": "Pre-event set", "minus_insurance": "$-$ health insurance",
      "minus_insurance_income_home": "$-$ insurance, income, home ownership"}
rows = []
for _, r in rc.iterrows():
    d = "--" if pd.isna(r.get("d_roc_auc")) else f"{r.d_roc_auc:+.4f} ({r.d_roc_lo:+.4f} to {r.d_roc_hi:+.4f})"
    rows.append(f"{vn[r.variant]} & {r.n_features} & {f(r.roc_auc, 4)} & {f(r.pr_auc, 4)} & {f(r.ece, 4)} & {f(r.sensitivity)} & {f(r.ppv)} & {d} \\\\")
extra.append(r"""\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}
\caption{Residual post-event signal in the pre-event set: the XGBoost pipeline refitted without predictors that may change after a stroke (2023 test set, primary split). $\Delta$ROC-AUC is the paired-bootstrap change versus the pre-event set (1,000 resamples). ECE, expected calibration error; Sens., sensitivity; PPV, positive predictive value.}\label{tab:s8}
\begin{tabular}{lrrrrrrl}\toprule
Predictor set & $n$ & ROC-AUC & PR-AUC & ECE & Sens. & PPV & $\Delta$ROC-AUC (95\% CI) \\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
(DST.parent / "supp_tables_extra.tex").write_text("\n".join(extra), encoding="utf-8")
print("wrote supp_tables_extra.tex")
