"""Supplementary tables from the outputs of run_v2_analyses.py, the NHIS script and the
robustness scripts.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "analysis" / "outputs"
R = ROOT / "robustness_experiments" / "outputs"
DST = ROOT / "manuscript_cmpb" / "supp_tables_v2.tex"
DST.parent.mkdir(parents=True, exist_ok=True)


def tex(x):
    return str(x).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("$", r"\$")


def f(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


v2 = []
# flexible calibration
fc = pd.read_csv(A / "v2_flexible_calibration.csv")
rows = [f"{r.model} & {r.measure.upper()} & {f(r.estimate, 4)} & {f(r.lo, 4)}--{f(r.hi, 4)} \\\\" for _, r in fc.iterrows()]
v2.append(r"""\begin{table}[h]\centering\footnotesize
\caption{Flexible calibration: logistic regression of the outcome on a cubic spline of the logit of predicted risk (knots at the 5th, 27.5th, 50th, 72.5th and 95th percentiles). ICI, mean absolute difference between the flexible curve and the predicted risk; E50 and E90, its median and 90th percentile; Emax, its maximum. 95\% CIs from 200 bootstrap resamples. The 2024 model uses the 35 predictors common to both years and was fitted on 2023 data only. ICI, integrated calibration index.}\label{tab:flex}
\begin{tabular}{llrl}\toprule Model and data & Measure & Estimate & 95\% CI \\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

# paired bootstrap and net benefit
pb = pd.read_csv(A / "v2_paired_bootstrap.csv").set_index("quantity")


def pbrow(label, key, d=4):
    r = pb.loc[key]
    return f"{label} & {f(r.estimate, d)} & {f(r.lo, d)} to {f(r.hi, d)} \\\\"


rows = [pbrow("ROC-AUC, XGBoost", "auc_xgb"), pbrow("ROC-AUC, logistic regression", "auc_lr"),
        pbrow(r"$\Delta$ROC-AUC (XGBoost $-$ LR)", "delta_auc_xgb_minus_lr"),
        pbrow("PR-AUC, XGBoost", "ap_xgb"), pbrow("PR-AUC, logistic regression", "ap_lr"),
        pbrow(r"$\Delta$PR-AUC (XGBoost $-$ LR)", "delta_ap_xgb_minus_lr"),
        pbrow("Brier score, XGBoost", "brier_xgb", 5), pbrow("Brier score, logistic regression", "brier_lr", 5),
        pbrow(r"$\Delta$Brier (XGBoost $-$ LR)", "delta_brier_xgb_minus_lr", 5), r"\midrule"]
for pt in (0.05, 0.08, 0.1, 0.15):
    rows += [pbrow(f"Net benefit at {pt:.2f}: XGBoost", f"nb_xgb_{pt}"),
             pbrow(r"\quad logistic regression", f"nb_lr_{pt}"),
             pbrow(r"\quad flag everyone", f"nb_all_{pt}"),
             pbrow(r"\quad $\Delta$ XGBoost $-$ LR", f"delta_nb_xgb_minus_lr_{pt}"),
             pbrow(r"\quad $\Delta$ XGBoost $-$ flag everyone", f"delta_nb_xgb_minus_all_{pt}")]
sb = pd.read_csv(A / "simple_baseline.csv").set_index("quantity")


def sbrow(label, key, d=4):
    r = sb.loc[key]
    return f"{label} & {f(r.estimate, d)} & {f(r.lo, d)} to {f(r.hi, d)} \\\\"


rows += [r"\midrule \multicolumn{3}{l}{\textit{Pre-event XGBoost versus a six-predictor logistic regression}} \\",
         sbrow("ROC-AUC, six-predictor model", "auc_simple"),
         sbrow(r"$\Delta$ROC-AUC (XGBoost $-$ six-predictor)", "delta_auc_xgb_minus_simple"),
         sbrow("PR-AUC, six-predictor model", "ap_simple"),
         sbrow(r"$\Delta$PR-AUC (XGBoost $-$ six-predictor)", "delta_ap_xgb_minus_simple"),
         sbrow(r"$\Delta$Brier (XGBoost $-$ six-predictor)", "delta_brier_xgb_minus_simple", 5),
         sbrow(r"$\Delta$ net benefit at 0.05", "delta_nb05_xgb_minus_simple"),
         sbrow(r"$\Delta$ net benefit at 0.08", "delta_nb08_xgb_minus_simple")]
v2.append(r"""\begin{longtable}{lrl}
\caption{Pre-event XGBoost versus pre-event logistic regression (LR) on the 2023 test set, with net benefit at selected threshold probabilities, and versus a six-predictor logistic regression of classic risk factors (age, sex, hypertension, diabetes, prior myocardial infarction or coronary heart disease, smoking status) fitted with the same pipeline. Differences use the same 1,000 bootstrap resamples for both models (paired bootstrap); 95\% percentile intervals.}\label{tab:paired}\\
\toprule Quantity & Estimate & 95\% CI \\ \midrule \endfirsthead
\toprule Quantity & Estimate & 95\% CI \\ \midrule \endhead
""" + "\n".join(rows) + "\n\\bottomrule\\end{longtable}\n")

# subgroups
sg = pd.read_csv(A / "v2_subgroups.csv")
rows, last = [], None
for _, r in sg.iterrows():
    if r.dimension != last:
        rows.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{tex(r.dimension)}}}}} \\\\"); last = r.dimension
    auc = "--" if pd.isna(r.roc_auc) else f"{f(r.roc_auc)} ({f(r.roc_auc_lo)}--{f(r.roc_auc_hi)})"
    rows.append(f"\\quad {tex(r.group)} & {r.n:,} & {r.cases:,} & {f(r.prevalence)} & {auc} & "
                f"{f(r.observed_over_expected, 2)} & {f(r.sensitivity)} & {f(r.specificity)} & {f(r.ppv)} & "
                f"{f(r.flagged_share)} \\\\")
v2.append(r"""{\scriptsize\setlength{\tabcolsep}{3pt}\begin{longtable}{lrrrlrrrrr}
\caption{Pre-event XGBoost model by subgroup on the 2023 test set, at the fixed primary threshold. Race and ethnicity from the BRFSS imputed variable \_IMPRACE; household income from INCOME3; education from EDUCA. O/E, observed-to-expected ratio (above 1: risk under-estimated). Flagged, share of the subgroup above the threshold. ROC-AUC 95\% CIs from 500 bootstrap resamples. Subgroup analyses are descriptive. AI/AN, American Indian or Alaska Native. Prev., prevalence; Sens., sensitivity; Spec., specificity; PPV, positive predictive value.}\label{tab:sub}\\
\toprule Group & $n$ & Cases & Prev. & ROC-AUC (95\% CI) & O/E & Sens. & Spec. & PPV & Flagged \\ \midrule \endfirsthead
\toprule Group & $n$ & Cases & Prev. & ROC-AUC (95\% CI) & O/E & Sens. & Spec. & PPV & Flagged \\ \midrule \endhead
""" + "\n".join(rows) + "\n\\bottomrule\\end{longtable}}\n")

# synthetic-share identity
ide = pd.read_csv(R / "identity_summary.csv")
DS = {"brfss2023": "BRFSS 2023 (3 splits)", "kaggle": "Kaggle (20 splits)"}
rows, last = [], None
for _, r in ide.sort_values(["dataset", "variant", "classifier"]).iterrows():
    if r.dataset != last:
        rows.append(f"\\midrule\\multicolumn{{9}}{{l}}{{\\textit{{{DS[r.dataset]}}}}} \\\\"); last = r.dataset
    rows.append(f"{r.variant} & {r.classifier} & {f(r.synthetic_share_s)} & {f(r.reported_sensitivity)} & "
                f"{f(r.sensitivity_synthetic)} & {f(r.sensitivity_real_internal)} & {f(r.actual_sensitivity)} & "
                f"{f(r.reported_roc_auc)} & {f(r.actual_roc_auc)} \\\\")
v2.append(r"""\begin{table}[h]\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{Decomposition of the sensitivity reported after SMOTE before splitting (threshold 0.5), by classifier. $s$, share of synthetic records among the positives of the pipeline's own test split. Reported sensitivity equals $s\,r_{\mathrm{syn}} + (1-s)\,r_{\mathrm{real}}$, where $r_{\mathrm{syn}}$ and $r_{\mathrm{real}}$ are the sensitivities on the synthetic and on the real positives of that split. ``Actual'' is the same model on the untouched holdout. In the rounding variant, every synthetic value of an integer-valued column (including one-hot columns) was rounded to the nearest integer in original units, so synthetic records carry no fractional values. Means over splits.}\label{tab:identity}
\begin{tabular}{llrrrrrrr}\toprule
Oversampling & Classifier & $s$ & \shortstack{Reported\\sens.} & $r_{\mathrm{syn}}$ & $r_{\mathrm{real}}$ & \shortstack{Actual\\sens.} & \shortstack{Reported\\ROC-AUC} & \shortstack{Actual\\ROC-AUC} \\
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

nn = pd.read_csv(R / "identity_nearest_neighbour.csv")
cols = ["synthetic positives, internal test", "real positives, internal test", "positives, audit holdout",
        "negatives, audit holdout"]
rows = [f"{ {'brfss2023': 'BRFSS 2023', 'kaggle': 'Kaggle'}[r.dataset]} & " + " & ".join(f(r[c]) for c in cols) + r" \\"
        for _, r in nn.iterrows()]
v2.append(r"""\begin{table}[h]\centering\footnotesize
\caption{Median Euclidean distance (standardised and one-hot encoded predictors) from a record to its nearest positive in the training part of a pipeline that applied SMOTE before splitting (first split; up to 2,000 sampled records per group).}\label{tab:nn}
\begin{tabular}{lrrrr}\toprule
 & \multicolumn{2}{c}{Pipeline's own test split} & \multicolumn{2}{c}{Untouched holdout} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Data & Synthetic positives & Real positives & Positives & Negatives \\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

gen = pd.read_csv(R / "identity_generalisation_summary.csv")
gen["a_est"] = (gen.reported_sensitivity - gen.s_counts) / (1 - gen.s_counts)
rows, last = [], None
for _, r in gen.iterrows():
    if r.oversampler != last:
        rows.append(f"\\midrule\\multicolumn{{9}}{{l}}{{\\textit{{{r.oversampler}}}}} \\\\"); last = r.oversampler
    rows.append(f"{100 * r.prevalence:.0f}\\% & {r.alpha:.1f} & {f(r.s_counts)} & {f(r.reported_sensitivity)} & "
                f"{f(r.reported_precision)} & {f(r.r_syn)} & {f(r.r_real)} & {f(r.a_est)} & {f(r.actual_sensitivity)} \\\\")
v2.append(r"""{\footnotesize\setlength{\tabcolsep}{4pt}\begin{longtable}{rrrrrrrrr}
\caption{Generalisation of the synthetic-share check. Samples of 40,000 BRFSS 2023 respondents with prevalence $\pi$ were oversampled before splitting to a minority/majority ratio $\alpha$; a default XGBoost was evaluated at threshold 0.5 on the pipeline's own test split and on the untouched holdout (``actual''). $s$, synthetic share of positives, $1-\pi/(\alpha(1-\pi))$; $r_{\mathrm{syn}}$, $r_{\mathrm{real}}$, sensitivity on synthetic and real positives of the pipeline's test split; $\hat a = (\text{reported} - s)/(1-s)$, the sensitivity implied by the check. Means of three repetitions. For random oversampling the ``synthetic'' records are duplicates. PPV, positive predictive value.}\label{tab:gen}\\
\toprule $\pi$ & $\alpha$ & $s$ & Reported sens. & Reported PPV & $r_{\mathrm{syn}}$ & $r_{\mathrm{real}}$ & $\hat a$ & Actual sens. \\ \endfirsthead
\toprule $\pi$ & $\alpha$ & $s$ & Reported sens. & Reported PPV & $r_{\mathrm{syn}}$ & $r_{\mathrm{real}}$ & $\hat a$ & Actual sens. \\ \endhead
""" + "\n".join(rows) + "\n\\bottomrule\\end{longtable}}\n")

# NHIS
hz = pd.read_csv(A / "nhis_harmonisation.csv")
MAP = {"age": (r"\_AGE80", r"AGE\_P (capped at 80)"), "sex": ("SEXVAR", "SEX"),
       "race_ethnicity": (r"\_IMPRACE", r"HISPAN\_I, RACERPI2"), "education": ("EDUCA", "EDUC1 (recoded)"),
       "marital": ("MARITAL", r"R\_MARITL (recoded)"), "uninsured": ("PRIMINS1 = 88", "NOTCOV"),
       "hypertension": ("BPHIGH6 = yes", "HYPEV"), "high_cholesterol": ("TOLDHI3", "CHLEV"),
       "diabetes": ("DIABETE4 = yes", "DIBEV/DIBEV1 = yes"), "heart_attack": ("CVDINFR4", "MIEV"),
       "coronary_heart_disease": ("CVDCRHD4", "CHDEV"), "asthma_ever": ("ASTHMA3", "AASMEV"),
       "cancer_any": ("CHCSCNC1 or CHCOCNC1", "CANEV"), "copd": ("CHCCOPD3", "COPDEV"),
       "kidney_disease": ("CHCKDNY2 (ever)", "KIDWKYR (past year)"), "arthritis": ("HAVARTH4", "ARTH1"),
       "bmi": (r"\_BMI5", "BMI"), "smoking_status": (r"\_SMOKER3", "SMKSTAT2")}
rows = [f"{tex(r.feature)} & {MAP[r.feature][0]} & {MAP[r.feature][1]} & {100 * r.brfss_missing:.1f} & "
        f"{100 * r.nhis_missing:.1f} \\\\" for _, r in hz.iterrows()]
v2.append(r"""\begin{table}[h]\centering\footnotesize
\caption{Harmonised predictors for the NHIS external validation, with the share of missing values (\%) in BRFSS 2023 and in NHIS (sample adults, 2012 and 2014--2018). All items were recoded to the BRFSS coding; binary items are 1 = yes, 0 = no.}\label{tab:nhisharm}
\setlength{\tabcolsep}{4pt}\begin{tabular}{lllrr}\toprule Predictor & BRFSS 2023 & NHIS & \shortstack{Missing\\BRFSS (\%)} & \shortstack{Missing\\NHIS (\%)} \\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

nh = pd.read_csv(A / "nhis_external_validation.csv")
SHORT = {"BRFSS 2023 test (internal)": "BRFSS 2023 test (internal)",
         "NHIS prevalent stroke (external)": "NHIS prevalent stroke",
         "NHIS cerebrovascular death, no baseline stroke (external)": "NHIS cerebrovascular death, all cohorts"}
rows, current = [], None
for _, r in nh.iterrows():
    if r.model != current:  # portrait layout: model as a group heading instead of a column
        current = r.model
        rows.append(("\\midrule\n" if rows else "") + f"\\multicolumn{{10}}{{l}}{{\\textit{{{tex(r.model)}}}}} \\\\")
    label = SHORT.get(r.evaluation) or "\\quad " + r.evaluation.replace("NHIS cerebrovascular death, ", "")
    auc = f(r.roc_auc) + ("" if pd.isna(r.get("roc_auc_lo")) else f" ({f(r.roc_auc_lo)}--{f(r.roc_auc_hi)})")
    n = "--" if pd.isna(r.n) else f"{int(r.n):,}"
    ev = "--" if pd.isna(r.events) else f"{int(r.events):,}"
    rows.append(f"{label} & {n} & {ev} & {auc} & {f(r.pr_auc)} & {f(r.observed_over_expected, 2)} & "
                f"{f(r.cal_slope, 2)} & {f(r.sensitivity)} & {f(r.specificity)} & {f(r.ppv)} \\\\")
v2.append(r"""\begin{table}\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{External validation of the harmonised 18-predictor model (fitted on BRFSS 2023 with the paper's pipeline and applied unchanged) in NHIS sample adults, 2012 and 2014--2018. Cerebrovascular death: underlying cause of death UCOD\_LEADING = 005 by 31 December 2019, among mortality-eligible adults without self-reported stroke at interview; the cohort rows split it by survey year. Sensitivity, specificity and PPV at the primary threshold fixed on BRFSS data. ``Age only'': ROC-AUC of age alone, for context. ROC-AUC 95\% CIs from 500 bootstrap resamples; O/E, observed-to-expected ratio; Sens., sensitivity; Spec., specificity; PPV, positive predictive value.}\label{tab:nhis}
\begin{tabular}{lrrlrrrrrr}\toprule
Evaluation & $n$ & Events & ROC-AUC (95\% CI) & PR-AUC & O/E & Slope & Sens. & Spec. & PPV \\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

# design-based intervals, re-tuning, generalisation to RF and Kaggle
dc = pd.read_csv(A / "design_ci.csv")
MEAS = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier score", "observed_over_expected": "O/E",
        "sensitivity": "Sensitivity", "specificity": "Specificity", "ppv": "PPV"}
rows, last = [], None
for _, r in dc.iterrows():
    if r.model != last:
        rows.append(f"\\midrule\\multicolumn{{3}}{{l}}{{\\textit{{{r.model}}}}} \\\\"); last = r.model
    rows.append(f"\\quad {MEAS[r.measure]} & {f(r.weighted_estimate, 4)} & {f(r.design_lo, 4)}--{f(r.design_hi, 4)} \\\\")
v2.append(r"""\begin{table}[h]\centering\footnotesize
\caption{Survey-weighted estimates (BRFSS final weight \_LLCPWT applied to frozen predictions) with design-based 95\% confidence intervals: primary sampling units (\_PSU) resampled with replacement within strata (\_STSTR); 500 replicates for the 2023 test set (test respondents only) and 200 for BRFSS 2024. O/E, observed-to-expected ratio; values above 1 mean that risk is under-estimated in the weighted population. Metrics at each model's primary threshold. PPV, positive predictive value.}\label{tab:design}
\begin{tabular}{lrl}\toprule Measure & Weighted estimate & Design-based 95\% CI \\
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

rt = pd.read_csv(A / "retune_comparison.csv")
rp = pd.read_csv(A / "retune_params.csv", index_col=0)
prow = [f"{tex(k)} & {rp.loc[k, 'original_T04']:.4g} & {rp.loc[k, 'retuned_corrected_data']:.4g} \\\\" for k in rp.index]
mrow = [f"{ {'xgb_pre': 'XGBoost, pre-event (37)', 'xgb_full': 'XGBoost, full (55)'}[r.model]} & {f(r.roc_auc_original, 4)} & "
        f"{f(r.roc_auc_retuned, 4)} & {r.delta_roc_auc:+.4f} ({r.delta_roc_lo:+.4f} to {r.delta_roc_hi:+.4f}) & "
        f"{f(r.pr_auc_original, 4)} & {f(r.pr_auc_retuned, 4)} & {f(r.sensitivity_retuned)} & {f(r.ppv_retuned)} \\\\"
        for _, r in rt.iterrows()]
v2.append(r"""\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Re-tuning check. The original Optuna search (TPE, seed 42, 100 trials, same ranges and composite objective on the calibration set; Table~\ref{tab:s2}) was repeated on the corrected data, and the models were refitted with the new hyperparameters through the same pipeline. $\Delta$ROC-AUC is retuned minus original on the same 2023 test set (paired bootstrap, 1,000 resamples). Sensitivity and PPV of the retuned models are at their own primary thresholds.}\label{tab:retune}
\begin{tabular}{lrr}\toprule Hyperparameter & Original & Re-tuned \\ \midrule
""" + "\n".join(prow) + r"""
\bottomrule\end{tabular}\\[6pt]
\resizebox{\textwidth}{!}{\begin{tabular}{lrrlrrrr}\toprule
Model & ROC-AUC orig. & ROC-AUC re-tuned & $\Delta$ROC-AUC (95\% CI) & PR-AUC orig. & PR-AUC re-tuned & Sens. & PPV \\ \midrule
""" + "\n".join(mrow) + "\n\\bottomrule\\end{tabular}}\\end{table}\n")

ge = pd.read_csv(R / "identity_generalisation_ext.csv")
ge0 = pd.read_csv(R / "identity_generalisation.csv").assign(dataset="BRFSS 2023", classifier="XGBoost")
ge0["abs_error"] = (ge0.reported_sensitivity - ge0.s_counts).abs()
ge0["a_est"] = (ge0.reported_sensitivity - ge0.s_counts) / (1 - ge0.s_counts)
allg = pd.concat([ge0, ge], ignore_index=True)
rows = []
for (ds, cl), g in allg.groupby(["dataset", "classifier"], sort=False):
    sm = g[g.oversampler != "Random oversampling"]; ro = g[g.oversampler == "Random oversampling"]
    lo, hi = sm[sm.prevalence <= 0.05], sm[sm.prevalence >= 0.10]
    rows.append(f"{ds} & {cl} & {len(lo)} & {f(lo.abs_error.mean())} & {f(lo.abs_error.max())} & {f(lo.reported_precision.min())} & "
                f"{f((hi.a_est - hi.actual_sensitivity).abs().mean())} & {f(ro[ro.prevalence <= 0.05].reported_sensitivity.mean())} \\\\")
v2.append(r"""\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{The synthetic-share check across classifiers and datasets (pipeline C3). BRFSS: samples of 40,000, prevalence 1--20\%, 3 repetitions; Kaggle: samples of 900 from the development pool, prevalence 2--20\%, 10 repetitions; oversampling ratios 0.5 and 1. SMOTE variants: SMOTE, Borderline-SMOTE and ADASYN. $|\text{reported}-s|$ at prevalence $\le$5\%; $|\hat a - \text{actual}|$, error of the implied sensitivity $\hat a=(\text{reported}-s)/(1-s)$ against the holdout sensitivity at prevalence $\ge$10\%; last column, mean reported sensitivity under random oversampling at prevalence $\le$5\%. The BRFSS XGBoost row summarises Table~\ref{tab:gen}. PPV, positive predictive value.}\label{tab:genext}
\begin{tabular}{llrrrrrr}\toprule
 & & \multicolumn{4}{c}{SMOTE variants, prevalence $\le$5\%} & SMOTE, $\ge$10\% & Random, $\le$5\% \\
\cmidrule(lr){3-6}\cmidrule(lr){7-7}\cmidrule(lr){8-8}
Data & Classifier & Runs & Mean $|\text{rep.}-s|$ & Max $|\text{rep.}-s|$ & Min PPV & Mean $|\hat a-\text{actual}|$ & Reported sens. \\ \midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

import re

for block in v2:  # one file per table, so the supplement controls the order (and hence the S-numbers)
    label = re.search(r"\\label\{tab:(\w+)\}", block).group(1)
    (DST.parent / f"supp_v2_{label}.tex").write_text(block, encoding="utf-8")
    print("wrote", f"supp_v2_{label}.tex")
