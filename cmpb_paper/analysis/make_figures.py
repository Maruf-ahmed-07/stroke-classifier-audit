"""Data figures, one vector file each. Figure 1 is drawn in LaTeX
(manuscript_cmpb/figures/Figure_1_source.tex).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "analysis" / "outputs"
R = ROOT / "robustness_experiments" / "outputs"
FIG = ROOT / "manuscript_cmpb" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# colours; line styles differ too, so the plots also work in grey
C_FULL, C_PRE, C_LR = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 600,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
LAB = {"xgb_full": "XGBoost, full (55)", "xgb_pre": "XGBoost, pre-event (37)",
       "lr_pre": "Logistic regression, pre-event (37)"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


# Figure 2: flexible calibration curves (run_v2_analyses.py)
cc = pd.read_csv(A / "v2_flexible_calibration_curves.csv")
fig, ax = plt.subplots(figsize=(4.4, 3.9))
lim = 0.40
for label, c, ls in (("XGBoost, 2023 test", C_PRE, "-"), ("Logistic regression, 2023 test", C_LR, (0, (1, 1))),
                     ("XGBoost, BRFSS 2024", C_FULL, "--")):
    g = cc[(cc.model == label) & (cc.predicted <= lim)]
    ax.fill_between(g.predicted, g.lo, g.hi, color=c, alpha=0.15, lw=0)
    ax.plot(g.predicted, g.observed, color=c, ls=ls, lw=1.6, label=label.replace("XGBoost", "XGBoost pre-event"))
ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls=":", label="Perfect calibration")
ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Predicted probability", ylabel="Observed stroke proportion (flexible)")
ax.legend(loc="upper left", fontsize=7.5)
save(fig, "Figure_2")

# Figure 3: decision curves
dca = pd.read_csv(A / "dca.csv")
thr = pd.read_csv(A / "primary_thresholds.csv", index_col=0)["primary_threshold"]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.axvline(thr["xgb_pre"], color=GRID, lw=4, zorder=0)
ax.plot(dca.pt, dca.treat_all, color=INK2, lw=1.0, ls="-.", label="Flag everyone")
ax.plot(dca.pt, dca.treat_none, color=INK2, lw=1.0, ls=":", label="Flag no one")
for key, c, ls in (("xgb_full", C_FULL, "-"), ("xgb_pre", C_PRE, "--"), ("lr_pre", C_LR, (0, (1, 1)))):
    ax.plot(dca.pt, dca[key], color=c, lw=1.5, ls=ls, label=LAB[key])
pb = pd.read_csv(A / "v2_paired_bootstrap.csv").set_index("quantity")
for who, c, dx in (("xgb", C_PRE, -0.0015), ("lr", C_LR, 0.0015)):
    for pt in (0.05, 0.08, 0.10, 0.15):
        r = pb.loc[f"nb_{who}_{pt}"]
        ax.errorbar(pt + dx, r.estimate, yerr=[[r.estimate - r.lo], [r.hi - r.estimate]], color=c, lw=1, capsize=2,
                    marker="o", ms=2.5, zorder=4)
ax.set(xlim=(0.01, 0.30), ylim=(-0.005, 0.036), xlabel="Threshold probability", ylabel="Net benefit")
ax.legend(loc="upper right", fontsize=8)
save(fig, "Figure_3")

# Figure 4: evaluation-optimism audit
ORDER = [("C1: t=0.5, no calibration", "Default model,\nthreshold 0.5"),
         ("C2: C1 + threshold tuned on test", "Threshold tuned\non test split"),
         ("C3: SMOTE before split, t=0.5", "SMOTE before\nsplitting"),
         ("C5 (control): SMOTE on train only, t=0.5", "SMOTE on training\nsplit only"),
         ("R: paper pipeline", "Proposed\npipeline")]
C_REP, C_ACT = "#eb6834", "#2a78d6"
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
for ax, (ds, label) in zip(axes, (("brfss2023", "(a) BRFSS 2023"), ("kaggle", "(b) Kaggle stroke dataset"))):
    s = pd.read_csv(R / f"exp2_{ds}_summary.csv").set_index(["pipeline", "kind"])["recall"]
    for i, (key, lab) in enumerate(ORDER):
        act = s[(key, "actual")]
        is_ref = key.startswith("R")
        dy = 0.0 if is_ref else 0.13
        if is_ref:
            ax.annotate("reported = actual", (act, i), xytext=(8, -3), textcoords="offset points",
                        fontsize=7.5, color=INK2)
        else:
            rep = s[(key, "reported")]
            ax.plot([act, rep], [i + dy, i - dy], color=GRID, lw=3, zorder=1, solid_capstyle="round")
            ax.scatter(rep, i - dy, s=40, color=C_REP, marker="D", zorder=3, edgecolor="white", linewidth=1,
                       label="Reported (own test split)" if i == 0 else None)
        ax.scatter(act, i + dy, s=40, color=C_ACT, zorder=3, edgecolor="white", linewidth=1,
                   label="Actual (untouched holdout)" if i == 0 else None)
    ax.text(0.0, 1.02, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    ax.set(xlim=(-0.02, 1.02), xlabel="Sensitivity")
    ax.set_yticks(range(len(ORDER)), [lab for _, lab in ORDER])
    ax.grid(axis="y", visible=False)
axes[0].invert_yaxis()
axes[0].legend(loc="upper right", fontsize=7.5)
save(fig, "Figure_4")

# Figure 5 and S1: SHAP importance
NAMES = {
    "age": "Age", "hypertension_status": "Hypertension", "computed_michd": "Prior MI or CHD",
    "health_insurance": "Health insurance", "arthritis": "Arthritis", "diabetes_status": "Diabetes",
    "income": "Income", "education": "Education", "cholesterol_status": "High cholesterol",
    "heart_attack": "Heart attack", "asthma_ever": "Asthma (ever)", "kidney_disease": "Kidney disease",
    "copd": "COPD", "home_ownership": "Home ownership", "smoked_100": "Smoked ≥100 cigarettes",
    "cholesterol_meds": "Cholesterol medication", "general_health": "General health",
    "walking_difficulty": "Walking difficulty", "cognitive_difficulty": "Cognitive difficulty",
    "bmi": "BMI", "employment": "Employment", "flu_vaccine": "Flu vaccine", "sex": "Sex",
    "marital": "Marital status", "race_ethnicity": "Race/ethnicity", "smoking_status": "Smoking status",
    "independent_living_difficulty": "Independent-living difficulty",
    "physical_health_days": "Poor physical-health days", "coronary_heart_disease": "Coronary heart disease",
    "depression": "Depression", "blind": "Blindness", "pneumonia_vaccine": "Pneumonia vaccine",
    "recent_checkup": "Recent check-up", "alcohol_days_code": "Drinking days per month",
    "average_drinks": "Drinks per drinking day", "max_drinks": "Maximum drinks", "children_count": "Children in household",
    "current_smoking_detail": "Current smoking frequency", "ecigarette_use": "E-cigarette use",
    "smokeless_tobacco": "Smokeless tobacco", "bmi_category": "BMI category", "skin_cancer": "Skin cancer",
    "other_cancer": "Other cancer", "asthma_current": "Asthma (current)", "binge_drinking": "Binge drinking",
    "heavy_drinking": "Heavy drinking", "alcohol_any": "Any alcohol", "medical_cost_barrier": "Cost barrier to care",
}
for key, name, color in (("xgb_pre", "Figure_5", C_PRE), ("xgb_full", "Figure_S1", C_FULL)):
    s = pd.read_csv(A / f"shap_{key}.csv", index_col=0)["mean_abs_shap"].head(15)[::-1]
    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    ax.barh([NAMES.get(n, n.replace("_", " ")) for n in s.index], s.values, color=color, height=0.62)
    ax.set(xlabel="Mean |SHAP value| (log-odds)")
    ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    save(fig, name)

# Figure S2: subgroup forest plot (run_v2_analyses.py)
sg = pd.read_csv(A / "v2_subgroups.csv")
sg = sg[(sg.group != "Missing") & (sg.cases >= 20)]
rows, ylab, y = [], [], 0
for dim in ["All", "Sex", "Age", "Race and ethnicity", "Household income", "Education"]:
    g = sg[sg.dimension == dim]
    if dim != "All":
        ylab.append((y, dim, True)); y += 1
    for _, r in g.iterrows():
        rows.append((y, r)); ylab.append((y, r.group if dim != "All" else "All respondents", False)); y += 1
fig, ax = plt.subplots(figsize=(5.4, 0.22 * y + 0.8))
overall = sg[sg.dimension == "All"].roc_auc.iloc[0]
ax.axvline(overall, color=GRID, lw=3, zorder=0)
for yy, r in rows:
    ax.errorbar(r.roc_auc, yy, xerr=[[r.roc_auc - r.roc_auc_lo], [r.roc_auc_hi - r.roc_auc]], color=C_PRE,
                marker="s", ms=3.5, lw=1.1, capsize=0)
    ax.text(1.005, yy, f"{int(r.cases):,} / {int(r.n):,}", transform=ax.get_yaxis_transform(), va="center",
            fontsize=7, color=INK2)
ax.set_yticks([t for t, _, _ in ylab], [l for _, l, _ in ylab], fontsize=7.5)
for lab in ax.get_yticklabels():
    if any(lab.get_text() == l and h for _, l, h in ylab):
        lab.set_fontweight("bold")
ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set(xlabel="ROC-AUC (95% CI)", xlim=(0.55, 0.95))
ax.text(1.005, -0.9, "cases / n", transform=ax.get_yaxis_transform(), fontsize=7, color=INK2)
save(fig, "Figure_S2")

# Figure S3: generalisation of the synthetic-share check (run_identity_generalisation.py)
gr = pd.read_csv(R / "identity_generalisation.csv")
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot([0.45, 1.0], [0.45, 1.0], color=INK2, lw=0.8, ls=":", label="Reported = $s$")
for (name, c, mk) in (("SMOTE", C_PRE, "o"), ("Borderline-SMOTE", C_LR, "s"), ("ADASYN", C_FULL, "^"),
                      ("Random oversampling", INK2, "x")):
    g = gr[gr.oversampler == name]
    ax.scatter(g.s_counts, g.reported_sensitivity, s=18, color=c, marker=mk, label=name, alpha=0.85, lw=0.8)
ax.set(xlim=(0.45, 1.01), ylim=(0.45, 1.01), xlabel="Synthetic share of positives, $s$",
       ylabel="Reported sensitivity (own test split)")
ax.legend(loc="upper left", fontsize=7.5)
save(fig, "Figure_S3")

print("figures written to", FIG)
