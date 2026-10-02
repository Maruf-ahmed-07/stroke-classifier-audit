"""Synthetic-share check applied to published open-access studies of the Kaggle stroke data.

Europe PMC search, 29-30 Sep 2026:
  ("stroke prediction" OR "stroke risk prediction") AND (SMOTE OR oversampling OR "over-sampling")
  AND ("5110" OR "5,110" OR kaggle) AND OPEN_ACCESS:y AND HAS_FT:y
plus open-access papers known from the literature review and two arXiv preprints.
Each value below is noted with where it was found in the paper.

Rules (fixed before extraction): a study is assessable if the order of oversampling and splitting
is known and the headline stroke sensitivity is reported. A: test set >= 30% stroke, or more
strokes than the whole dataset. B: headline sensitivity >= s - 0.03, s = 1 - 249/4861 = 0.949.
"""
from pathlib import Path

import pandas as pd

V = Path(__file__).resolve().parents[1]
S_KAGGLE = 1 - 249 / 4861
S_CEREBRAL = 1 - 783 / (43400 - 783)  # Kaggle "Cerebral Stroke Prediction" data, before deletion of incomplete rows

ROWS = [
    # oversampling before splitting
    dict(study="Tazin et al. 2021, J Healthc Eng", doi="10.1155/2021/7633381", group="oversampled before split",
         order="Text: SMOTE balances the dataset; 'after splitting, the model is trained' (80/20)",
         test_stroke=1413, test_n=2820, headline="Random forest", headline_sens=1341 / 1413,
         others="Recall 0.95 in classification report (Fig. 14)", s=S_KAGGLE,
         source="Fig. 13 confusion matrix (1,366/41/72/1,341)"),
    dict(study="Sahriar et al. 2024, Heliyon", doi="10.1016/j.heliyon.2024.e27411", group="oversampled before split",
         order="Figs. 4 and 6: SMOTE in preprocessing, then 'Balanced Data -> Train-Test Split' (70/30)",
         test_stroke=None, test_n=None, headline="Random forest", headline_sens=0.9612,
         others="", s=S_KAGGLE, source="Results text: recall 96.12%"),
    dict(study="Rehman et al. 2023, PeerJ Comput Sci", doi="10.7717/peerj-cs.1684", group="oversampled before split",
         order="Text: 'after balancing the stroke dataset, we divide the dataset into two sets: train and test'",
         test_stroke=1425, test_n=2917, headline="Stacking (also RF, DT, ET, GBM, KNN)", headline_sens=1.000,
         others="LR 0.836, SVM 0.623 (SMOTE)", s=S_KAGGLE,
         source="Tables 4 and 6 (stroke recall = TN/(TN+FN) in the paper's labelling)"),
    dict(study="Sheela Lavanya J M et al. 2024, Sci Rep (retracted)", doi="10.1038/s41598-024-70354-1", group="oversampled before split",
         order="Test set has 616 stroke cases, more than the 249 in the whole dataset",
         test_stroke=616, test_n=1106, headline="Random forest (also DT, KNN, XGB)", headline_sens=1.00,
         others="GB 0.94; LR 0.85; SVM 0.85", s=S_KAGGLE,
         source="Table 4 and confusion counts in text (paper labels non-stroke as positive)"),
    dict(study="Chakraborty et al. 2024, BMC Bioinformatics", doi="10.1186/s12859-024-05866-8", group="oversampled before split",
         order="Text: 'post-oversampling, both classes comprise 4861 cases each ... for training and testing'",
         test_stroke=945, test_n=1969, headline="Proposed stacking", headline_sens=942 / 945,
         others="RF 0.771, KNN 0.838 (base models)", s=S_KAGGLE, source="Table 3 (TP 942, FP 20, FN 3, TN 1,004)"),
    dict(study="Chakraborty et al. 2025, PLOS ONE", doi="10.1371/journal.pone.0328967", group="oversampled before split",
         order="Text: dataset oversampled to 4,861 per class before modelling",
         test_stroke=907, test_n=1968, headline="EnShap", headline_sens=846 / 907,
         others="RF 0.772 (base-model rows identical to Chakraborty et al. 2024)", s=S_KAGGLE,
         source="Table 6 (TP 846, FP 89, FN 61, TN 972)"),
    dict(study="Singh et al. 2024, Sci Rep", doi="10.1038/s41598-024-80129-3", group="oversampled before split",
         order="Text: random oversampling to 4,861 per class; 'after data preprocessing, the data has been split' (70/30)",
         test_stroke=None, test_n=None, headline="RF + DT voting (also RF, DT, XGB, KNN)", headline_sens=1.000,
         others="SVM 0.890, AdaBoost 0.904, SGD 0.874", s=S_KAGGLE, source="Table 3"),
    # resampling confined to training data, or none
    dict(study="Akinwumi et al. 2025, Front Neurol", doi="10.3389/fneur.2025.1668420", group="training-only or none",
         order="Text: random over-sampling 'applied to the training folds'",
         test_stroke=50, test_n=1022, headline="Gradient boosting / KNN (best)", headline_sens=1 / 50,
         others="LR, RF, SVM 0.00", s=S_KAGGLE, source="Fig. 6 caption (confusion matrices)"),
    dict(study="Sutcu et al. 2025, Stroke Res Treat", doi="10.1155/srat/2892726", group="training-only or none",
         order="Text: no rebalancing; stratified 70/30 split",
         test_stroke=None, test_n=None, headline="Naive Bayes (best)", headline_sens=0.404,
         others="Other models 0.011-0.213", s=S_KAGGLE, source="Table 7"),
    dict(study="Kokkotis et al. 2022, Diagnostics", doi="10.3390/diagnostics12102392", group="training-only or none",
         order="Text: random under-sampling within each training fold (nested 10-fold CV)",
         test_stroke=None, test_n=None, headline="MLP (best)", headline_sens=0.814,
         others="Different Kaggle dataset (43,400 records, 783 strokes)", s=S_CEREBRAL, source="Table 3"),
    dict(study="El-Geneedy et al. 2025, Sci Rep (retracted)", doi="10.1038/s41598-025-11263-9", group="training-only or none",
         order="Text: resampling 'only used in the training datasets in each cross-validation iteration'",
         test_stroke=None, test_n=None, headline="DNN (best test recall)", headline_sens=0.72,
         others="Other models 0.04-0.64 (training recall 0.92-0.98)", s=S_KAGGLE, source="Table 7"),
]

NOT_ASSESSABLE = [
    ("Wijaya et al. 2024, Bioengineering", "10.3390/bioengineering11070672", "SMOTE before 80/20 split stated, but only class-weighted recall reported"),
    ("Hassan et al. 2024, Sci Rep (retracted)", "10.1038/s41598-024-61665-4", "only accuracy and AUC reported"),
    ("Saleem et al. 2024, Sci Rep (retracted)", "10.1038/s41598-024-73570-x", "'sensitivity' refers to the non-stroke class; test counts not reported"),
    ("Dritsas et al. 2022, Sensors", "10.3390/s22134670", "oversampled before 10-fold CV, but only averaged recall reported"),
    ("Mochurad et al. 2025, BMC Med Inform Decis Mak (retracted)", "10.1186/s12911-025-02894-z", "order and stroke-class recall not determinable"),
    ("Srinivasu et al. 2024, Diagnostics", "10.3390/diagnostics14020128", "stroke-class sensitivity not reported"),
    ("Tashkova et al. 2025, arXiv:2505.09812", "10.48550/arXiv.2505.09812", "split order and metric type not stated"),
]
INELIGIBLE = [
    ("Singh et al. 2025, Sci Rep", "10.1038/s41598-025-30203-1", "different dataset, no oversampling"),
    ("arXiv:2512.01333 (2025)", "10.48550/arXiv.2512.01333", "Kaggle data mixed with synthetic symptom profiles; prevalence unknown"),
]


def main():
    df = pd.DataFrame(ROWS)
    df["test_stroke_share"] = df.test_stroke / df.test_n
    sig_a = ((df.test_stroke_share >= 0.30) | (df.test_stroke > 249)).astype("object")
    sig_a[df.test_stroke.isna()] = "not reported"
    df["signature_A_test_composition"] = sig_a
    df["signature_B_sens_ge_s_minus_0.03"] = df.headline_sens >= df.s - 0.03
    df.to_csv(V / "analysis" / "outputs" / "published_check.csv", index=False)
    for g, x in df.groupby("group"):
        print(f"{g}: n={len(x)}; signature B in {int(x['signature_B_sens_ge_s_minus_0.03'].sum())}; "
              f"headline sensitivity {x.headline_sens.min():.3f}-{x.headline_sens.max():.3f}; "
              f"test stroke share {x.test_stroke_share.min():.2f}-{x.test_stroke_share.max():.2f}")

    def tex(s):
        return (str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("->", r"$\rightarrow$")
                .replace("'", "'").replace(">=", r"$\ge$"))

    rows = []
    for g, label in (("oversampled before split", "Oversampling before splitting"), ("training-only or none", "Resampling of training data only, or none")):
        rows.append(f"\\midrule\\multicolumn{{6}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for _, r in df[df.group == g].iterrows():
            comp = "--" if pd.isna(r.test_n) else f"{int(r.test_stroke):,}/{int(r.test_n):,} ({100 * r.test_stroke_share:.0f}\\%)"
            rows.append(f"{tex(r.study)} & {tex(r.order)} & {comp} & {tex(r.headline)}: {r.headline_sens:.3f} & {r.s:.3f} & "
                        f"{'yes' if r['signature_B_sens_ge_s_minus_0.03'] else 'no'} \\\\")
            if r.others:
                rows.append(f" & \\multicolumn{{5}}{{p{{12.9cm}}}}{{\\footnotesize Other models: {tex(r.others)}. Source: {tex(r.source)}; doi:{r.doi}.}} \\\\")
            else:
                rows.append(f" & \\multicolumn{{5}}{{p{{12.9cm}}}}{{\\footnotesize Source: {tex(r.source)}; doi:{r.doi}.}} \\\\")
    na = "; ".join(f"{tex(a)} (doi:{b}): {tex(c)}" for a, b, c in NOT_ASSESSABLE)
    ie = "; ".join(f"{tex(a)} (doi:{b}): {tex(c)}" for a, b, c in INELIGIBLE)
    table = r"""{\scriptsize\setlength{\tabcolsep}{3pt}\begin{longtable}{p{2.8cm}p{5.4cm}p{1.9cm}p{3.1cm}p{0.7cm}p{0.8cm}}
\caption{The synthetic-share check applied to published open-access studies of the Kaggle stroke data. Order: how the order of oversampling and splitting was established. Test set: stroke cases/total in the reported test set (a natural split has about 5\% stroke). Headline: the stroke-class sensitivity of the study's headline model. $s$: synthetic share of positives after 1:1 oversampling. Check: headline sensitivity $\ge s-0.03$. Values were extracted from the papers' tables, figures and text (source given for each). The order was taken from the text or a pipeline figure, except for Sheela Lavanya J M et al., where it was inferred from the test-set counts. The two Chakraborty et al.\ studies share their first two authors and report identical base-model results, so they are not independent. Model abbreviations: CV, cross-validation; DNN, deep neural network; DT, decision tree; ET, extra trees; GB, gradient boosting; GBM, gradient boosting machine; KNN, k-nearest neighbours; LR, logistic regression; MLP, multilayer perceptron; RF, random forest; SGD, stochastic gradient descent; SVM, support vector machine; XGB, XGBoost. AUC, area under the receiver operating characteristic curve; TP, FP, FN and TN, true positives, false positives, false negatives and true negatives.}\label{tab:published}\\
\toprule Study & Order of oversampling and splitting & Test set & Headline sensitivity & $s$ & Check \\ \midrule \endfirsthead
\toprule Study & Order & Test set & Headline sensitivity & $s$ & Check \\ \midrule \endhead
""" + "\n".join(rows) + r"""
\bottomrule
\multicolumn{6}{p{15.9cm}}{\footnotesize \textit{Read but not assessable:} """ + na + r""". \textit{Ineligible:} """ + ie + r""".}\\
\end{longtable}}
"""
    (V / "manuscript_cmpb").mkdir(exist_ok=True)
    (V / "manuscript_cmpb" / "supp_v5_published.tex").write_text(table, encoding="utf-8")
    print("wrote supp_v5_published.tex")


if __name__ == "__main__":
    main()
