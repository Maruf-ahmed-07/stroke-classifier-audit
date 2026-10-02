# Auditing machine-learning stroke classifiers: code and released model

Code for the paper "Auditing machine-learning stroke classifiers: a forensic check for oversampling leakage, the cost of
post-event predictors, and a leakage-controlled pipeline with external validation" (submitted to Computer Methods and
Programs in Biomedicine).

This repository reproduces every number, table and figure of the paper and its supplement, and contains the
released model. Folder names are kept exactly as the scripts expect them.

## Quick use, no data needed
```
pip install numpy pandas xgboost
cd "cmpb_paper/release"
python predict.py --check          # reproduces the paper's predictions for 2,000 test respondents
python synthetic_share_check.py --pos 249 --neg 4861 --reported-sensitivity 0.955   # forensic check
```
The model card is `cmpb_paper/release/MODEL_CARD.md`.

## Full reproduction (several hours on a 16-core desktop)
1. `pip install -r requirements-cmpb.txt`
2. Download the BRFSS 2023 and 2024 SAS transport files (`LLCP2023.XPT`, `LLCP2024.XPT`) from
   https://www.cdc.gov/brfss/annual_data/annual_data.htm into `seprate experiment/`.
   For the optimism audit, also place the Kaggle stroke file (`healthcare-dataset-stroke-data.csv`) in `data/raw/kaggle/`.
   It is used only as the object of the audit.
3. `python "seprate experiment/scripts/run_separate_brfss_experiment.py"`: builds the analytic files.
4. Then run, in order:
   - `cmpb_paper/analysis/extract_survey_weights.py`
   - `cmpb_paper/analysis/rebuild_clean_data.py`: corrected missing-value codes
   - `cmpb_paper/analysis/run_paper_analysis.py`
   - `cmpb_paper/analysis/run_v2_analyses.py`
   - `cmpb_paper/analysis/run_simple_baseline.py`
   - `cmpb_paper/analysis/run_nhis_external_validation.py` (downloads the NHIS files from the CDC)
   - `cmpb_paper/robustness_experiments/run_robustness_experiments.py`
   - `cmpb_paper/robustness_experiments/run_identity_robustness.py`
   - `cmpb_paper/robustness_experiments/run_identity_generalisation.py`
   - `cmpb_paper/analysis/run_subgroup_and_residual.py` and `run_lr_onehot_check.py`
   - `cmpb_paper/analysis/run_design_ci.py`, `cmpb_paper/analysis/run_retune.py` (about 1-2 hours) and
     `cmpb_paper/robustness_experiments/run_identity_generalisation_ext.py`
   - `cmpb_paper/analysis/published_studies_check.py` (values extracted from the published studies)
   - `cmpb_paper/analysis/make_figures.py`, `make_supplement_tables.py`, `make_supplement_tables_v2.py`

The XGBoost hyperparameters are read from
`simillar reasearch test/rich_feature_best_4way_final_search/outputs/tables/T04_optuna_best_params.csv`, the
result of the Optuna search in the script next to it. The reported models use them; `run_retune.py` repeats the search on the corrected data as a check.
The result tables produced by our run are included under each `outputs/` folder for comparison.
`rebuild_clean_data.py` corrects the handling of special codes in the original thesis cleaning; the
supplement (Section S1) describes the correction and its effect.
