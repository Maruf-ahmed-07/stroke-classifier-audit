# Model card: pre-event stroke-status model (BRFSS 2023)

Follows Mitchell et al. (2019), "Model cards for model reporting". Numbers come from `analysis/outputs/`.

## Model details
- **Model.** XGBoost gradient-boosted trees (binary logistic), followed by isotonic probability calibration.
- **Inputs.** 37 self-reported survey items: the "pre-event" set, which excludes disability, health-status, treatment and healthcare-contact items that can follow a stroke.
- **Output.** A calibrated probability that the respondent reports a previous physician-diagnosed stroke, plus flags at three operating points.
- **Developers.** MD Maruf Ahmed, MD Aminur Rahman, MD Junaid Ahmed Zama, Farhin Rahman, Najifa Tahsin, Jannatun Noor (BRAC University).
- **Paper.** Computer Methods and Programs in Biomedicine (submitted).
- **Code and model.** https://github.com/Maruf-ahmed-07/stroke-classifier-audit (folder `cmpb_paper/`).
- **Licence.** As in the code repository.

## Files (`model/`)

| File | Content |
|---|---|
| `xgb_pre_event.json` | XGBoost booster; use trees `0 .. best_iteration` (see `model_meta.json`) |
| `preprocessing.json` | Predictor order; training-set medians, means and scales for numeric items; most frequent level and category list for one-hot items (the first level is the dropped reference) |
| `calibration_isotonic.csv` | Isotonic map from raw score to calibrated risk; apply with linear interpolation, clipped at the ends |
| `thresholds.csv` | Broad, primary and focused thresholds, fixed on the threshold-selection split at 80%, 60% and 40% target sensitivity |
| `model_meta.json` | Early-stopping iteration, library versions, hyperparameters |
| `check_inputs.csv`, `check_predictions.csv` | 2,000 test rows and the paper pipeline's predictions for them; `python predict.py --check` must reproduce them |

## Intended use
- **Intended.** Research on questionnaire-based risk flagging and screening prioritisation in population survey data, and as a reference implementation for leakage-controlled evaluation.
- **Task.** Detection of *prevalent*, self-reported stroke. The model does not predict incident stroke.
- **Not intended.** Diagnosis, individual clinical decisions, or triage in health-care settings. The model has not been evaluated in clinical populations.

## Training and evaluation data
- **Training data.** BRFSS 2023 public-use file, restricted to respondents who answered the stroke question (n = 431,849; 4.25% report stroke). The data were split into training (60%), calibration (10%), threshold-selection (10%) and test (20%) sets with seed 42.
- **Evaluation data.**
  - The untouched 2023 test set.
  - BRFSS 2024: a 35-predictor version, because two items are absent from 2024.
  - NHIS 2012 and 2014–2018: a harmonised 18-predictor version, evaluated for prevalent stroke and for cerebrovascular death.

## Performance (2023 test set, primary threshold)
See the paper's Table 2 and `analysis/outputs/table2_performance.csv`:

- ROC-AUC 0.811 (95% CI 0.805–0.817)
- sensitivity 0.610, specificity 0.820, PPV 0.131
- about 6.6 false positives per detected case
- calibration ICI 0.003

Other evaluations:
- **BRFSS 2024** (frozen 35-predictor version): ROC-AUC 0.805, calibration slope 0.96.
- **NHIS 2012–2018** (frozen harmonised 18-predictor version):
  - prevalent stroke: ROC-AUC 0.838, observed/expected 0.99;
  - cerebrovascular death: ROC-AUC 0.845, which is **below age alone (0.883)**. Do not use the model as a prognostic tool.

## Ethical considerations and caveats
- **Outcome.** The label is self-reported. Against medical records, self-reported stroke has high specificity but moderate PPV, and transient ischaemic attacks may be reported as strokes.
- **Survivor bias.** BRFSS excludes people who died or live in institutions, so the most severe strokes are under-represented.
- **Age.** One threshold behaves differently by age: sensitivity is low in adults under 45, and specificity is low at age 80 and over (see the supplement's subgroup table). Consider age-specific thresholds or presenting the risk without a threshold.
- **Residual post-event information.** Hypertension or diabetes diagnosed after a stroke can remain among the predictors.
- **Input precision.** Pass values at full precision. Rounded inputs, for example drinking days per month = 30/7, can fall on the other side of a tree split.
- **Missing inputs.** Missing values are imputed with training-set statistics. Rows with many missing items get predictions close to those of an average respondent, and should be interpreted with care.

## How to use
```
pip install numpy pandas xgboost
python predict.py --check                 # verifies the release against the paper pipeline
python predict.py input.csv output.csv    # input columns and codes as in preprocessing.json / Table S1
```
