# Model card: pre-event stroke-status model (BRFSS 2023)

This card follows the model-card format of Mitchell et al. (2019). All numbers come from `cmpb_paper/analysis/outputs/`.

## Model details
The model is an XGBoost classifier (gradient-boosted trees, binary logistic objective) followed by isotonic calibration. Its inputs are 37 self-reported BRFSS survey items, the "pre-event" set, which leaves out disability, health-status, treatment and healthcare-contact items because these can change after a stroke. It outputs a calibrated probability that a respondent reports a previous physician-diagnosed stroke, and flags at three operating points.

Developed by MD Maruf Ahmed, MD Aminur Rahman, MD Junaid Ahmed Zama, Farhin Rahman, Najifa Tahsin and Jannatun Noor (BRAC University), for a paper submitted to Computer Methods and Programs in Biomedicine. Code and model: https://github.com/Maruf-ahmed-07/stroke-classifier-audit (folder `cmpb_paper/`), under the repository's licence.

## Files in `model/`

| File | Content |
|---|---|
| `xgb_pre_event.json` | XGBoost booster; use trees `0 .. best_iteration` (see `model_meta.json`) |
| `preprocessing.json` | Predictor order; training-set medians, means and scales for numeric items; most frequent level and category list for one-hot items (the first level is the dropped reference) |
| `calibration_isotonic.csv` | Isotonic map from raw score to calibrated risk; apply with linear interpolation, clipped at the ends |
| `thresholds.csv` | Broad, primary and focused thresholds, fixed on the threshold-selection split at 80%, 60% and 40% target sensitivity |
| `model_meta.json` | Early-stopping iteration, library versions, hyperparameters |
| `check_inputs.csv`, `check_predictions.csv` | 2,000 test rows and the paper pipeline's predictions for them; `python predict.py --check` must reproduce them |

## Intended use
The model is meant for research on questionnaire-based risk flagging and screening prioritisation in population survey data, and as a reference implementation of leakage-controlled evaluation. It detects prevalent, self-reported stroke; it does not predict incident stroke. It is not meant for diagnosis, individual clinical decisions or triage, and it has not been evaluated in clinical populations.

## Training and evaluation data
Training used the BRFSS 2023 public-use file, restricted to respondents who answered the stroke question (n = 431,849; 4.25% report stroke), split with seed 42 into training (60%), calibration (10%), threshold-selection (10%) and test (20%) sets. The model was evaluated on the untouched 2023 test set; a 35-predictor version on BRFSS 2024, because two items are missing from the 2024 core; and a harmonised 18-predictor version in NHIS 2012 and 2014–2018, for prevalent stroke and for cerebrovascular death.

## Performance
On the 2023 test set (paper, Table 2; `table2_performance.csv`) the ROC-AUC was 0.811 (95% CI 0.805–0.817) and the integrated calibration index 0.003. At the primary threshold, sensitivity was 0.610, specificity 0.820 and PPV 0.131, about 6.6 false positives per detected case.

The frozen 35-predictor version reached ROC-AUC 0.805 on BRFSS 2024 (calibration slope 0.96). In NHIS the harmonised version reached 0.838 for prevalent stroke (observed/expected 0.99). For cerebrovascular death it reached 0.845, below age alone (0.883), so the model should not be used as a prognostic tool.

## Caveats
- The label is self-reported. Against medical records, self-reported stroke has high specificity but moderate PPV, and transient ischaemic attacks may be reported as strokes.
- BRFSS excludes people who died or live in institutions, so the most severe strokes are under-represented.
- A single threshold behaves differently by age: sensitivity is low under 45 and specificity is low at 80 and over (supplement, Table S6). Age-specific thresholds, or reporting the risk without a threshold, may suit some uses better.
- Hypertension or diabetes diagnosed after a stroke can remain among the predictors.
- Pass inputs at full precision; rounded values (for example drinking days per month) can fall on the other side of a tree split.
- Missing values are imputed with training-set statistics, so rows with many missing items get predictions close to those of an average respondent.

## How to use
```
pip install numpy pandas xgboost
python predict.py --check                 # verifies the release against the paper pipeline
python predict.py input.csv output.csv    # input columns and codes as in preprocessing.json / Table S1
```
