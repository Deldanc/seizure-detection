# Anticipatory prediction evaluation

## Protocol and split

Prediction horizon 20 min; SPH 5 min; windows 5.0 s. Feature schema v2 has 345 features.
Training-only GA chose `ch5_beta_relative_power, ch7_gamma_relative_power, ch17_energy, ch19_gamma_relative_power` using grouped five-minute blocks; validation selected threshold 0.96.
Train ['chb03_15.edf', 'chb03_16.edf', 'chb03_17.edf', 'chb03_34.edf']; validation ['chb03_35.edf']; untouched test ['chb03_36.edf'].
Test class counts: {'negative': 49, 'positive': 240}.

## Held-out test results

Window metrics: precision 0.833, sensitivity 0.062, specificity 0.939, F1 0.116, AP 0.798, ROC-AUC 0.414; TN/FP/FN/TP [46, 3, 225, 15].
Event result: 0/1 events; 0 false alarms over 0.401 eligible hours (0.000/eligible hour); lead times in seconds [].
Eligible monitoring time under warning: 0.0%.

Test set is one annotated event in one patient; these are descriptive one-event estimates and do not support a performance claim.

## Validation feature-subset comparison

See `reports/feature_selection_report.md` for the same-protocol validation metrics for the GA subset, existing detector quartet, and theta/alpha absolute/relative subsets. Validation metrics are exploratory and were not used to select among the GA candidates.

## Baselines

Model | Precision | Sensitivity | Specificity | F1 | AP | ROC-AUC | TN/FP/FN/TP
---|---:|---:|---:|---:|---:|---:|---|
always_interictal | 0.000 | 0.000 | 1.000 | 0.000 | 0.830 | 0.500 | [49, 0, 240, 0]
logistic_regression | 0.855 | 0.246 | 0.796 | 0.382 | 0.833 | 0.468 | [39, 10, 181, 59]
random_forest | 0.832 | 0.988 | 0.020 | 0.903 | 0.782 | 0.381 | [1, 48, 3, 237]

## Limitations

- Only one subject is locally available; three annotated chb03 EDFs are missing.
- Training contains one seizure event; training-only GA folds are five-minute temporal recording blocks, not independent-event folds.
- Validation and test each contain one annotated seizure event. Candidate validation metrics and selected thresholds are unstable.
- The test recording is not used for GA selection, feature comparisons, scaler/model fitting, or threshold selection.
- Scores are uncalibrated. Rates below 177.78 Hz and channel sets other than the saved montage are rejected; supported rates are causally resampled.
- The four-feature cap is an ANFIS implementation constraint (2^d fuzzy rules), not an empirically established optimum.
- Threshold alarms require three consecutive crossings; matching requires onset inside the SPH-plus-horizon interval.
- A one-event event-sensitivity result is not a dependable prediction estimate.
