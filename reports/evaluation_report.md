# Anticipatory prediction evaluation

## Protocol and split

Prediction horizon 20 min; SPH 5 min; windows 5.0 s. Feature schema v3 has 345 features.
Training-only GA chose `ch1_rms, ch10_gamma_relative_power, ch13_gamma_relative_power, ch17_mean` using grouped five-minute blocks; validation selected threshold 0.89.
Train ['chb03_15.edf', 'chb03_16.edf', 'chb03_17.edf', 'chb03_34.edf']; validation ['chb03_35.edf']; untouched test ['chb03_36.edf'].
Test class counts: {'negative': 49, 'positive': 240}.

## Held-out test results

Window metrics: precision 0.821, sensitivity 0.629, specificity 0.327, F1 0.712, AP 0.756, ROC-AUC 0.373; TN/FP/FN/TP [16, 33, 89, 151].
Event result: 0/1 events anticipated; 1 pre-onset unmatched false alarm over 1.000 recording hour (1.000/recording hour; 2.491/eligible hour), plus 1 late alert at 1980 s (255 s after onset). Lead times: none.
Eligible monitoring time under warning: 75.8%.

Alarms are generated on the full continuous recording timeline. Window metrics use only labeled eligible windows; late/post-onset alerts are reported separately from prospective false alarms. The first alarm at 35 s forecast 335–1535 s, ending 190 s before the annotated onset at 1725 s. Its 30-minute refractory interval suppressed alarms until 1835 s; the next alarm at 1980 s was late. The refractory setting was selected using validation only; this test timeline explains the miss and was not used to retune it. The test set is one annotated event in one patient; these descriptive one-event estimates do not support a performance claim.

Independent reproduction from the saved model and annotation-derived labels found zero label mismatches and maximum score difference 1.1e-16 from the saved predictions. The class-1 score is the documented preictal score. Reversing the score would raise ROC-AUC to 0.627, but is unsupported by the model's class ordering and task definition; the poor AUC reflects distribution shift, not an indexing mistake.

## Validation feature-subset comparison

See `reports/feature_selection_report.md` for the same-protocol validation metrics for the GA subset, existing detector quartet, and theta/alpha absolute/relative subsets. Validation metrics are exploratory and were not used to select among the GA candidates.

## Baselines

Model | Precision | Sensitivity | Specificity | F1 | AP | ROC-AUC | TN/FP/FN/TP
---|---:|---:|---:|---:|---:|---:|---|
always_interictal | 0.000 | 0.000 | 1.000 | 0.000 | 0.830 | 0.500 | [49, 0, 240, 0]
logistic_regression | 0.829 | 0.746 | 0.245 | 0.785 | 0.801 | 0.454 | [12, 37, 61, 179]
random_forest | 0.815 | 0.458 | 0.490 | 0.587 | 0.828 | 0.458 | [24, 25, 130, 110]

## Limitations

- Only one subject is locally available; three annotated chb03 EDFs are missing.
- Training contains one seizure event; training-only GA folds are five-minute temporal recording blocks, not independent-event folds.
- Validation and test each contain one annotated seizure event. Candidate validation metrics and selected thresholds are unstable.
- The test recording is not used for GA selection, feature comparisons, scaler/model fitting, or threshold selection.
- Scores are uncalibrated. Rates below 177.78 Hz and channel sets other than the saved montage are rejected; supported rates are causally resampled.
- The four-feature cap is an ANFIS implementation constraint (2^d fuzzy rules), not an empirically established optimum.
- Threshold alarms require three consecutive crossings; matching requires onset inside the SPH-plus-horizon interval.
- A one-event event-sensitivity result is not a dependable prediction estimate.
