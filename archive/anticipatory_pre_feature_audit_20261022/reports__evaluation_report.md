# Anticipatory prediction evaluation

## Protocol and split

Prediction horizon: 20 min; SPH: 5 min; windows: 5.0 s.
Threshold 0.94 selected on validation recording only. GA features: `ch8_std, ch16_std, ch17_variance, ch17_energy`.
Train ['chb03_15.edf', 'chb03_16.edf', 'chb03_17.edf', 'chb03_34.edf']; validation ['chb03_35.edf']; test ['chb03_36.edf'].
Held-out sample counts: {'negative': 49, 'positive': 240}.

## Held-out chb03_36 results

Window metrics at the validation threshold: precision 1.000, sensitivity 0.025, specificity 1.000, F1 0.049, average precision 0.831, ROC-AUC 0.506; confusion matrix TN/FP/FN/TP `[49, 0, 234, 6]`.
Event result: 1/1 events; 0 false alarms (0.00 per eligible hour) over 0.401 eligible hours; lead-time distribution seconds `{'min': 1180.0, 'median': 1180.0, 'max': 1180.0}`.
Eligible monitoring time under warning: 40.5% (duration-based estimate).

## Baselines

Model | Precision | Sensitivity | Specificity | F1 | PR-AUC | ROC-AUC | TN/FP/FN/TP
---|---:|---:|---:|---:|---:|---:|---|
always_interictal | 0.000 | 0.000 | 1.000 | 0.000 | 0.830 | 0.500 | [49, 0, 240, 0]
logistic_regression | 0.800 | 0.117 | 0.857 | 0.204 | 0.824 | 0.456 | [42, 7, 212, 28]
random_forest | 0.823 | 0.329 | 0.653 | 0.470 | 0.838 | 0.532 | [32, 17, 161, 79]

## Interpretation and limits

- Only one subject is locally available.
- The test set contains one annotated seizure event.
- The GA fitness uses five-minute temporal groups within one training seizure recording; this is not independent event validation.
- Scores are not calibrated probabilities; threshold is selected on one validation event by sensitivity, false-alarm count, then window F1.
- The 20-minute horizon and 5-minute SPH are experimental settings, not clinical recommendations.
- Event matching counts an alarm only when seizure onset lies in alarm time + SPH through alarm time + SPH + horizon; three consecutive high-risk windows initiate an alarm, followed by a 30-minute refractory interval.
- A single event makes event sensitivity a one-trial result; false-alarm rate, lead-time distribution, and active-warning fraction are descriptive and highly unstable.
- The window ROC-AUC is close to chance and average precision is near the positive-class prevalence; the event hit must not be read as strong discrimination.
- This is not evidence of clinical reliability. Test metrics are not a prospective or patient-independent validation.

## Machine-readable outputs and plots

- `predictions_csv`: `results/csv/anticipatory_test_predictions.csv`
- `artifact`: `models/anticipatory_ga_anfis.joblib`
- `risk_plot`: `results/plots/anticipatory_test_risk.png`
- `confusion_plot`: `results/plots/anticipatory_test_confusion.png`
- `metrics_json`: `results/csv/anticipatory_evaluation.json`
