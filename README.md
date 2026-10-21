# NeuroGuard: GA–ANFIS seizure detection and anticipatory research

The Flask `/predict` workflow is the earlier seizure-activity detector. A new
and separate `/anticipate` workflow scores future seizure risk under an explicit
preictal labeling protocol. The anticipatory model is trained and evaluated
separately; the detector is never used as a silent fallback. This is a small
research prototype, not a clinically validated system.

## Local data

Seven one-hour EDFs from CHB-MIT subject chb03 are present. They use 256 Hz and
23 channels. The local files are 01, 15, 16, 17, 34, 35, and 36. The supplied
summary also annotates 02, 03, and 04, but those EDFs are absent. Only one
subject is available, so patient-independent validation is not possible.
See `reports/dataset_audit.md` and `data/processed/dataset_manifest.csv`.

The earlier detector dataset (`data/processed/features.csv`) has 5,033
five-second windows labeled by current seizure overlap. Do not use it as
anticipation labels. The new pipeline regenerates causal filtered features and
future-risk labels in `data/processed/anticipatory_features.csv`.

## Feature audit

The anticipatory extractor has 345 candidate features: 15 per EEG channel
(time statistics, absolute band power, and relative band power). The candidate
frequency ranges are **delta 0.5–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta
13–30 Hz, and gamma 30–80 Hz**. Absolute power integrates Welch PSD across
exact band edges; relative power divides by total 0.5–80 Hz power. See
`reports/feature_selection_report.md` and the complete feature/channel/formula/
unit catalog at `data/processed/anticipatory_feature_catalog.csv`.

The saved seizure-activity detector quartet (`ch1_energy`, `ch6_rms`,
`ch14_gamma_power`, `ch18_std`) was genuinely produced by a GA for the older
seizure-presence task. It is not presumed optimal for anticipation. The
anticipatory GA reselects from the new candidates using training-only grouped
cross-validation and compares its subset with that quartet and theta/alpha
candidates. The current ANFIS rule base limits subsets to at most four inputs because it
creates `2^d` fuzzy rules; four is an implementation constraint, not a proven
optimal feature count.

## Prediction protocol

Defaults are a 5-second window and step, a 20-minute seizure occurrence period,
5-minute SPH, and 30-minute postictal exclusion. The positive decision-time
interval before an annotated seizure at `T` is
`T - (20 min + 5 min) < window_end <= T - 5 min`. The SPH, ictal interval, and
postictal exclusion are not positive. See `reports/prediction_protocol.md` for
the full rules, causal filtering, features, and event matching.

The GA uses binary feature chromosomes, grouped temporal-block validation F1
with a feature-count penalty, and a reproducible seed. ANFIS is first-order
Sugeno with Gaussian membership functions, normalized product rule firing, and
trained linear consequents. Scores are uncalibrated. A warning is issued after
three consecutive threshold crossings; its risk interval starts after SPH and
lasts through the prediction horizon. Alarms are suppressed for 30 minutes.
No exact seizure onset time or countdown is reported.

## macOS setup and commands

The existing `.venv` is Python 3.12.5 on Apple Silicon. From the repository
root:

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/preprocessing/prepare_anticipatory_data.py
python src/models/train_anticipatory.py
python -m unittest discover -s tests -v
python -m flask --app app run --host 127.0.0.1 --port 5001
```

Preparation can be configured with `--window-seconds`, `--step-seconds`,
`--prediction-horizon-minutes`, `--sph-minutes`,
`--postictal-exclusion-minutes`, and `--interictal-guard-minutes`. The selected
settings are stored in `data/processed/anticipatory_config.json` and carried
through training and inference. Training accepts `--refractory-minutes`,
`--confirm-windows`, `--max-features` (1–4), `--population-size`, and
`--generations`. Its risk threshold is selected on validation event
sensitivity, then fewer full-timeline false alarms, then window specificity,
precision, and F1. In the app the validation-selected
threshold is the default; a slider override is marked unvalidated.

Training creates the versioned model
`models/anticipatory_ga_anfis_featureaudit_v4.joblib`,
`results/csv/anticipatory_featureaudit_v4_predictions.csv`,
`results/csv/anticipatory_featureaudit_v4_evaluation.json`,
`reports/feature_selection_report.md`, and `reports/evaluation_report.md`.
Earlier artifacts are preserved under
`archive/anticipatory_pre_feature_audit_20261022/`. The Flask app exposes
the original detector at `/predict` and anticipatory research at `/anticipate`.
The latter accepts the trained CHB03 23-channel set in any order and reorders
channels by name. It causally resamples input rates of at least 177.78 Hz to 256 Hz and
rejects missing/extra channels or rates unable to represent the 30–80 Hz gamma
band. Training and inference use the same causal preprocessing.

## Evaluation and limitations

The anticipatory split is chb03_34 and any locally present 02/03/04 files plus
seizure-free 15/16/17 for training, chb03_35 for validation/threshold choice,
and chb03_36 for the single test event. Only one training, validation, and test
seizure event are available.
GA fitness uses separated five-minute blocks within its training data; these
are not independent event folds. After the audit, the GA subset produced 0/1
event matches and 2 false alarms in the one-hour held-out recording. Its window
sensitivity was 62.9%, specificity 32.7%, F1 0.712, and ROC-AUC 0.373 at the
validation threshold. The high preictal-window prevalence makes these window
metrics an unreliable proxy for useful warning performance; average precision (0.756) is below the test-set positive prevalence baseline (0.830). Continuous-timeline evaluation includes alarms from unlabeled intervals; the previous 19m40s hit is retained only as a historical experiment. On validation, the absolute
theta/alpha pair had higher F1 and ROC-AUC than the GA subset; the existing
quartet had higher precision and zero false alarms but lower sensitivity and
F1. These one-event tradeoffs are unstable. No improvement is established.
Scores are not calibrated probabilities. The prototype has no
patient-independent evidence and must not guide clinical decisions.

Older CSVs, model artifacts, plots, and experiments are preserved. The prior
`results/csv/ga_anfis_evaluation.json` is the seizure-activity detector
evaluation, not the anticipatory evaluation.

The summary names missing seizure EDFs chb03_02, chb03_03, and chb03_04. To
restore only those local files from PhysioNet (rather than the full database),
run:

```sh
mkdir -p data/raw
for f in chb03_02.edf chb03_03.edf chb03_04.edf; do
  curl -fL "https://physionet.org/files/chbmit/1.0.0/chb03/$f" -o "data/raw/$f"
done
python src/preprocessing/prepare_anticipatory_data.py
python src/models/train_anticipatory.py
```
