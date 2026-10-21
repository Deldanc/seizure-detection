# Repository audit

Audit date: 2026-10-22. Workspace: `/Users/deldan/seizure-detection`.

## Project contents and execution path

- The repository contains a Flask site (`app.py`), templates and CSS, EDF inference in `inference/predictor.py`, feature preparation in `src/preprocessing/`, model training and evaluation scripts in `src/models/` and `src/evaluation/`, tests, and saved model/results directories.
- Before this task, `POST /api/analyze` called `inference.predictor.predict_edf`; it loaded `models/ga_anfis_artifact.joblib`, emitted a score for each five-second window, thresholded those scores, then merged positive windows. That artifact is the earlier seizure-activity detector. It is not an anticipatory model. The separate legacy RF files (`neuroguard_rf.joblib`, `neuroguard_scaler.joblib`) are not the active upload inference path.
- The old `src/preprocessing/features.py` used hardcoded seizure-overlap labels: a window was positive when at least half of its five seconds overlapped an ictal annotation. Its original loop omitted the last complete window. Its output in `data/processed/features.csv` had 5,033 rows and 230 descriptors for seven files.
- Existing `src/models/ga_anfis.py` implements binary GA chromosomes and first-order Sugeno ANFIS, but its earlier use selected features for the ictal-window detection task. The prior `ga_anfis_evaluation.json`, assorted CSV/PNG experiments, and RF artifacts concern earlier detector experiments and do not establish future prediction.
- Existing experimental scripts, CSVs, plots, and the standalone `seizure_detection_upload_app.py` were retained. Some historical dashboard plots/values are explicitly marked as archived/illustrative in the UI; they are not the new anticipatory evaluation.
- The current Flask app now has a distinct `/anticipate` page and `POST /api/anticipate` endpoint. It loads `models/anticipatory_ga_anfis.joblib`, checks the trained channel layout and 256 Hz sampling, computes causal filtered features, returns warning timing/risk intervals, and provides a CSV. It does not fall back to the detector if the new artifact is absent.

## Local data and environment

- Locally present raw EEG: seven EDFs from subject chb03, plus `chb03-summary.txt` and one `.seizures` sidecar. The complete CHB-MIT database is not present.
- The local summary lists seizures in files 01–04 and 34–36; the matching EDFs among local files are only 01, 34, 35, and 36. Files 02, 03, and 04, despite having summary annotations, are absent. Only 01 has a local sidecar. This prevents using the complete subject seizure history or a multi-subject split.
- Each available EDF was opened successfully in the preparation run: 3,600 seconds, 256 Hz, 23 channels. MNE disambiguates repeated `T8-P8` channel labels as `T8-P8-0` and `T8-P8-1`; their channel positions are retained. Details are in `data/processed/dataset_manifest.csv`.
- The new reproducible pipeline created 3,520 labeled/eligible five-second windows, including 720 preictal windows. `data/processed/anticipatory_features.csv` records feature values and annotation-derived labels. No synthetic EEG or labels were used.
- Existing `.venv` is Python 3.12.5 on macOS 26.5.1 arm64; dependencies are declared in `requirements.txt`. No package installation was needed for this implementation.

## New research workflow

`src/preprocessing/anticipatory_pipeline.py` parses the local summary, applies a causal 0.5–100 Hz filter and causal 60 Hz notch, extracts the same 230 time/frequency family over non-overlapping five-second windows, and applies the prediction protocol in `reports/prediction_protocol.md`. The separate training script evolves GA-selected subsets on the training partition, fits Sugeno ANFIS, selects a score threshold on a held-out validation recording, evaluates on a separate held-out seizure recording, compares baselines, writes a machine-readable prediction CSV, saves an artifact, and creates an evaluation report/plots.

The evaluation is a minimal same-subject demonstration: chb03_34 trains (with three seizure-free local recordings supplying interictal examples; any locally restored chb03_02/03/04 files are assigned to training), chb03_35 validates, and chb03_36 is the single-event test. The GA's internal fitness folds are five-minute temporal blocks within the training set because only one labelable training seizure event exists in the current local files. This cannot establish event-independent performance. No patient-independent test is possible from the local files.

## Verification performed

- Rebuilt the anticipatory feature dataset from all seven local EDFs.
- Ran `.venv/bin/python -m unittest discover -s tests -v`; fifteen tests pass.
- Trained/evaluated the anticipatory model on the fixed recording split (results in `reports/evaluation_report.md`, `results/csv/anticipatory_evaluation.json`, and machine-readable `results/csv/anticipatory_test_predictions.csv`).
- Exercised the Flask upload/prediction result and CSV route in a unit test using a stubbed predictor; separately verified a real local EDF through the inference path after training.
