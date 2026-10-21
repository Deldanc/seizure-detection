# Final requirement audit: GA–ANFIS anticipatory seizure research prototype

**Audit date:** 2026-10-22
**Project:** Development of a Neuro-Fuzzy-Genetic Hybrid Framework for Anticipatory Seizure Prediction
**Active anticipatory artifact:** `models/anticipatory_ga_anfis_featureaudit_v4.joblib`
**Overall verdict: PARTIALLY IMPLEMENTED.** The repository contains a functioning, causal, annotation-based future-risk pipeline that uses a GA-selected feature list and a genuinely trained Sugeno ANFIS. The final held-out seizure was not predicted, however, and the available data cannot support a reliable event-independent or patient-independent performance claim. Treat this as an experimental research implementation, not a demonstrated successful seizure predictor.

## Requirement checklist

| Area | Implementation and verification | Status |
|---|---|---|
| Local data and annotations | `data/raw/chb03-summary.txt`, seven local EDFs, and `data/processed/dataset_manifest.csv` inspected. Seven one-hour recordings from one subject (chb03) are present; each has 23 channels at 256 Hz. The summary contains seven seizure annotations, but EDFs 02/03/04 are absent. Local files 01/34/35/36 contain annotations; file 01 has only 362 seconds of pre-onset history and contributes no prediction event. | PASS for inventory; data scope is a major limitation |
| Candidate features and EEG bands | `src/preprocessing/anticipatory_pipeline.py`, `data/processed/anticipatory_feature_catalog.csv`, and tonal tests verified. There are 345 candidates (15 per channel): mean, population standard deviation, population variance, RMS, sum-of-squares energy, and absolute/relative power for each band. Exact ranges: **delta 0.5–4 Hz; theta 4–8 Hz; alpha 8–13 Hz; beta 13–30 Hz; gamma 30–80 Hz.** Tests confirm contiguous boundaries and band-dominant synthetic tones. | PASS |
| Units, channel mapping, and preprocessing | MNE-calibrated samples are in volts; absolute powers are V²; relative powers are unitless; energy is V²·samples. Features use two-second Hann Welch segments, 50% overlap, 0.5 Hz bins, exact-edge interpolation and 0.5–80 Hz relative-power normalization. Causal fourth-order 0.5–100 Hz Butterworth and 60 Hz notch filtering follow one-sided causal FIR resampling to 256 Hz. Channel order is normalized by name; missing/extra/ambiguous layouts reject. | PASS |
| Labels and anticipatory timing | `prediction_labels` uses recording-relative seizure onset/offset annotations. A completed five-second window is positive if its end time satisfies `T−25 min < end <= T−5 min` (20-minute occurrence period after a 5-minute SPH). SPH, ictal, 30-minute postictal, and incomplete-history intervals are unlabeled, not positive. Files are never concatenated across recording gaps. | PASS for implemented protocol |
| Causality and split isolation | Causal resampling/filter prefix-invariance tests pass. Each feature window is complete at its prediction timestamp; preprocessing has no learned whole-dataset normalization. The scaler fits training rows only. Recording sets do not overlap: train 15/16/17/34, validation 35, test 36. The test recording is excluded from GA, threshold selection, feature comparisons and fitting. | PASS for leakage checks |
| Genetic Algorithm | `src/models/ga_anfis.py::GeneticFeatureSelector` implements seeded binary chromosomes, initialization, grouped folds, fitness, top-third parent selection, uniform crossover, mutation, elitism, feature penalty and stopping. The saved v4 history/config confirms an actual GA search over 345 candidates. It selected `ch1_rms`, `ch10_gamma_relative_power`, `ch13_gamma_relative_power`, `ch17_mean`. | PARTIAL: fitness is a Random Forest proxy evaluated on five-minute groups within training, not ANFIS validation performance. Validation compares the GA subset but does not select the GA winner. This distinction is retained to avoid reusing one event for both selection and threshold tuning. More independent validation events are needed to satisfy strict validation-based GA selection. |
| Genuine ANFIS and artifact integration | The saved model is `src.models.ga_anfis.SugenoANFIS`: two Gaussian memberships per feature, 16 product fuzzy rules for four features, normalized rule firing, first-order linear Sugeno consequents, and joint premise/consequent optimization. Inspection found an objective/gradient mismatch in v3. The L2 term is now included in the scalar loss as well as its gradient; v4 converges in 102 L-BFGS iterations (projected-gradient norm `1.85e-5`). v2/v3 outputs were retained in the archive. | PASS for v4 model; v3 retained as historical |
| Ordered feature list and inference preprocessing | `inference/anticipatory_predictor.py::load_artifact` verifies schema, unique/available feature names and equal model/scaler/list dimensions. v4 has 4 inputs in all three. Training and inference call the same causal resampler, filter and feature extractor. Real-EDF upload scores match the held-out CSV to maximum absolute difference `5.77e-13`. | PASS |
| Warning policy and timestamps | Risk is scored per completed five-second window. Three consecutive threshold crossings issue a warning at the third window end; 30-minute refractory suppression applies. The warning interval is `[alarm+5 min, alarm+25 min]`; event matching uses the same SPH-plus-horizon limits. Alarms are evaluated across the full timeline, while window metrics use eligible labels. | PASS for retrospective protocol |
| Flask upload, result page and export | A Flask test-client integration uploaded the real `data/raw/chb03_36.edf`, processed it with the active v4 artifact, rendered the result page, and exported 720 window rows with timestamps, scores, predictions and warning columns. The Flask server also launched on 127.0.0.1:5012 and the dashboard returned HTTP 200 over loopback. Invalid suffix/malformed EDF requests return 400; missing model returns 503. `/anticipate` does not fall back to the current-activity detector. The UI documents horizon/SPH, score meaning, research limits, channel mapping and supported rates. | PASS for upload workflow |
| Dependencies and reproducibility | `requirements_web.txt` had incompatible NumPy bounds and a malformed final dependency line; it is now aligned with `requirements.txt` plus Gunicorn. `.venv/bin/python -m pip check` reported no broken requirements. Seeds, GA configuration/history, scaler, selected list, channels, schema, bands, filters, resampling, split, threshold and warning settings are stored/documented. | PASS for current environment |
| Predictive performance | Held-out test predictions were independently recomputed from `results/csv/anticipatory_featureaudit_v4_predictions.csv`; see evidence below. | FAIL for a successful prediction claim |
| Patient/event generalization | Only one subject and one usable complete seizure event in each train, validation and test are available. | BLOCKED by available data |

## Critical gaps

1. **The held-out seizure was missed.** Test alarms occurred at 35 s and 1980 s; neither alarm's future-risk interval contains the annotated onset at 1725 s. Event sensitivity is 0/1 and no test warning lead time exists. The project genuinely computes anticipatory risk, but it has not demonstrated successful prediction on the untouched test event.
2. **Reliable generalization cannot be evaluated from these files.** The test is one event from the same patient as training and validation. Patient-independent evaluation is impossible with the local data. No population or clinical claim is supported.

## Major gaps

- GA fitness is grouped-CV Random Forest F1 on five-minute training blocks; the final GA winner is not chosen by validation performance or ANFIS fitness. Validation results are used for subset comparison and threshold selection. Strictly choosing features on the same single validation event would double-use it; more independent recordings/events are required to design separate selection and threshold validation.
- The test set has 289 eligible windows, of which 240 (83.0%) are positive. The remaining 49 eligible negatives are concentrated before the preictal interval; much of the one-hour recording is excluded by SPH, ictal, and postictal rules. Window scores therefore do not represent a balanced or broad interictal-monitoring test. Average precision 0.756 is below the positive-prevalence baseline 0.830.
- The Flask workflow analyzes a fully uploaded EDF offline. Window calculations are causal and timestamped as if generated at each window end, but the application is not a live streaming monitor and displays results after processing the recording.
- Input rates below 177.78 Hz and channel sets other than the trained 23-channel montage are rejected. Non-256 Hz resampling has unit coverage at 250 Hz but no real non-256 CHB-MIT EDF integration test.
- No automated artifact rejection, electrode-quality handling, or explicit within-file discontinuity/segment detection is implemented. Recordings remain separate and no cross-file time gaps are bridged.

## Minor gaps

- The `/predict` workflow is a separate seizure-activity detector, not the anticipatory model. It retains its older 30–100 Hz gamma definition and different feature pipeline; `/anticipate` uses 30–80 Hz and schema v3. The UI and route names now distinguish these tasks.
- The result chart loads Chart.js from a CDN, so chart rendering depends on network access even though model inference is local.
- The ANFIS rule count grows exponentially with inputs; the current implementation permits one to four selected features. Four is a computational cap, not an experimentally justified optimal count.

## Corrections made during this audit

- Corrected the Sugeno ANFIS objective so its L2 penalty matches the analytic gradient. Saved v4 optimizer metadata includes status, gradient norm and convergence. The prior v3 model and reports remain preserved under `archive/anticipatory_pre_feature_audit_20261022/schema_v3_pre_optimizer_fix/`.
- Repointed `/anticipate`, dashboard evaluation data, README outputs and tests to v4.
- Corrected the anticipatory upload-page text: channel order is matched by name and supported input rates are resampled to 256 Hz, rather than requiring exactly 256 Hz and exact channel order.
- Fixed `requirements_web.txt` dependency conflicts and malformed concatenated line.
- Added frequency-range tone checks, ANFIS convergence/scaler checks, real EDF upload/result/CSV integration coverage, malformed-file handling, and missing-model/no-fallback checks.
- Recomputed held-out window metrics independently from the saved v4 CSV and updated evaluation/audit documentation. No performance improvement is claimed.

## Actual evaluation evidence

Dataset preparation produced **5,040 windows** from seven local recordings; **3,520 windows** have usable labels (720 preictal and 2,800 interictal). The recording split contains four training recordings (2,496 eligible windows, 240 positive), one validation recording (458 eligible, 240 positive), and one test recording (289 eligible, 240 positive). This is **one patient**; train/validation/test each contain one full-history seizure event.

At validation-selected threshold **0.89**, the v4 test confusion counts are TN/FP/FN/TP = **16/33/89/151**. Independently recomputed window metrics: precision **0.821**, sensitivity **0.629**, specificity **0.327**, F1 **0.712**, average precision **0.756**, ROC-AUC **0.373**. Event-level result: **0/1** seizures predicted, **2 false alarms per 1.000 recording hour** (4.983 per eligible hour), and **no matched warning lead time**. The model's high window sensitivity must not be presented alone; the event was missed and its specificity/ranking are poor.

Validation subset comparison is in `reports/feature_selection_report.md`. On its one event, the GA subset had window F1 0.169 / ROC-AUC 0.167 and one false alarm; the retrained existing detector quartet had F1 0.296 / ROC-AUC 0.818 and zero false alarms; the theta+alpha absolute pair had F1 0.315 / ROC-AUC 0.842 and zero false alarms. Each reported 1/1 event hit, but these one-event results are unstable and were not used to replace the training-CV GA winner.

## Tests and commands

Executed successfully:

```sh
MPLCONFIGDIR=/private/tmp/mplconfig .venv/bin/python -m unittest discover -s tests -v
# Ran 25 tests; OK (3.9 seconds on this environment)

.venv/bin/python -m pip check
# No broken requirements found
```

The 25 tests include label boundaries, annotation parsing, disjoint recording splits, frequency-band dominance, candidate catalog/channel mapping, causal filter/resampler prefix invariance, GA reproducibility, ANFIS rule firing and save/load, trained artifact convergence and scaler fit, real EDF upload/result/CSV, malformed inputs, missing model behavior, and dashboard rendering.

Reproduction and launch from the project root:

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/preprocessing/prepare_anticipatory_data.py
python src/models/train_anticipatory.py
python -m unittest discover -s tests -v
python -m flask --app app run --host 127.0.0.1 --port 5001
```

## Remaining limits and next required work

Do not describe the current metrics as reliable anticipatory seizure prediction. The implementation is a functioning causal, future-risk research pipeline, but v4 missed its only held-out event. Obtain additional annotated recordings from multiple patients, preserve patient/recording-level splits, and create enough independent events to separate GA feature selection, threshold tuning, and final testing. Then repeat validation with event-aware outcomes and a fixed false-alarm objective. Artifact review, clinically meaningful warning utility, and external/patient-independent evaluation remain future work.
