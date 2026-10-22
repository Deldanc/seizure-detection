# Repository and anticipatory feature audit

Audit date: 2026-10-22. Scope: the active seizure detector, the separate anticipatory GA–ANFIS path, feature construction, saved feature metadata, training splits, and inference consistency.

## Two distinct models

- `/predict` uses `models/ga_anfis_artifact.joblib` for seizure-activity detection in current EEG windows. Its saved quartet is `ch1_energy`, `ch6_rms`, `ch14_gamma_power`, and `ch18_std`. The artifact contains GA population/generation/seed/history metadata. Inspection of the saved GA configuration confirms these features were produced by a GA with population 12, six generations, seed 42, and a maximum of four inputs. That search targeted ictal-window detection, so the quartet is not evidence of optimal anticipatory features. Its historical gamma definition is 30–100 Hz.
- The often-described 107-feature Random Forest is a separate legacy artifact: `models/neuroguard_rf.joblib` deserializes as `RandomForestClassifier` with 107 inputs, matching the 107 entries in `models/selected_features.txt`. It is not the model used by `/predict`; the active route loads the four-input Sugeno ANFIS artifact above. The two artifacts implement different historical experiments and must not be conflated.
- `/anticipate` now loads `models/anticipatory_theta_alpha_event_v1.joblib`, a validation-selected two-feature ANFIS with one-window confirmation. The prior GA model `models/anticipatory_ga_anfis_featureaudit_v4.joblib` remains preserved and its poor results are documented. The candidate issued an alarm within the configured warning window for the previously examined CHB03_36 event, but only one of 240 positive windows crossed threshold; it also issued three warnings on two seizure-free CHB01 recordings. This is exploratory evidence, not a reliable performance claim.

## Feature extraction audit

- The earlier anticipatory schema had 230 candidates (five time features and five absolute band powers per each of 23 channels). It used gamma 30–100 Hz, no relative band power, SciPy Welch defaults (256-sample segments in a five-second window), and integrated only discrete bins inside each band. Exact band edges could be omitted.
- Schema v3 has 345 candidates: 15 features/channel, five time-domain descriptors, five absolute powers, and five relative powers. Bands are delta 0.5–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, and gamma 30–80 Hz. PSD boundaries are interpolated before integration. Relative power uses integrated 0.5–80 Hz power as the denominator. Units and formulas for every feature are recorded in `data/processed/anticipatory_feature_catalog.csv`.
- MNE supplies calibrated EEG values in volts. The pipeline causally resamples supported rates to 256 Hz, then applies the causal fourth-order 0.5–100 Hz Butterworth bandpass and causal 60 Hz notch. Five-second windows use two-second Hann Welch segments with 50% segment overlap, constant detrending, density scaling, and 0.5 Hz bins at 256 Hz. Signal and feature code is shared by preparation and inference.
- Features `chN_*` map to zero-based channel indices in the saved canonical channel list. Inference accepts channel permutations and reorders by name. Missing, extra, or ambiguous names fail validation. Sampling rates below 177.78 Hz fail because the preprocessing filter must pass the full 30–80 Hz gamma range. Higher supported rates use a causal polyphase FIR; its delay is retained.
- The four-input cap is explicit in the GA command and the ANFIS implementation (2^d fuzzy rules). It is a computational/model implementation constraint, not an experimentally proven optimum. The GA may choose one through four features from the full candidate set; no band is forced.

## Data and label protocol

- Seven one-hour EDFs from CHB-MIT subject chb03 are available locally at 256 Hz with 23 channels. Annotated EDFs 02, 03, and 04 are missing. No other subject is present.
- Summary onset and offset times are recording-relative. With five-second windows, a 20-minute prediction horizon, and a five-minute SPH, positives satisfy `onset - 25 min < window_end <= onset - 5 min`. SPH, ictal, postictal, and incomplete-history regions are excluded rather than mislabeled as positive or negative. The 30-minute postictal exclusion is applied after each annotation.
- Preparation retained the full timeline of 5,040 windows; 3,520 have eligible labels (720 preictal, 2,800 interictal) and 1,520 are unlabeled/ineligible. Full-timeline feature rows allow alarms to be evaluated even during SPH, ictal, postictal, and incomplete-history intervals. The fixed split uses chb03_34/15/16/17 for train, chb03_35 for validation, and chb03_36 for final test. The test recording is not used in GA search, feature comparison, scaling, model fitting, or threshold choice.

## Selection and evaluation

- The versioned anticipatory GA searched all 345 features using training-only five-minute grouped temporal CV and selected `ch1_rms`, `ch10_gamma_relative_power`, `ch13_gamma_relative_power`, and `ch17_mean`. The search setting permits one to four features; the selected count is four in this run.
- Validation comparisons retrained ANFIS with the GA subset, the existing detector quartet recomputed under schema v3 definitions, and theta/alpha absolute and relative pairs. The theta/alpha absolute pair had higher window F1/ROC-AUC; the existing quartet had higher precision and no validation false alarms but lower sensitivity/F1. These one-event tradeoffs are exploratory and unstable.
- The final v4 chb03_36 result at threshold 0.89 is in `reports/evaluation_report.md` and `results/csv/anticipatory_featureaudit_v4_evaluation.json`. It matched 0/1 events, generated two false alarms in one full recording hour, and had 62.9% window sensitivity, 32.7% specificity, 0.712 F1, 0.756 average precision and 0.373 ROC-AUC. The preictal prevalence is 83.0%, so window metrics are a poor proxy for advance warning. No improvement is claimed.
- A genuinely event-aware GA validation is not possible with only one labeled training seizure. Five-minute temporal groups reduce adjacent-window leakage but are not independent events. Event-generalization and patient-independent performance remain incomplete.

## Verification

- Rebuilt the schema v3 feature set and catalog from all seven local EDFs.
- Trained and saved the versioned ANFIS artifact with its exact ordered selected list, scaler, feature schema, band/filter/resampling settings, channels, threshold, and alarm protocol. V4 fixes an objective/gradient mismatch in L2 regularization; the saved ANFIS converged.
- Compared real EDF inference scores against held-out CSV scores (maximum absolute difference 5.77e-13), checked artifact dimensions and serialization, tested causal filter/resampler prefix invariance, channel reordering, label boundaries, full-timeline event alarms, and Flask output.
- Exact test command and result are recorded in `reports/requirement_verification.md`; 26 tests pass.

## Limits

This implementation is a research prototype, not clinical validation. The local dataset has one patient and only one event in each of train, validation, and test. Test-event sensitivity, false alarms per eligible hour, and lead-time estimates from one event are highly uncertain. Scores are uncalibrated. More subjects and events are required for an event-independent and patient-independent assessment.
