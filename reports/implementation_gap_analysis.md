# Implementation gap analysis

## Current implementation

The repository has two distinct workflows. `/predict` is the older seizure-activity detector. `/anticipate` is a separate future-risk workflow and loads `models/anticipatory_ga_anfis_featureaudit_v4.joblib` without falling back to the detector.

The anticipatory pipeline extracts 345 features across 23 channels, with time statistics plus absolute and relative delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30–80 Hz) power. It maps channels by name, uses causal resampling/filtering, and labels completed windows from summary annotations using a 20-minute occurrence horizon, 5-minute SPH, and 30-minute postictal exclusion. The GA performs seeded grouped-CV training-block search; its final four-feature list is stored in the artifact. A first-order Sugeno ANFIS with Gaussian memberships, normalized product firing, and learned linear consequents performs the final scoring. Its objective/gradient mismatch was corrected for v4; v4 optimizer convergence is recorded and verified.

The Flask `/anticipate` route was exercised with an actual local EDF. It produced a result page and CSV for 720 windows. Malformed files return errors, missing model artifacts return 503, and there is no detector fallback. Training and inference use the same feature schema and preprocessing. Current full audit and evidence are in `reports/final_requirement_audit.md`.

## Remaining gaps

- The held-out chb03_36 seizure was missed: 0/1 event hits, one pre-onset false alarm, one late alert, and no warning lead time. The score pipeline is anticipatory in its labels and warning horizon, but predictive success was not demonstrated. The late alert is no longer mislabeled as a prospective false alarm.
- The local data are one patient and seven one-hour EDFs. EDFs 02/03/04 are missing; only one complete event is available in each train, validation, and test. Patient-independent and reliable event-independent evaluation are blocked by data availability.
- GA fitness is grouped-CV Random Forest F1 on training blocks, not ANFIS validation performance. The validation file is used for subset comparisons and threshold selection; using its single event to select features as well would double-use that event. Independent validation recordings/events are needed.
- The held-out eligible windows are 83% preictal and contain only 49 negatives. This short eligible interictal period makes window metrics a poor proxy for sustained monitoring performance.
- The app processes uploaded EDFs offline rather than streaming. Its per-window features are causal, but the UI displays results after upload and analysis.
- Only the CHB03 23-channel montage is accepted. Rates below 177.78 Hz are rejected; non-256 resampling has unit coverage but no real non-256 EDF integration evidence.
- Signal artifact rejection, acquisition quality review, explicit within-file discontinuity detection, external validation, calibration, and clinical validation are absent.

No population-level or clinical prediction claim is supported. See `reports/final_requirement_audit.md` for requirement-level statuses, test commands, actual metrics, and next steps.
