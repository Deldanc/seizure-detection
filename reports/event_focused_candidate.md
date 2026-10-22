# Event-focused theta/alpha ANFIS candidate

## Candidate and validation selection

This version is separate from the GA-selected v4 baseline. The theta/alpha absolute-power pair was the strongest simple alternative in the existing CHB03_35 validation comparisons. I retrained Sugeno ANFIS using only `ch14_theta_power` and `ch14_alpha_power`, then compared 1, 2, and 3 consecutive-window alarm confirmation policies on validation. With each policy, the validation threshold was selected by event hit, fewer alarms, specificity, precision, and F1. The selected policy was **one threshold-crossing window at score 0.99**, with a 30-minute refractory period.

On the single validation event, this setting produced one matched warning 1,002 seconds before onset, no false alarms, and window sensitivity 1.7% (4/240). Validation ROC-AUC was 0.842. This is only one event and is highly uncertain.

## CHB03_36 result

The saved candidate was run through the same live inference function used by `/anticipate`. It emitted one warning at **1,285 seconds**. The warning's predicted risk interval is **1,585–2,785 seconds**. The annotated seizure onset is **1,725 seconds**, so this historical event fell inside the interval, **440 seconds (7 minutes 20 seconds) after the warning**.

Window-level results remain weak: 1 of 240 preictal windows crossed the threshold (sensitivity 0.4%), specificity 100%, F1 0.008, ROC-AUC 0.550, and average precision 0.853. Event-level result is 1/1 with no false alarm in that one-hour recording. The three-window confirmation version emitted no warning on this test event; the one-window confirmation policy was selected on validation, not by optimizing this event.

## Seizure-free CHB01 negative controls

The same saved model, feature order, preprocessing, threshold (0.99), and one-window confirmation policy were applied without retuning to `chb01_01.edf` and `chb01_36.edf`. Their annotations list no seizures. The model emitted one warning in `chb01_01` at 30 s, and two in `chb01_36` at 200 s and 2,005 s: **3 warnings across 2 recording hours (1.5 per recording hour)**. These are false alarms for these annotated seizure-free files. This makes clear that the CHB03 event hit does not establish a useful operating point; in particular, an isolated high score can trigger an alarm under the selected one-window policy.

## Interpretation and limitations

This candidate **did issue a pre-onset warning for the known CHB03_36 sample**, under its defined 5-minute SPH and 20-minute occurrence interval. That demonstrates how the warning workflow operates, not reliable prediction. The test event has been examined in earlier project iterations, there is only one validation and one test event, window sensitivity is 0.4%, and the CHB01 negative controls produced 3 warnings in 2 hours. The threshold is not calibrated as a probability. More independent seizure events and substantially longer seizure-free recordings are needed to estimate event sensitivity and false alarms per hour with useful precision.

The active `/anticipate` route uses `models/anticipatory_theta_alpha_event_v1.joblib`. The previous GA-selected v4 artifact remains available at `models/anticipatory_ga_anfis_featureaudit_v4.joblib` for comparison. This two-feature candidate is ANFIS but was not selected by the GA; it is clearly labeled as a validation-selected experimental alternative.

Reproduce with `MPLCONFIGDIR=/private/tmp/mplconfig .venv/bin/python -m src.models.evaluate_theta_alpha_candidate`. The validation search, metrics, and per-window test predictions are in `results/diagnostics/anticipatory_theta_alpha_candidate/`.
