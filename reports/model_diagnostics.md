# Anticipatory model diagnostics

## Independent reproduction

Command: `MPLCONFIGDIR=/private/tmp/mplconfig .venv/bin/python -m src.models.diagnose_anticipatory`.

The script reloads the v4 artifact, recomputes predictions from the saved scaler/model and processed EEG feature table, and independently reconstructs labels from `chb03-summary.txt`. Test labels agree with the saved prediction CSV for all rows. Maximum difference between saved and recomputed model scores is `1.11e-16`. Independently calculated test confusion matrix is TN=16, FP=33, FN=89, TP=151; accuracy .578, precision .821, sensitivity .629, specificity .327, F1 .712, AP .756, ROC-AUC .373.

The model's `predict_proba` columns are `[class 0, class 1]`, and class 1 is the annotation-derived preictal target. The saved score is column 1. Negating it would yield AUC .627, but code, target, and class mapping do not justify that operation. The low AUC is real on this recording.

## Split composition and score shift

| Split | Eligible negatives | Eligible preictal positives | Positive rate |
|---|---:|---:|---:|
| Train | 2,256 | 240 | 9.6% |
| Validation | 218 | 240 | 52.4% |
| Test | 49 | 240 | 83.0% |

These are not random examples from a prospective monitoring stream. Every split's positive examples surround its one annotated seizure, while eligibility discards the SPH, ictal, postictal, and insufficient-history windows. In the one-hour test EDF, the single 20-minute preictal target interval gives 240 positive windows, while only 49 windows (245 seconds total) remain eligible as negative. The early 225 seconds and the few seconds after the postictal exclusion account for nearly all eligible negatives. Thus 83% prevalence is a consequence of this event-centered exclusion scheme and recording duration, not a plausible estimate of deployment prevalence. The AP baseline is already .830.

The class-conditional scores reverse between training and held-out recordings:

| Split | Class | Mean score | Median score |
|---|---|---:|---:|
| Train | Interictal | .159 | .064 |
| Train | Preictal | .862 | .941 |
| Validation | Interictal | .849 | .931 |
| Validation | Preictal | .560 | .524 |
| Test | Interictal | .802 | .949 |
| Test | Preictal | .754 | .925 |

This points to severe recording/event distribution shift and overfitting to the single training seizure or its recording context. Validation and test preictal score distributions do not outrank interictal scores, consistent with ROC-AUC below .5. Logistic Regression and Random Forest on the same features also have AUC below .5, which makes an ANFIS-only gradient defect or class-index reversal an inadequate explanation.

## Labels, timing, and event protocol

For each seizure at onset `T`, a five-second window ending at time `t` is positive iff `T−1500 < t <= T−300`. The future seizure occurrence period is 20 minutes; the preceding five minutes are the SPH. Ictal and 30-minute postictal windows and incomplete-history windows are excluded. Every window's causal features use samples no later than the window end; regenerated test labels have zero mismatches. No post-onset EEG enters a positive preictal feature window.

In `chb03_36.edf`, onset is 1725 s and offset is 1778 s. The held-out alarm at 35 s covers predicted risk `[335,1535]`; that interval ends 190 s before onset, so the warning is early and unmatched. The 30-minute refractory time after 35 s ends at 1835 s, after onset. The next warning at 1980 s is 255 s after onset and cannot count as anticipation. There was no successful pre-onset warning and no lead-time measurement. We now report that late alert separately from the one prospective false alarm. This clarifies event reporting; it does not improve model performance.

## What was audited and what remains

- GA fitness uses RF window F1 with a feature-count penalty and five-minute temporal block groups. Such folds do not represent independent seizure events. The sole training event means the GA has no event-level diversity for estimating transfer.
- The ANFIS optimization's scalar objective and gradient were already made consistent in the saved v4 implementation; its four-input limit produces 16 rules. This is a computational cap, not an evidence-based optimum. Rerunning the fixed-seed GA reproduced the same four features and the same poor test metrics.
- Threshold and alarm policy were selected on the single validation event only. That is still highly unstable. The test was not used to change either.
- Several feature subsets were compared on the same validation split. Theta/alpha absolute power on channel 14 had validation AUC .842 and F1 .315 versus GA subset AUC .167 and F1 .169; the old detector quartet had AUC .818 and F1 .296. With one validation event and no independent patients, these apparent differences are exploratory and do not justify claiming a better predictor. The GA's grouped-training fitness did not generalize to the validation event.
- New diagnostics are in `results/diagnostics/anticipatory_v4/`: independent metrics JSON, per-split/class score statistics, and scores CSV. Plots are `reports/figures/anticipatory_v4/test_window_diagnostics.png` (ROC, PR, confusion matrix) and `test_event_timeline.png`.

The exact additional data needed is more CHB-MIT recordings with EDF and seizure annotation files for independent seizure events in multiple patients, including sufficient pre-event history and long seizure-free recording time. In particular, obtain the missing annotated CHB03 EDFs 02, 03, and 04 already listed in the local summary, then include other patients for a patient-held-out evaluation. More negative monitoring hours are essential for estimating false alarms per hour under realistic prevalence. A single held-out event cannot establish dependable event sensitivity.
