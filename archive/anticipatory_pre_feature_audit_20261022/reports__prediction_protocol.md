# Anticipatory prediction protocol

## Intended task

For a seizure with annotated onset `T` in a specific EDF, the model scores each completed EEG window for an upcoming seizure whose onset is expected in a future occurrence period. It does not detect current seizure activity and it does not predict an exact onset time.

Default experimental settings:

| Parameter | Value |
|---|---:|
| Analysis window | 5 seconds |
| Step/overlap | 5 seconds / no overlap |
| Seizure occurrence period (called prediction horizon in the code/UI) | 20 minutes |
| Seizure prediction horizon (SPH) | 5 minutes |
| Postictal exclusion | 30 minutes after annotated offset |
| Alarm confirmation | 3 successive 5-second scores at/above threshold |
| Alarm refractory period | 30 minutes after an alarm |
| Score threshold | selected using the validation recording only by event sensitivity, then fewer false alarms, then window F1 |

These are research settings chosen to exercise an explicit prediction task on the limited local data. They are not clinically validated or recommended settings.

## Window labels

For each seizure onset `T`, the positive decision-time interval is:

`T - (20 min + 5 min) < window_end <= T - 5 min`

Each non-overlapping window is scored at its end timestamp. A positive example's complete EEG window is available by that timestamp, and the positive label is assigned from the onset annotation, not from EEG activity after the window. The next interval, from `T-5 min` up to onset, is the SPH: no window there is labeled positive. Ictal and postictal windows are also never positive.

For one seizure, windows whose decision times are from the beginning of the EDF to before `T-25 min` are treated as interictal negatives. The entire SPH, ictal interval, and 30 minutes after annotated offset are excluded from training/evaluation. The policy uses the defined 25-minute prediction interval as the separation boundary; there is no extra preictal guard. Where the complete 25-minute lead-up is not present in a recording, that event has no positive examples. In chb03_01, onset is only 362 seconds after file start, so this seizure is not used as a prediction event.

Annotations are read from `chb03-summary.txt` as elapsed seconds within the matching EDF. EDF clock gaps and gaps between files are not bridged. Annotation timestamps are never inferred from feature values. For recording with no annotated seizure, its available windows are interictal; this is subject-specific data and not evidence that another patient or unobserved time period is seizure-free.

## Preprocessing and features

MNE loads each EDF at its native 256 Hz rate. The input must have the training 23-channel order (including MNE's disambiguated repeated labels) and 256 Hz sampling. Each channel is filtered with causal fourth-order Butterworth 0.5–100 Hz bandpass followed by a causal 60 Hz notch. Filtering is applied from file start with causal state, so an output prefix is unchanged if later samples are appended. No zero-phase whole-record filter, cross-window normalization, resampling, or artifact rejection is used.

For each channel/window, the pipeline computes mean, standard deviation, variance, RMS, sum-of-squares energy, and Welch integrated absolute power in delta (0.5–4 Hz), theta (4–8), alpha (8–13), beta (13–30), and gamma (30–100). There are 230 features for 23 channels. Signal values are in MNE calibrated volts; power and energy are in voltage-squared units (energy also depends on sample count). Welch uses SciPy defaults, including a 256-sample segment for these 5-second windows. Non-finite signal samples fail processing instead of becoming artificial feature values.

## Training and alarms

The fixed split is chb03_34 plus seizure-free local files 15/16/17 for training, chb03_35 for validation, and chb03_36 for test. The GA only sees training rows and evaluates binary feature chromosomes with grouped five-minute temporal blocks, class-weighted random-forest F1 fitness, a feature-count penalty, seed 42, elitism, crossover, mutation and a maximum of four selected features. Because there is only one training seizure event, these internal folds are correlated temporal chunks, not independent event folds. Scaler and ANFIS parameters fit on training rows only. Validation selects the alarm score threshold; test is not used for selection.

Three consecutive threshold-crossing scores issue an alarm at the third window end. That alarm's future risk interval is `[alarm_time + SPH, alarm_time + SPH + 20 min]`. An annotated seizure event is matched if its onset lies in this interval. A second alarm is suppressed for 30 minutes. The model emits scores and warning intervals, not a predicted seizure timestamp/countdown. The scores are not calibrated probabilities.
