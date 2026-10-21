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
| Score threshold | selected using the validation recording only by event sensitivity, then fewer full-timeline false alarms, then window specificity, precision, F1, and higher threshold |

These are research settings chosen to exercise an explicit prediction task on the limited local data. They are not clinically validated or recommended settings.

## Window labels

For each seizure onset `T`, the positive decision-time interval is:

`T - (20 min + 5 min) < window_end <= T - 5 min`

Each non-overlapping window is scored at its end timestamp. A positive example's complete EEG window is available by that timestamp, and the positive label is assigned from the onset annotation, not from EEG activity after the window. The next interval, from `T-5 min` up to onset, is the SPH: no window there is labeled positive. Ictal and postictal windows are also never positive.

For one seizure, windows whose decision times are before `T-25 min` are treated as interictal negatives. The entire SPH, ictal interval, and 30 minutes after annotated offset are excluded from supervised window training/evaluation. The feature table retains every window, with `eligible=0` and an empty label for these intervals, so alarms can be evaluated on the full continuous timeline. There is no extra preictal guard. Where the complete 25-minute lead-up is not present, the event's recording history through its postictal exclusion is unlabeled and the event is not a prediction event. In chb03_01, onset is only 362 seconds after file start, so no windows from that recording are used as negative or positive training examples.

Annotations are read from `chb03-summary.txt` as elapsed seconds within the matching EDF. EDF clock gaps and gaps between files are not bridged. Annotation timestamps are never inferred from feature values. For recording with no annotated seizure, its available windows are interictal; this is subject-specific data and not evidence that another patient or unobserved time period is seizure-free.

## Preprocessing and features

The canonical montage has 23 channels; the exact channel mapping is in `data/processed/anticipatory_feature_catalog.csv`. Inference maps channel names and reorders an equivalent channel set into training order. Missing, extra, or ambiguous names are rejected. Inputs below 177.78 Hz are rejected because the causal bandpass uses a 0.45*sfreq high-cutoff margin and must retain the complete 30–80 Hz band. Other supported rates are resampled to 256 Hz using a one-sided causal Kaiser-windowed polyphase FIR. Its delay is retained, so no future samples are used. Filtering then applies a causal fourth-order Butterworth 0.5–100 Hz bandpass and a causal 60 Hz notch. Training and inference use the same order of resampling and filtering. There is no artifact rejection or cross-window normalization.

Schema v3 supplies 345 candidate features: 15 for each of 23 channels. Time-domain features are mean (V), standard deviation (V), variance (V²), RMS (V), and sum-of-squares energy (V² samples). Welch estimates PSD using two-second Hann segments with 50% overlap, constant detrending, density scaling, and 0.5 Hz frequency resolution at 256 Hz. Absolute band powers integrate PSD (V²/Hz) with interpolated exact band edges, yielding V². Relative power is absolute band power divided by integrated 0.5–80 Hz power and is unitless. Bands are delta 0.5–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, and gamma 30–80 Hz. The ranges are contiguous; no candidate band is forced into the selected model. Non-finite samples fail extraction.

## Training and alarms

The fixed split is chb03_34 plus seizure-free local files 15/16/17 for training, chb03_35 for validation, and chb03_36 for test. The GA only sees training rows and evaluates binary feature chromosomes with grouped five-minute temporal blocks, class-weighted random-forest F1 fitness, a feature-count penalty, seed 42, elitism, crossover, mutation and a configurable maximum of four selected inputs. The four-input maximum is a code limit because ANFIS builds 2^d rules, not an experimentally proven optimum. Since only one training seizure event exists, the internal folds are correlated temporal chunks, not independent event folds. Scaler and ANFIS parameters fit on training rows only. Validation compares GA-selected, existing-detector, and theta/alpha subsets and selects thresholds; test is not used for selection.

Three consecutive threshold-crossing scores issue an alarm at the third window end. Alarm generation and event matching use every completed window in the recording, including windows excluded from supervised metrics; otherwise a live alarm during such a window could be missed in evaluation. Window metrics use eligible labels only, while false alarms count all unmatched alarms and are reported per full recording hour (with eligible-hour rate also shown). That alarm's future risk interval is `[alarm_time + SPH, alarm_time + SPH + 20 min]`. An annotated seizure event is matched if its onset lies in this interval. A second alarm is suppressed for 30 minutes. The model emits scores and warning intervals, not a predicted seizure timestamp/countdown. The scores are not calibrated probabilities.
