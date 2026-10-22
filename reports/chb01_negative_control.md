# CHB01 seizure-free recording check

## Files and annotation check

The repository now contains `chb01_01.edf`, `chb01_36.edf`, and `chb01-summary.txt`. The CHB01 summary marks both EDFs as having zero seizures. Each EDF is 3,600 seconds at 256 Hz with 23 channels matching the trained CHB03 channel layout. The other seizure-containing recordings listed by the summary are not present locally.

## Fixed-model negative-control result

Command: `MPLCONFIGDIR=/private/tmp/mplconfig .venv/bin/python -m src.models.evaluate_external_negative`.

I scored these recordings with the unchanged v4 artifact and its existing threshold (0.89), selected previously on CHB03 validation. They were excluded from model fitting, feature selection and threshold tuning. Continuous recording windows were processed with the saved feature list and preprocessing. Warnings were generated with the saved three-window confirmation and 30-minute refractory policy.

| Recording | Windows | Median score | 95th percentile | Emitted warnings | Seizures in file |
|---|---:|---:|---:|---:|---:|
| `chb01_01.edf` | 720 | .326 | .767 | 0 | 0 |
| `chb01_36.edf` | 720 | .919 | .992 | 2 (at 15 s and 1850 s) | 0 |

Both warnings in `chb01_36.edf` have their full 25-minute prediction intervals observable before the recording ends, and neither interval contains an annotated event. Across the two recordings this is **2 unmatched warnings in 1.167 hours with the complete future interval observable**, or **1.71 warnings per observable hour**. No seizure sensitivity, ROC-AUC, or positive-window metric can be computed from these seizure-free recordings.

## Interpretation and next data

These recordings are useful: they provide a small independent-patient false-warning check, and they reveal high scores and two warnings in one seizure-free file. They do not improve or validate seizure sensitivity. The marked difference between `chb01_01` and `chb01_36` also reinforces that recording conditions vary substantially within a patient.

## Exploratory negative-data augmentation

I ran a separate, non-deployed retrain: `chb01_01` was added as 720 interictal training windows, the GA was rerun, the threshold was selected on the same CHB03 validation recording, and `chb01_36` remained outside training and threshold selection. The active v4 artifact was not replaced. The resulting features were `ch7_gamma_relative_power`, `ch8_std`, `ch17_rms`, and `ch19_theta_relative_power`; the validation-selected threshold was 0.96.

| Result | Existing v4 | CHB01-negative-augmented candidate |
|---|---:|---:|
| CHB03_35 validation ROC-AUC | .167 | .357 |
| CHB03_35 validation F1 | .169 | .203 |
| CHB03_36 historical test ROC-AUC | .373 | .461 |
| CHB03_36 historical test specificity | .327 | .592 |
| CHB03_36 historical test sensitivity | .629 | .350 |
| CHB03_36 event hits | 0/1 | 0/1 |
| CHB01_36 seizure-free file warnings | 2 | 1 |

The candidate reduces sensitivity and F1 while improving specificity and the AUC estimates; it still misses the seizure and remains below chance AUC on the historical CHB03 test. The previously inspected CHB03 test is not an untouched test for this new experiment, and the CHB01 holdout is a single recording from the same patient whose other recording was used for negative training. These results do **not** establish improvement or patient-independent prediction. They show the added interictal recording changes the learned subset and operating point, but one event is far too little to know whether that helps.

Reproduction command: `MPLCONFIGDIR=/private/tmp/mplconfig .venv/bin/python -m src.models.train_anticipatory_with_chb01_negative`. Versioned candidate model and metrics are in `results/diagnostics/anticipatory_chb01_augmented/` and do not replace the active detector.

Do not add `chb01_36` to training if keeping this external warning check. `chb01_01` could later augment interictal training, but doing so requires a fresh model version and must be judged on independent seizure events. To test anticipatory sensitivity for CHB01, obtain the seizure-containing EDFs listed in `chb01-summary.txt` (currently `chb01_03`, `_04`, `_15`, `_16`, `_18`, `_21`, and `_26`) with their annotation files. Preferably add multiple other patients and long seizure-free periods, then split by patient and event. Two negative files cannot demonstrate that the model predicts seizures.

Raw EEG is not duplicated in this report; per-window outputs and exact scores are in `results/diagnostics/anticipatory_v4/chb01_negative_control/`.
