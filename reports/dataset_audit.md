# CHB-MIT local dataset audit

Audit date: 2026-10-22. The official [CHB-MIT Scalp EEG Database 1.0.0 page](https://physionet.org/content/chbmit/1.0.0/) describes 256 samples/second, 23 EEG channels in most files, per-record `.seizure` annotations, and `chbnn-summary.txt` files that give elapsed seconds from each EDF start to seizure onsets/offsets. The local timestamps below come from the supplied `chb03-summary.txt`; all are interpreted relative to the named EDF's own start, never across file boundaries.

## Files actually present

| Recording | Patient | Duration | Rate | Channels | Summary annotations (s) | Processing |
|---|---|---:|---:|---:|---|---|
| chb03_01.edf | chb03 | 3600 s | 256 Hz | 23 | 362–414 | Processed; only 277 eligible windows because seizure occurs too early for the complete warning interval and seizure-associated windows are excluded |
| chb03_15.edf | chb03 | 3600 s | 256 Hz | 23 | none | Processed; 720 eligible windows |
| chb03_16.edf | chb03 | 3600 s | 256 Hz | 23 | none | Processed; 720 eligible windows |
| chb03_17.edf | chb03 | 3600 s | 256 Hz | 23 | none | Processed; 720 eligible windows |
| chb03_34.edf | chb03 | 3600 s | 256 Hz | 23 | 1982–2029 | Processed; 336 eligible windows, 240 positive |
| chb03_35.edf | chb03 | 3600 s | 256 Hz | 23 | 2592–2656 | Processed; 458 eligible windows, 240 positive |
| chb03_36.edf | chb03 | 3600 s | 256 Hz | 23 | 1725–1778 | Processed; 289 eligible windows, 240 positive |

The machine-readable inventory, including actual MNE channel names/order, is `data/processed/dataset_manifest.csv`. EDFs opened successfully. All channels are used; missing or non-finite feature windows raise an error rather than being silently imputed. Two repeated montage labels are reported by MNE as `T8-P8-0` and `T8-P8-1`, preserving their original positions. The complete feature-to-channel mapping is in `data/processed/anticipatory_feature_catalog.csv`.

## Annotations, mapping, and missing data

The local summary contains onset/offset entries for chb03_01–04 and chb03_34–36. Only EDFs 01, 34, 35, and 36 from those annotated records are local. The annotated files 02, 03, and 04 are missing; so are all other patients' EDFs. The sidecar `chb03_01.edf.seizures` is present, but the implemented parser uses the matching human-readable summary entry; it does not attempt to decode the binary sidecar format. Summary timestamps mapped to EDF-relative elapsed seconds, consistent with PhysioNet's description.

The summary includes clock start/end times and shows acquisition gaps and time jumps. The labels deliberately use within-file elapsed seconds only, so a gap between files is never treated as continuous EEG. One local seizure (file 01 at 362 s) has less than the configured 25-minute pre-onset period in its file. It is not relabeled as a complete preictal example. It contributes no preictal positive window.

## Processed data provenance

The old `data/processed/features.csv` (5,033 rows, 230 features) uses the earlier seizure-presence label and is not suitable as anticipation labels. The anticipatory table was regenerated from seven raw EDFs under schema v3. It retains all 5,040 five-second windows, with 3,520 eligible labels (720 preictal and 2,800 interictal); SPH, ictal, postictal, and incomplete-history windows are unlabeled and marked ineligible. Keeping these rows allows event alarms to be evaluated over the full continuous recording. Each window has 345 candidate features (15 per channel): five time statistics, five absolute band powers, and five relative band powers. MNE calibrated values are in volts; absolute powers are in V² and relative powers are dimensionless. No downloaded or synthetic research data was substituted.

The v3 frequency ranges are delta 0.5–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, and gamma 30–80 Hz. Welch uses two-second Hann segments with 50% segment overlap, 0.5 Hz resolution, and interpolated exact band edges. Relative power is normalized by total 0.5–80 Hz power. The earlier anticipatory extractor instead used gamma 30–100 Hz, omitted relative powers, and integrated only sampled frequency bins; its dataset and model were preserved in `archive/anticipatory_pre_feature_audit_20261022/`.
