# Anticipatory EEG feature-selection audit

## Saved detector quartet provenance

The existing seizure-activity detector artifact stores `['ch1_energy', 'ch6_rms', 'ch14_gamma_power', 'ch18_std']` and its GA metadata records a six-generation, population-12, seed-42 search with max four features. These inputs were genuinely emitted by a GA for the older ictal-window detection task. Four was a configured maximum, not a demonstrated optimal count. The anticipatory GA is rerun separately against annotation-derived preictal labels.

## Candidate set and signal processing

Schema v2 has 345 candidates: 15 per channel across 23 saved channels (5 time-domain, 5 absolute band-power, and 5 relative band-power descriptors). Signals are in MNE calibrated volts. A one-sided causal polyphase FIR resamples supported inputs to 256 Hz, then a causal fourth-order Butterworth 0.5–100 Hz filter and causal 60 Hz notch are applied. Features use non-overlapping 5-second windows at 256 Hz. Welch uses the full window, Hann taper, 50% segment overlap, density scaling and 0.2 Hz resolution. Absolute integrated powers are V²; relative powers are unitless fractions of total 0.5–80 Hz power.

| Band | Range | Present features |
|---|---:|---|
| Delta | 0.5–4 Hz | absolute and relative power |
| Theta | 4–8 Hz | absolute and relative power |
| Alpha | 8–13 Hz | absolute and relative power |
| Beta | 13–30 Hz | absolute and relative power |
| Gamma | 30–80 Hz | absolute and relative power |

The band ranges are contiguous and non-overlapping at boundaries. Exact edge PSD values are interpolated before integration. Relative-band denominator is integrated 0.5–80 Hz power; the five relative bands sum to approximately 1 for nonzero signals. No band is forced into the GA subset.

## Channel mapping

Feature prefix `chN` maps to zero-based position N in the canonical channel table below. Inference accepts the same channel set in a different order and reorders by normalized channel name. Missing/extra/ambiguous channels are rejected. Input rates below 177.78 Hz are rejected because the causal 0.5-100 Hz bandpass uses a 0.45*sfreq Nyquist margin; higher supported rates are resampled to 256 Hz with a causal one-sided polyphase FIR and its delay is retained.

| Index | EDF channel |
|---:|---|
| 0 | `FP1-F7` |
| 1 | `F7-T7` |
| 2 | `T7-P7` |
| 3 | `P7-O1` |
| 4 | `FP1-F3` |
| 5 | `F3-C3` |
| 6 | `C3-P3` |
| 7 | `P3-O1` |
| 8 | `FP2-F4` |
| 9 | `F4-C4` |
| 10 | `C4-P4` |
| 11 | `P4-O2` |
| 12 | `FP2-F8` |
| 13 | `F8-T8` |
| 14 | `T8-P8-0` |
| 15 | `P8-O2` |
| 16 | `FZ-CZ` |
| 17 | `CZ-PZ` |
| 18 | `P7-T7` |
| 19 | `T7-FT9` |
| 20 | `FT9-FT10` |
| 21 | `FT10-T8` |
| 22 | `T8-P8-1` |

The complete 345-row feature-name/channel/formula/unit mapping is machine-readable at `data/processed/anticipatory_feature_catalog.csv`.

## GA selection and subset comparisons

The anticipatory GA searched 345 candidates, allowing 1–4 features. Four is the ANFIS implementation limit because it creates 2^d rules; it is not claimed to be biologically or statistically optimal. The selector's fitness uses training-only five-minute grouped cross-validation with RF window F1 and a small feature-count penalty. It selected `['ch5_beta_relative_power', 'ch7_gamma_relative_power', 'ch17_energy', 'ch19_gamma_relative_power']` (training grouped-CV fitness 0.715).

Validation comparison uses the same train-fitted scaler/model approach and same event-aware threshold policy for each subset. Validation is one event and therefore exploratory; the final test was not used.

| Candidate subset | Count | Features | Val threshold | Val precision | Val sensitivity | Val F1 | Val ROC-AUC | Val event hit | Val false alarms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ga_selected | 4 | `ch5_beta_relative_power, ch7_gamma_relative_power, ch17_energy, ch19_gamma_relative_power` | 0.96 | 0.838 | 0.237 | 0.370 | 0.635 | 1/1 | 0 |
| existing_detector_four | 4 | `ch1_energy, ch6_rms, ch14_gamma_power, ch18_std` | 0.86 | 0.910 | 0.671 | 0.772 | 0.816 | 1/1 | 0 |
| theta_alpha_absolute_ch14 | 2 | `ch14_theta_power, ch14_alpha_power` | 0.71 | 0.833 | 0.688 | 0.753 | 0.842 | 1/1 | 0 |
| theta_alpha_relative_ch14 | 2 | `ch14_theta_relative_power, ch14_alpha_relative_power` | 0.70 | 0.255 | 0.050 | 0.084 | 0.315 | 1/1 | 0 |

The existing detector quartet is retrained using anticipatory labels only for this comparison; it is not reused as a trained predictor. Its ch14 gamma feature is recomputed with the corrected 30-80 Hz definition, so this is a feature-set comparison rather than byte-for-byte reproduction of the old detector input. Theta/alpha absolute and relative subsets are diagnostic comparisons, not forced selections. The GA winner is determined by training-only grouped CV, not by the validation ranking.

## Limits

- Only subject chb03 is available locally; EDFs 02, 03 and 04, which include additional annotated seizures, are missing.
- Only one positive seizure event is present in train, validation and test respectively. Recording-independent event selection cannot be estimated reliably.
- Training temporal blocks reduce leakage between nearby windows but are not independent events. A genuinely event-aware GA validation is blocked by too few labeled events.
- The final test recording is used once for the selected GA subset after all comparisons and threshold selection.
- Adding relative power makes the candidate features available; it does not establish that they improve prediction.
