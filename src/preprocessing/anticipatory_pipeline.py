"""Causal, recording-scoped features and annotation-based prediction labels."""
from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import butter, firwin, iirnotch, sosfilt, tf2sos, upfirdn, welch

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
FEATURE_PATH = ROOT / "data" / "processed" / "anticipatory_features.csv"
MANIFEST_PATH = ROOT / "data" / "processed" / "dataset_manifest.csv"
CATALOG_PATH = ROOT / "data" / "processed" / "anticipatory_feature_catalog.csv"
CONFIG_PATH = ROOT / "data" / "processed" / "anticipatory_config.json"
WINDOW_SEC = 5.0
STEP_SEC = 5.0
TARGET_SFREQ = 256.0
PREDICTION_HORIZON_SEC = 20 * 60
SPH_SEC = 5 * 60
POSTICTAL_EXCLUSION_SEC = 30 * 60
INTERICTAL_GUARD_SEC = 0
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 80.0)}
FEATURE_SCHEMA_VERSION = 3


def parse_summary(summary_path: Path = RAW_DIR / "chb03-summary.txt") -> dict[str, list[tuple[float, float]]]:
    """Parse per-file onset/offset seconds from the locally supplied summary."""
    text = summary_path.read_text(errors="replace")
    result: dict[str, list[tuple[float, float]]] = {}
    current = None
    onset = None
    for line in text.splitlines():
        name = re.search(r"File Name:\s*(\S+\.edf)", line, re.I)
        if name:
            current = name.group(1).lower()
            result.setdefault(current, [])
        start = re.search(r"Seizure Start Time:\s*([0-9.]+)\s*seconds", line, re.I)
        end = re.search(r"Seizure End Time:\s*([0-9.]+)\s*seconds", line, re.I)
        if start:
            onset = float(start.group(1))
        if end and current and onset is not None:
            result[current].append((onset, float(end.group(1))))
            onset = None
    return result


def prediction_labels(start: float, end: float, seizures: list[tuple[float, float]],
                      horizon: float = PREDICTION_HORIZON_SEC,
                      sph: float = SPH_SEC,
                      postictal: float = POSTICTAL_EXCLUSION_SEC,
                      guard: float = INTERICTAL_GUARD_SEC) -> int | None:
    """Label by decision time; preictal windows end before the SPH starts.

    A complete preictal interval is ``(onset - horizon - SPH, onset - SPH]``.
    SPH, ictal, postictal and optional guard windows are excluded, never positive.
    """
    positives = []
    exclusion_zones = []
    for onset, offset in seizures:
        complete_history = onset >= horizon + sph
        positive_start = onset - sph - horizon
        positive_end = onset - sph
        if complete_history and positive_start < end <= positive_end:
            positives.append(onset)
        exclusion_start = max(0.0, positive_start - guard) if complete_history else 0.0
        exclusion_zones.append((onset, exclusion_start, offset + postictal))
    # A window can be positive for one event only if it is not ambiguous with
    # another event's SPH/ictal/postictal zone or an incomplete event history.
    for onset, zone_start, zone_end in exclusion_zones:
        if onset not in positives and start < zone_end and end > zone_start:
            return None
    if positives:
        return 1
    return 0


def causal_filter(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Causal 0.5-100 Hz Butterworth and causal 60 Hz notch filtering."""
    if sfreq / 2 <= 80:
        raise ValueError(f"Sampling rate {sfreq:g} Hz cannot represent the 30-80 Hz gamma band.")
    high = min(100.0, sfreq * 0.45)
    if high <= 1:
        raise ValueError(f"Sampling frequency {sfreq:g} Hz is too low for EEG features.")
    sos = butter(4, [0.5, high], btype="bandpass", fs=sfreq, output="sos")
    filtered = sosfilt(sos, np.asarray(data, dtype=float), axis=-1)
    if sfreq / 2 > 60:
        b, a = iirnotch(60.0, 30.0, fs=sfreq)
        filtered = sosfilt(tf2sos(b, a), filtered, axis=-1)
    return filtered


def causal_resample(data: np.ndarray, sfreq: float,
                    target_sfreq: float = TARGET_SFREQ) -> np.ndarray:
    """Causally resample with a one-sided polyphase FIR; output retains delay."""
    data = np.asarray(data, dtype=float)
    if np.isclose(sfreq, target_sfreq, atol=1e-8):
        return data
    if sfreq * 0.45 <= 80:
        raise ValueError("Input sampling rate must be at least 177.78 Hz to retain the 30-80 Hz band through filtering.")
    ratio = Fraction(target_sfreq / sfreq).limit_denominator(1000)
    represented = sfreq * ratio.numerator / ratio.denominator
    if abs(represented - target_sfreq) > target_sfreq * 1e-3:
        raise ValueError(f"Could not accurately resample {sfreq:g} Hz EEG to {target_sfreq:g} Hz.")
    up, down = ratio.numerator, ratio.denominator
    taps = 10 * max(up, down) + 1
    kernel = firwin(taps, 1.0 / max(up, down), window=("kaiser", 5.0)) * up
    result = upfirdn(kernel, data, up=up, down=down, axis=-1)
    output_samples = int(np.ceil(data.shape[-1] * up / down))
    return result[..., :output_samples]


def _integrated_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    """Integrate a PSD over exact band edges using interpolated endpoints."""
    if low < freqs[0] or high > freqs[-1]:
        raise ValueError(f"Requested {low:g}-{high:g} Hz band exceeds the PSD frequency range.")
    inside = (freqs > low) & (freqs < high)
    x = np.concatenate(([low], freqs[inside], [high]))
    y = np.interp(x, freqs, psd)
    return float(trapezoid(y, x))


def feature_row(window: np.ndarray, sfreq: float) -> dict[str, float]:
    """Extract 15 descriptors/channel from calibrated MNE signals in volts.

    Welch PSD uses two-second Hann segments (0.5 Hz bins at 256 Hz) with
    50% overlap inside the five-second window. Absolute power integrates PSD in V^2/Hz
    over exact boundaries to produce V^2; relative power is dimensionless and
    normalized by the 0.5-80 Hz total power.
    """
    window = np.asarray(window, dtype=float)
    if window.ndim != 2:
        raise ValueError("EEG window must have shape (channels, samples).")
    if sfreq / 2 <= max(hi for _, hi in BANDS.values()):
        raise ValueError("Sampling rate must exceed 160 Hz to calculate the 30-80 Hz band.")
    out: dict[str, float] = {}
    nperseg = min(int(round(2.0 * sfreq)), window.shape[1])
    if nperseg < 8:
        raise ValueError("EEG feature window is too short for Welch power estimation.")
    for ch, signal in enumerate(window):
        signal = np.asarray(signal, dtype=float)
        if not np.isfinite(signal).all():
            raise ValueError("Window contains non-finite EEG samples.")
        prefix = f"ch{ch}"
        out[f"{prefix}_mean"] = float(np.mean(signal))
        out[f"{prefix}_std"] = float(np.std(signal))
        out[f"{prefix}_variance"] = float(np.var(signal))
        out[f"{prefix}_rms"] = float(np.sqrt(np.mean(signal ** 2)))
        out[f"{prefix}_energy"] = float(np.sum(signal ** 2))
        freqs, psd = welch(signal, fs=sfreq, window="hann", nperseg=nperseg,
                           noverlap=nperseg // 2, detrend="constant",
                           scaling="density", average="mean")
        total_power = _integrated_power(freqs, psd, 0.5, 80.0)
        for band, (low, high) in BANDS.items():
            power = _integrated_power(freqs, psd, low, high)
            out[f"{prefix}_{band}_power"] = power
            out[f"{prefix}_{band}_relative_power"] = power / total_power if total_power > 0 else 0.0
    return out


def feature_catalog(channel_names: list[str]) -> pd.DataFrame:
    """Return the complete feature-to-channel and unit mapping."""
    time_defs = [("mean", "mean(signal)", "V"),
        ("std", "population standard deviation", "V"),
        ("variance", "population variance", "V^2"),
        ("rms", "sqrt(mean(signal^2))", "V"),
        ("energy", "sum(signal^2) over window samples", "V^2 samples")]
    records = []
    for ch, channel in enumerate(channel_names):
        for suffix, formula, unit in time_defs:
            records.append({"feature": f"ch{ch}_{suffix}", "channel_index": ch,
                "channel_name": channel, "family": "time domain", "frequency_range_hz": "",
                "formula": formula, "unit": unit})
        for band, (low, high) in BANDS.items():
            records.append({"feature": f"ch{ch}_{band}_power", "channel_index": ch,
                "channel_name": channel, "family": "absolute band power",
                "frequency_range_hz": f"{low:g}-{high:g}",
                "formula": f"integral Welch PSD from {low:g} to {high:g} Hz; interpolated edge bins",
                "unit": "V^2"})
            records.append({"feature": f"ch{ch}_{band}_relative_power", "channel_index": ch,
                "channel_name": channel, "family": "relative band power",
                "frequency_range_hz": f"{low:g}-{high:g}",
                "formula": f"absolute {band} power / total 0.5-80 Hz power",
                "unit": "unitless fraction"})
    return pd.DataFrame(records)


def _channel_indices(actual: list[str], expected: list[str]) -> list[int]:
    normalize = lambda name: re.sub(r"\s+", "", name).upper()
    actual_norm = [normalize(name) for name in actual]
    expected_norm = [normalize(name) for name in expected]
    if len(set(actual_norm)) != len(actual_norm) or len(set(expected_norm)) != len(expected_norm):
        raise ValueError("EDF channel names are ambiguous after normalization.")
    if set(actual_norm) != set(expected_norm):
        missing = sorted(set(expected_norm) - set(actual_norm))
        extra = sorted(set(actual_norm) - set(expected_norm))
        raise ValueError(f"EDF channel layout mismatch (missing={missing}, extra={extra}).")
    return [actual_norm.index(name) for name in expected_norm]


def extract_recording(edf_path: Path, seizures: list[tuple[float, float]],
                      window_sec: float = WINDOW_SEC, step_sec: float = STEP_SEC,
                      horizon_sec: float = PREDICTION_HORIZON_SEC,
                      sph_sec: float = SPH_SEC,
                      postictal_sec: float = POSTICTAL_EXCLUSION_SEC,
                      guard_sec: float = INTERICTAL_GUARD_SEC,
                      channel_order: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    try:
        sfreq = float(raw.info["sfreq"])
        actual_names = list(raw.ch_names)
        ordered_names = channel_order or actual_names
        indices = _channel_indices(actual_names, ordered_names)
        if sfreq * 0.45 <= 80:
            raise ValueError(f"Input sampling rate {sfreq:g} Hz is too low to retain the complete 30-80 Hz gamma band through filtering.")
        data = causal_resample(raw.get_data(picks=indices), sfreq, TARGET_SFREQ)
        data = causal_filter(data, TARGET_SFREQ)
        window_samples = round(window_sec * TARGET_SFREQ)
        step_samples = round(step_sec * TARGET_SFREQ)
        rows = []
        for start_sample in range(0, data.shape[1] - window_samples + 1, step_samples):
            end_sample = start_sample + window_samples
            start, end = start_sample / TARGET_SFREQ, end_sample / TARGET_SFREQ
            label = prediction_labels(start, end, seizures, horizon_sec, sph_sec,
                                      postictal_sec, guard_sec)
            row = feature_row(data[:, start_sample:end_sample], TARGET_SFREQ)
            row.update(file=edf_path.name.lower(), start=start, end=end,
                       prediction_time=end, label=np.nan if label is None else label,
                       eligible=int(label is not None))
            rows.append(row)
        return pd.DataFrame(rows), {"duration_sec": raw.n_times / sfreq,
            "sampling_frequency": sfreq, "channel_count": len(actual_names),
            "channel_names": ordered_names, "original_channel_names": actual_names,
            "channel_reordered": actual_names != ordered_names,
            "processing_status": "processed", "total_windows": len(rows),
            "eligible_windows": int(sum(row["eligible"] for row in rows))}
    finally:
        raw.close()


def build_dataset(raw_dir: Path = RAW_DIR, feature_path: Path = FEATURE_PATH,
                  manifest_path: Path = MANIFEST_PATH, window_sec: float = WINDOW_SEC,
                  step_sec: float = STEP_SEC,
                  horizon_sec: float = PREDICTION_HORIZON_SEC,
                  sph_sec: float = SPH_SEC,
                  postictal_sec: float = POSTICTAL_EXCLUSION_SEC,
                  guard_sec: float = INTERICTAL_GUARD_SEC) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations = parse_summary(raw_dir / "chb03-summary.txt")
    edfs = sorted(raw_dir.glob("*.edf"))
    all_rows, manifest = [], []
    canonical_channels = None
    for path in edfs:
        frame, info = extract_recording(path, annotations.get(path.name.lower(), []),
            window_sec, step_sec, horizon_sec, sph_sec, postictal_sec, guard_sec,
            channel_order=canonical_channels)
        if canonical_channels is None:
            canonical_channels = info["channel_names"]
        all_rows.append(frame)
        info.update(file=path.name.lower(), patient="chb03",
                    seizures=annotations.get(path.name.lower(), []))
        manifest.append(info)
        eligible = frame[frame.eligible == 1]
        print(f"{path.name}: {len(frame)} total windows, {len(eligible)} eligible, {int(eligible.label.sum()) if len(eligible) else 0} preictal")
    frame = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(feature_path, index=False)
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    catalog = feature_catalog(canonical_channels or [])
    catalog.to_csv(CATALOG_PATH, index=False)
    config = {"feature_schema_version": FEATURE_SCHEMA_VERSION,
        "window_seconds": window_sec, "step_seconds": step_sec,
        "target_sampling_frequency_hz": TARGET_SFREQ,
        "minimum_input_sampling_frequency_hz": 80.0 / 0.45,
        "prediction_horizon_seconds": horizon_sec, "sph_seconds": sph_sec,
        "postictal_exclusion_seconds": postictal_sec, "interictal_guard_seconds": guard_sec,
        "channel_names": canonical_channels or [], "bands_hz": BANDS,
        "relative_power_denominator_hz": [0.5, 80.0],
        "welch": {"window": "hann", "nperseg": min(round(2.0 * TARGET_SFREQ), round(window_sec * TARGET_SFREQ)),
            "noverlap": min(round(2.0 * TARGET_SFREQ), round(window_sec * TARGET_SFREQ)) // 2,
            "detrend": "constant", "scaling": "density", "average": "mean"},
        "resampling": {"method": "one-sided causal Kaiser-windowed polyphase FIR (upfirdn)",
            "target_hz": TARGET_SFREQ, "minimum_input_hz": 80.0 / 0.45,
            "rational_ratio_max_denominator": 1000,
            "fir_taps": "10*max(up,down)+1", "kaiser_beta": 5.0,
            "delay": "preserved in output timeline; no future samples used"},
        "filtering": {"bandpass": "causal Butterworth", "order": 4,
            "passband_hz": [0.5, 100.0], "notch_hz": 60.0, "notch_q": 30.0},
        "preprocessing": "causal polyphase resampling to 256 Hz when needed, then causal fourth-order Butterworth 0.5-100 Hz and causal 60 Hz notch; channels reordered by name; mismatched channel sets rejected"}
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")
    return frame, pd.DataFrame(manifest)
