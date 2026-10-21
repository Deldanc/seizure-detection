"""Inference for the separate future-risk GA–ANFIS model."""
from pathlib import Path
import uuid

import joblib
import mne
import numpy as np

from src.preprocessing.anticipatory_pipeline import (causal_filter, feature_row,
    causal_resample,
    _channel_indices, FEATURE_SCHEMA_VERSION, WINDOW_SEC,
    PREDICTION_HORIZON_SEC, SPH_SEC)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "anticipatory_ga_anfis_featureaudit_v4.joblib"


def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Anticipatory GA–ANFIS feature audit model is not trained. Run data preparation and then src/models/train_anticipatory.py.")
    artifact = joblib.load(MODEL_PATH)
    required = {"model", "scaler", "selected_features", "threshold", "channel_names",
                "feature_names", "feature_schema_version", "sampling_frequency_hz",
                "prediction_horizon_seconds", "sph_seconds"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("Anticipatory model artifact is incomplete or incompatible.")
    if artifact["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("Anticipatory artifact feature schema is incompatible; retrain with the current pipeline.")
    if len(set(artifact["selected_features"])) != len(artifact["selected_features"]):
        raise ValueError("Anticipatory artifact contains duplicate selected features.")
    if not set(artifact["selected_features"]).issubset(set(artifact["feature_names"])):
        raise ValueError("Anticipatory selected feature list contains unknown inference features.")
    if len(artifact["selected_features"]) != artifact["model"].n_features_in_ or len(artifact["selected_features"]) != artifact["scaler"].n_features_in_ or len(artifact["selected_features"]) != artifact.get("n_features_in"):
        raise ValueError("Anticipatory model feature metadata dimensions do not match.")
    return artifact


def _anticipatory_channel_indices(actual_names, expected_names):
    """Resolve named CHB03 montage channels or the project's sequential EEG aliases.

    The bundled sample uses EEG01..EEG23 labels in the canonical CHB03 channel
    order. Preserve the same positional alias already accepted by the detector;
    arbitrary or partial channel sets still fail strict montage validation.
    """
    normalized = [name.strip().upper() for name in actual_names]
    sequential_aliases = [f"EEG{i:02d}" for i in range(1, len(expected_names) + 1)]
    if normalized == sequential_aliases:
        return list(range(len(expected_names)))
    return _channel_indices(list(actual_names), list(expected_names))


def validate_anticipatory_edf(edf_path):
    artifact = load_artifact()
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    try:
        channel_indices = _anticipatory_channel_indices(list(raw.ch_names), artifact["channel_names"])
        input_sfreq = float(raw.info["sfreq"])
        if input_sfreq * 0.45 <= 80:
            raise ValueError("EDF sampling rate must be at least 177.78 Hz to retain the trained 30-80 Hz gamma band through filtering.")
        if raw.n_times < int(round(artifact.get("window_seconds", WINDOW_SEC) * raw.info["sfreq"])):
            raise ValueError("Recording is shorter than one analysis window.")
        return {"channels": len(raw.ch_names), "sampling_frequency": float(raw.info["sfreq"]),
                "channel_indices": channel_indices,
                "duration_sec": raw.n_times / float(raw.info["sfreq"]), "compatible": True}
    finally:
        raw.close()


def generate_alarms(times, scores, threshold, confirm_windows=3, refractory_sec=1800, step_seconds=5):
    alarms, streak, next_allowed = [], 0, -np.inf
    previous = None
    for t, score in zip(times, scores):
        if previous is not None and t - previous > step_seconds * 1.5:
            streak = 0
        streak = streak + 1 if score >= threshold else 0
        if streak >= confirm_windows and t >= next_allowed:
            alarms.append(float(t))
            next_allowed = float(t) + refractory_sec
            streak = 0
        previous = t
    return alarms


def predict_anticipatory_edf(edf_path, threshold=None):
    artifact = load_artifact()
    validated_threshold = float(artifact["threshold"])
    threshold_source = "validation-selected artifact threshold"
    if threshold is not None:
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("Risk threshold must be between 0 and 1.")
        threshold_source = "user override (not validation-verified)" if abs(threshold - validated_threshold) > 1e-12 else threshold_source
    else:
        threshold = validated_threshold
    info = validate_anticipatory_edf(edf_path)
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    try:
        sf = float(raw.info["sfreq"])
        resampled = causal_resample(raw.get_data(picks=info["channel_indices"]), sf,
                                    float(artifact["sampling_frequency_hz"]))
        sf = float(artifact["sampling_frequency_hz"])
        filtered = causal_filter(resampled, sf)
        window_sec = float(artifact.get("window_seconds", WINDOW_SEC))
        step_sec = float(artifact.get("step_seconds", window_sec))
        win = int(round(window_sec * sf))
        step_samples = int(round(step_sec * sf))
        starts = range(0, filtered.shape[1] - win + 1, step_samples)
        rows = []
        selected = artifact["selected_features"]
        for start_sample in starts:
            start = start_sample / sf
            end = start + window_sec
            feats = feature_row(filtered[:, start_sample:start_sample + win], sf)
            rows.append({"start": start, "end": end, "prediction_time": end,
                         "features": [feats[x] for x in selected]})
    finally:
        raw.close()
    if not rows:
        raise ValueError("No complete EEG windows were available for inference.")
    X = np.asarray([r["features"] for r in rows], dtype=float)
    scores = artifact["model"].predict_proba(artifact["scaler"].transform(X))[:, 1]
    alarms = generate_alarms([r["prediction_time"] for r in rows], scores, threshold,
        artifact.get("confirm_windows", 3), artifact.get("alarm_refractory_seconds", 1800),
        artifact.get("step_seconds", artifact.get("window_seconds", WINDOW_SEC)))
    windows = []
    for r, score in zip(rows, scores):
        now = r["prediction_time"]
        sph = artifact.get("sph_seconds", SPH_SEC)
        horizon = artifact.get("prediction_horizon_seconds", PREDICTION_HORIZON_SEC)
        warning = any(t + sph <= now < t + sph + horizon for t in alarms)
        generated = any(abs(t - now) < 1e-7 for t in alarms)
        warning_event = next((t for t in alarms if abs(t - now) < 1e-7), None)
        windows.append({"start": r["start"], "end": r["end"], "prediction_time": now,
                        "risk_score": float(score), "prediction": int(score >= threshold),
                        "warning_active": int(warning), "warning_event": int(generated),
                        "warning_generated_at": warning_event,
                        "future_risk_start": warning_event + sph if warning_event is not None else None,
                        "future_risk_end": warning_event + sph + horizon if warning_event is not None else None})
    warning_events = [{"generated_at": t, "risk_period_start": t + artifact.get("sph_seconds", SPH_SEC),
                       "risk_period_end": t + artifact.get("sph_seconds", SPH_SEC) + artifact.get("prediction_horizon_seconds", PREDICTION_HORIZON_SEC)}
                      for t in alarms]
    return {"job_id": uuid.uuid4().hex, "filename": Path(edf_path).name,
        "model_id": artifact.get("model_id", "anticipatory GA–ANFIS"),
        "score_type": "uncalibrated future-seizure risk score",
        "threshold": threshold, "validated_threshold": validated_threshold,
        "threshold_source": threshold_source,
        "prediction_horizon_seconds": artifact.get("prediction_horizon_seconds", PREDICTION_HORIZON_SEC),
        "sph_seconds": artifact.get("sph_seconds", SPH_SEC), "window_sec": artifact.get("window_seconds", WINDOW_SEC),
        "step_sec": artifact.get("step_seconds", artifact.get("window_seconds", WINDOW_SEC)),
        "windows_analyzed": len(windows), "peak_score": float(np.max(scores)),
        "warning_events": warning_events, "warning_count": len(warning_events),
        "warning_active_at_end": bool(windows[-1]["warning_active"]) if windows else False,
        "duration_sec": info["duration_sec"], "channels": info["channels"],
        "sampling_frequency": info["sampling_frequency"],
        "alarm_refractory_seconds": artifact.get("alarm_refractory_seconds", 1800),
        "windows": windows}
