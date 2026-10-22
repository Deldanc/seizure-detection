"""Re-evaluate the strongest previously validation-ranked theta/alpha candidate.

The feature pair and threshold are fixed from CHB03_35 validation. CHB03_36
was inspected in earlier runs, so its results are explicitly exploratory.
This script writes a versioned candidate artifact without overwriting v4.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.models.train_anticipatory as training
from src.models.train_anticipatory import (TRAIN, VALIDATION, TEST, fit_candidate,
    get_features, scores_metrics, event_evaluation)
from src.preprocessing.anticipatory_pipeline import FEATURE_PATH, CONFIG_PATH

FEATURES = ["ch14_theta_power", "ch14_alpha_power"]
OUT = ROOT / "results/diagnostics/anticipatory_theta_alpha_candidate"
MODEL_OUT = ROOT / "models/anticipatory_theta_alpha_event_v1.joblib"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(FEATURE_PATH)
    train = frame[frame.file.isin(TRAIN) & frame.eligible.eq(1)].copy()
    validation = frame[frame.file.isin(VALIDATION)].copy()
    test = frame[frame.file.isin(TEST)].copy()
    candidates, operating_points = [], []
    for confirm in (1, 2, 3):
        training.CONFIRM_WINDOWS = confirm
        fit = fit_candidate(train, validation, FEATURES)
        m = fit["validation_metrics"]
        operating_points.append({"confirm_windows": confirm, "threshold": fit["threshold"],
            "metrics": m})
        candidates.append((fit, confirm))
    # Rank alarm policies using validation only: event sensitivity, fewer
    # alarms, window specificity, precision, F1, then higher threshold.
    candidate, confirm = max(candidates, key=lambda pair: (
        pair[0]["validation_metrics"]["event_level"]["event_sensitivity"] or 0,
        -pair[0]["validation_metrics"]["event_level"]["false_alarm_count"],
        pair[0]["validation_metrics"]["specificity"] or 0,
        pair[0]["validation_metrics"]["precision"],
        pair[0]["validation_metrics"]["f1"], pair[0]["threshold"]))
    training.CONFIRM_WINDOWS = confirm
    scores = candidate["model"].predict_proba(candidate["scaler"].transform(
        get_features(test, FEATURES)))[:, 1]
    eligible = test.eligible.to_numpy(bool)
    metrics = scores_metrics(test.loc[eligible, "label"].to_numpy(int),
                             scores[eligible], candidate["threshold"])
    metrics["event_level"] = event_evaluation(test, scores, candidate["threshold"])
    test["risk_score"] = scores
    test["threshold_prediction"] = (scores >= candidate["threshold"]).astype(int)
    alarms = metrics["event_level"]["alarms_generated"]
    test["warning_event"] = test.prediction_time.isin(alarms).astype(int)
    test["warning_active"] = test.prediction_time.map(lambda t: int(any(
        a + 300 <= t < a + 1500 for a in alarms)))
    test.to_csv(OUT / "test_predictions.csv", index=False)
    # Independent seizure-free CHB01 recordings are negative controls only.
    # They do not tune or select this operating point.
    external_source = ROOT / "results/diagnostics/anticipatory_v4/chb01_negative_control/predictions.csv"
    external = pd.read_csv(external_source)
    external_results = []
    external_frames = []
    for filename, part in external.groupby("file", sort=True):
        part = part.copy()
        external_scores = candidate["model"].predict_proba(candidate["scaler"].transform(
            get_features(part, FEATURES)))[:, 1]
        part["risk_score"] = external_scores
        part["threshold_prediction"] = (external_scores >= candidate["threshold"]).astype(int)
        event = event_evaluation(part, external_scores, candidate["threshold"])
        part["warning_event"] = part.prediction_time.isin(event["alarms_generated"]).astype(int)
        external_frames.append(part)
        external_results.append({"file": filename, "annotated_seizures": 0,
            "warnings": event["alarms_generated"], "false_alarm_count": event["false_alarm_count"],
            "recording_hours": event["total_monitoring_hours"],
            "false_alarms_per_recording_hour": event["false_alarms_per_hour"],
            "median_score": float(np.median(external_scores))})
    pd.concat(external_frames, ignore_index=True).to_csv(OUT / "chb01_negative_control_predictions.csv", index=False)
    config = json.loads(CONFIG_PATH.read_text())
    artifact = joblib.load(ROOT / "models/anticipatory_ga_anfis_featureaudit_v4.joblib")
    artifact.update({"model_id": "event-focused theta-alpha Sugeno ANFIS v1",
        "task": "anticipatory risk score", "model": candidate["model"],
        "scaler": candidate["scaler"], "selected_features": FEATURES,
        "n_features_in": len(FEATURES), "threshold": candidate["threshold"],
        "confirm_windows": confirm,
        "selection_method": "theta/alpha pair ranked on CHB03_35 validation; confirmation count and threshold selected by validation event sensitivity, false alarms, specificity, precision and F1",
        "warning": "Exploratory candidate; CHB03_36 was examined in prior project iterations."})
    artifact.pop("ga", None)
    artifact.pop("history", None)
    joblib.dump(artifact, OUT / "model.joblib")
    joblib.dump(artifact, MODEL_OUT)
    result = {"features": FEATURES, "threshold": candidate["threshold"],
        "confirm_windows": confirm, "refractory_seconds": artifact["alarm_refractory_seconds"],
        "sph_seconds": artifact["sph_seconds"],
        "prediction_horizon_seconds": artifact["prediction_horizon_seconds"],
        "threshold_selected_on_validation": True, "validation_operating_point_search": operating_points,
        "validation_metrics": candidate["validation_metrics"],
        "chb03_36_exploratory_metrics": metrics,
        "chb01_seizure_free_negative_controls": external_results,
        "chb01_combined_false_alarms": sum(x["false_alarm_count"] for x in external_results),
        "model_artifact": str(MODEL_OUT.relative_to(ROOT)), "v4_artifact_overwritten": False,
        "limitations": ["Test event was examined in previous project iterations; not an untouched test.",
            "The model uses one-window confirmation; window sensitivity remains very low.",
            "Validation has only one event; the apparent validation AUC is unstable."]}
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
