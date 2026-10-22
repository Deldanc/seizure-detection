"""Score unseen seizure-free CHB01 recordings with the saved v4 model.

This is a negative-control transfer check only. It must not be used to tune the
model, threshold, or features because these recordings contain no seizures.
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
from src.models.train_anticipatory import generate_alarms
from src.preprocessing.anticipatory_pipeline import RAW_DIR, parse_summary, extract_recording

ARTIFACT_PATH = ROOT / "models/anticipatory_ga_anfis_featureaudit_v4.joblib"
OUT_DIR = ROOT / "results/diagnostics/anticipatory_v4/chb01_negative_control"
FILES = ["chb01_01.edf", "chb01_36.edf"]


def main():
    artifact = joblib.load(ARTIFACT_PATH)
    annotations = parse_summary(RAW_DIR / "chb01-summary.txt")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"purpose": "external patient seizure-free negative-control check",
        "model": str(ARTIFACT_PATH.relative_to(ROOT)), "threshold": artifact["threshold"],
        "threshold_source": "saved threshold selected on CHB03 validation only",
        "feature_or_threshold_tuning_on_chb01": False, "recordings": []}
    all_frames = []
    for filename in FILES:
        path = RAW_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        events = annotations.get(filename.lower(), [])
        if events:
            raise ValueError(f"{filename} is not a seizure-free negative control: {events}")
        frame, meta = extract_recording(path, events, window_sec=artifact["window_seconds"],
            step_sec=artifact["step_seconds"], horizon_sec=artifact["prediction_horizon_seconds"],
            sph_sec=artifact["sph_seconds"], postictal_sec=artifact["postictal_exclusion_seconds"],
            guard_sec=artifact["interictal_guard_seconds"], channel_order=artifact["channel_names"])
        X = frame[artifact["selected_features"]].to_numpy(float)
        score = artifact["model"].predict_proba(artifact["scaler"].transform(X))[:, 1]
        times = frame.prediction_time.to_numpy(float)
        alarms = generate_alarms(times, score, artifact["threshold"],
            artifact["alarm_refractory_seconds"], artifact["confirm_windows"],
            artifact["step_seconds"])
        duration = float(meta["duration_sec"])
        horizon_end = artifact["sph_seconds"] + artifact["prediction_horizon_seconds"]
        fully_observed = [alarm for alarm in alarms if alarm + horizon_end <= duration]
        censored = [alarm for alarm in alarms if alarm + horizon_end > duration]
        frame["risk_score"] = score
        frame["threshold_prediction"] = (score >= artifact["threshold"]).astype(int)
        frame["warning_event"] = frame.prediction_time.isin(alarms).astype(int)
        frame["warning_active"] = frame.prediction_time.map(lambda t: int(any(
            a + artifact["sph_seconds"] <= t < a + horizon_end for a in alarms)))
        frame["label"] = 0
        frame["eligible"] = 1
        all_frames.append(frame)
        report["recordings"].append({"file": filename, "duration_seconds": duration,
            "sampling_frequency_hz": meta["sampling_frequency"], "channel_count": meta["channel_count"],
            "annotation_seizures": events, "windows": len(frame),
            "score_mean": float(np.mean(score)), "score_median": float(np.median(score)),
            "score_q95": float(np.quantile(score, .95)), "fraction_windows_over_threshold": float(np.mean(score >= artifact["threshold"])),
            "all_emitted_alarms_seconds": alarms,
            "fully_observed_alarms_seconds": fully_observed,
            "right_censored_alarm_seconds": censored,
            "false_alarms_per_fully_observed_recording_hour": len(fully_observed) / max((duration-horizon_end)/3600, 1e-12),
            "recording_metadata": meta})
    pd.concat(all_frames, ignore_index=True).to_csv(OUT_DIR / "predictions.csv", index=False)
    total_observable = sum(max(r["duration_seconds"] - artifact["sph_seconds"] -
                               artifact["prediction_horizon_seconds"], 0) for r in report["recordings"])
    total_false = sum(len(r["fully_observed_alarms_seconds"]) for r in report["recordings"])
    report["combined"] = {"recording_count": len(report["recordings"]),
        "total_hours_with_full_future_horizon_observed": total_observable / 3600,
        "fully_observed_unmatched_alert_count": total_false,
        "false_alarms_per_observable_hour": total_false / (total_observable / 3600) if total_observable else None,
        "interpretation": "No event sensitivity or ROC-AUC can be computed because these recordings have no positive seizure events. These two recordings are small negative controls, not population-level validation."}
    (OUT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str((OUT_DIR / "evaluation.json").relative_to(ROOT)),
        "combined": report["combined"], "recordings": [{k:v for k,v in x.items() if k != "recording_metadata"}
            for x in report["recordings"]]}, indent=2))


if __name__ == "__main__":
    main()
