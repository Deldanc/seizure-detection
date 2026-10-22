"""Exploratory GA–ANFIS retrain augmented with one CHB01 negative EDF.

CHB01_36 remains a separate seizure-free recording check. CHB03_36 has already
been examined in prior work, so its results here are historical/exploratory,
not a newly untouched test. This script never replaces the active model.
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
from src.models.ga_anfis import GeneticFeatureSelector
from src.models.train_anticipatory import (TRAIN, VALIDATION, TEST, SEED, split_groups,
    candidate_names, get_features, get_xy, fit_candidate, scores_metrics, event_evaluation,
    EXISTING_DETECTOR_FEATURES)

FEATURES = ROOT / "data/processed/anticipatory_features.csv"
NEGATIVE_CHECK = ROOT / "results/diagnostics/anticipatory_v4/chb01_negative_control/predictions.csv"
BASE_ARTIFACT = ROOT / "models/anticipatory_ga_anfis_featureaudit_v4.joblib"
OUT = ROOT / "results/diagnostics/anticipatory_chb01_augmented"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(FEATURES)
    extra = pd.read_csv(NEGATIVE_CHECK)
    extra = extra[extra.file.eq("chb01_01.edf")].copy()
    if len(extra) != 720 or extra.label.isna().any() or extra.label.sum() != 0:
        raise ValueError("Expected exactly 720 annotated seizure-free CHB01_01 windows.")
    excluded = {"file", "start", "end", "prediction_time", "label", "eligible",
                "risk_score", "threshold_prediction", "warning_event", "warning_active"}
    candidate = [c for c in frame.columns if c not in excluded]
    extra = extra[[*candidate, "file", "start", "end", "prediction_time", "label", "eligible"]]
    frame = pd.concat([frame, extra], ignore_index=True)
    train = frame[frame.file.isin(TRAIN | {"chb01_01.edf"}) & frame.eligible.eq(1)].copy()
    validation = frame[frame.file.isin(VALIDATION)].copy()
    test = frame[frame.file.isin(TEST)].copy()
    external = pd.read_csv(NEGATIVE_CHECK)
    external = external[external.file.eq("chb01_36.edf")].copy()
    if train.label.nunique() != 2 or validation.label.nunique() != 2:
        raise ValueError("Augmented training and CHB03 validation need both labels.")
    names = candidate_names(train)
    X, y = get_xy(train, names)
    selector = GeneticFeatureSelector(population_size=12, generations=6,
        max_features=4, mutation_rate=.04, random_state=SEED, n_estimators=20)
    selector.fit(X, y, split_groups(train), names)
    subsets = {"ga_selected": selector.selected_features_,
        "existing_detector_four": EXISTING_DETECTOR_FEATURES,
        "theta_alpha_absolute_ch14": ["ch14_theta_power", "ch14_alpha_power"]}
    fitted, validation_results = {}, {}
    for key, subset in subsets.items():
        fitted[key] = fit_candidate(train, validation, subset)
        validation_results[key] = {"features": subset,
            "threshold_selected_on_validation": fitted[key]["threshold"],
            "metrics": fitted[key]["validation_metrics"]}
    chosen = fitted["ga_selected"]
    threshold = chosen["threshold"]
    out_artifact = {"format_version": 1,
        "model_id": "exploratory CHB01 negative-augmented GA-ANFIS",
        "task": "anticipatory risk score", "model": chosen["model"], "scaler": chosen["scaler"],
        "selected_features": selector.selected_features_, "n_features_in": len(selector.selected_features_),
        "threshold": threshold, "threshold_source": "CHB03_35 validation only",
        "training_files": sorted(train.file.unique()), "validation_files": sorted(validation.file.unique()),
        "excluded_negative_check": "chb01_36.edf",
        "ga": {"configuration": selector.config_, "fitness": selector.best_fitness_,
               "history": selector.history_, "seed": SEED},
        "warning": "Exploratory; CHB03_36 was previously examined and CHB01 files contain no seizures."}
    joblib.dump(out_artifact, OUT / "model.joblib")

    test_scores = chosen["model"].predict_proba(chosen["scaler"].transform(
        get_features(test, selector.selected_features_)))[:, 1]
    test_mask = test.eligible.to_numpy(bool)
    test_metrics = scores_metrics(test.loc[test_mask, "label"].to_numpy(int),
                                  test_scores[test_mask], threshold)
    test_metrics["event_level"] = event_evaluation(test, test_scores, threshold)
    external_scores = chosen["model"].predict_proba(chosen["scaler"].transform(
        get_features(external, selector.selected_features_)))[:, 1]
    external_metrics = {"file": "chb01_36.edf", "all_windows_negative": True,
        "score_mean": float(np.mean(external_scores)), "score_median": float(np.median(external_scores)),
        "fraction_windows_above_threshold": float(np.mean(external_scores >= threshold)),
        "threshold": threshold, "event_level": event_evaluation(external, external_scores, threshold)}
    external["risk_score"] = external_scores
    external["threshold_prediction"] = (external_scores >= threshold).astype(int)
    external.to_csv(OUT / "chb01_36_predictions.csv", index=False)
    base = joblib.load(BASE_ARTIFACT)
    base_model_test = base["model"].predict_proba(base["scaler"].transform(
        get_features(test, base["selected_features"])))[:, 1]
    base_model_external = base["model"].predict_proba(base["scaler"].transform(
        get_features(external, base["selected_features"])))[:, 1]
    summary = {"method": "same fixed recording split; added only CHB01_01 seizure-free windows to train; GA rerun; threshold chosen on CHB03_35; CHB01_36 excluded from fit and threshold selection",
        "train_class_counts": {"negative": int((y == 0).sum()), "positive": int((y == 1).sum())},
        "training_recordings": sorted(train.file.unique()),
        "selected_features": selector.selected_features_, "ga_fitness": selector.best_fitness_,
        "validation_comparisons": validation_results, "threshold": threshold,
        "chb03_36_historical_test_metrics": test_metrics,
        "chb01_36_external_negative_check": external_metrics,
        "baseline_v4_test_metrics": scores_metrics(test.loc[test_mask, "label"].to_numpy(int),
            base_model_test[test_mask], base["threshold"]),
        "baseline_v4_chb01_36": {"alarms": event_evaluation(external, base_model_external, base["threshold"])["alarms_generated"],
            "score_median": float(np.median(base_model_external))},
        "limitations": ["CHB03_36 test was examined in prior project iterations; its metrics are not a fresh untouched test.",
            "CHB01_36 has no seizure, so it measures only a recording-level false-warning tendency.",
            "Training still has only one positive seizure event and one patient with seizure-positive labels.",
            "This negative-only augmentation cannot establish event sensitivity or patient-independent prediction."],
        "artifact": str((OUT / "model.joblib").relative_to(ROOT))}
    (OUT / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(OUT.relative_to(ROOT)),
        "selected_features": selector.selected_features_, "threshold": threshold,
        "test": test_metrics, "external_negative": external_metrics,
        "baseline_v4_test": summary["baseline_v4_test_metrics"],
        "baseline_v4_chb01_36": summary["baseline_v4_chb01_36"]}, indent=2))


if __name__ == "__main__":
    main()
