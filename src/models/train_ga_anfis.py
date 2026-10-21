"""Train/evaluate the GA–ANFIS pipeline from repository data.

Run: .venv/bin/python src/models/train_ga_anfis.py
Evaluation split is recording-level: train on 01/15/16/34, validate on 35,
test on 17/36. After reporting that isolated evaluation, a deployment model is
refit on all supplied windows; that deployment artifact is not the test model.
"""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.ga_anfis import GeneticFeatureSelector
from src.models.anfis import SugenoANFIS

DATA = ROOT / "data" / "processed" / "features.csv"
OUT = ROOT / "models" / "ga_anfis_artifact.joblib"
REPORT = ROOT / "results" / "csv" / "ga_anfis_evaluation.json"
SEED = 42
TRAIN_FILES = {"chb03_01.edf", "chb03_15.edf", "chb03_16.edf", "chb03_34.edf"}
VALIDATION_FILES = {"chb03_35.edf"}
TEST_FILES = {"chb03_17.edf", "chb03_36.edf"}
TEST_ANNOTATIONS = {"chb03_17.edf": [], "chb03_36.edf": [(1725.0, 1778.0)]}


def select_and_fit(frame, selector_seed=SEED):
    feature_names = [c for c in frame.columns if c not in {"file", "start", "end", "label"}]
    x = frame[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    y = frame.label.astype(int).to_numpy()
    selector = GeneticFeatureSelector(random_state=selector_seed).fit(
        x, y, frame.file.to_numpy(), feature_names)
    selected = selector.selected_features_
    values = frame[selected].replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy(float)
    scaler = StandardScaler().fit(values)
    X_scaled = scaler.transform(values)
    model = SugenoANFIS().fit(X_scaled, y)
    return selector, scaler, model


def metric_row(y, score, threshold):
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "accuracy": float(accuracy_score(y,pred)),
        "precision": float(precision_score(y,pred,zero_division=0)),
        "sensitivity": float(recall_score(y,pred,zero_division=0)),
        "specificity": float(tn/(tn+fp)) if tn+fp else None,
        "f1": float(f1_score(y,pred,zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp": [int(tn),int(fp),int(fn),int(tp)],
        "roc_auc": float(roc_auc_score(y,score)) if len(np.unique(y)) == 2 else None}


def event_metrics(frame, scores, threshold, merge_gap=30.0, min_duration=10.0):
    """Exploratory event summary using annotations and fixed post-processing."""
    predicted = []
    scored = frame[["file", "start", "end"]].copy()
    scored["positive"] = np.asarray(scores) >= threshold
    for filename, group in scored.groupby("file", sort=False):
        intervals = [(float(row.start), float(row.end)) for row in group.itertuples()
                     if row.positive]
        merged = []
        for start, end in intervals:
            if merged and start - merged[-1][1] <= merge_gap:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        predicted.extend((filename, start, end) for start, end in merged
                         if end - start >= min_duration)
    truth = [(name, start, end) for name, events in TEST_ANNOTATIONS.items()
             for start, end in events]
    hits = []
    for name, onset, offset in truth:
        matches = [p for p in predicted if p[0] == name and p[1] < offset and p[2] > onset]
        hits.append(matches)
    matched = sum(bool(matches) for matches in hits)
    false_alarms = sum(not any(p[0] == name and p[1] < offset and p[2] > onset
                              for name, onset, offset in truth)
                       for p in predicted)
    durations = sum(float(np.sum(group.end.to_numpy() - group.start.to_numpy()))
                    for _, group in frame.groupby("file"))
    latencies = [min(event[1] for event in matches) - truth[i][1]
                 for i, matches in enumerate(hits) if matches]
    return {"ground_truth_events":len(truth), "detected_events":int(matched),
        "event_sensitivity":float(matched/len(truth)) if truth else None,
        "predicted_events":len(predicted), "false_alarm_events":int(false_alarms),
        "false_alarms_per_hour":float(false_alarms/(durations/3600)) if durations else None,
        "onset_offset_seconds":latencies,
        "postprocessing":{"merge_gap_seconds":merge_gap,"minimum_duration_seconds":min_duration},
        "interpretation":"Exploratory event detection on one held-out annotated event; onset offset is not a preictal warning latency."}


def main(evaluation_only=False):
    df = pd.read_csv(DATA)
    train = df[df.file.isin(TRAIN_FILES)].copy()
    validation = df[df.file.isin(VALIDATION_FILES)].copy()
    test = df[df.file.isin(TEST_FILES)].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Recording-level train/validation/test split is missing data.")
    selector, scaler, model = select_and_fit(train)
    Xv = scaler.transform(validation[selector.selected_features_].replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy(float))
    val_score = model.predict_proba(Xv)[:,1]
    # Choose threshold using validation F1 only; test remains untouched.
    thresholds = np.arange(0.05, 0.951, 0.01)
    val_threshold = max(thresholds, key=lambda t: (f1_score(validation.label, val_score >= t, zero_division=0),
        recall_score(validation.label, val_score >= t, zero_division=0), -t))
    Xt = scaler.transform(test[selector.selected_features_].replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy(float))
    test_score = model.predict_proba(Xt)[:,1]
    # Same inputs/splits/threshold for the RF comparator.
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
        min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    rf.fit(scaler.transform(train[selector.selected_features_].replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy(float)), train.label)
    rf_val = rf.predict_proba(Xv)[:,1]
    rf_threshold = max(thresholds, key=lambda t: (f1_score(validation.label, rf_val >= t, zero_division=0),
        recall_score(validation.label, rf_val >= t, zero_division=0), -t))
    rf_test = rf.predict_proba(Xt)[:,1]
    report = {"dataset": {"rows": int(len(df)), "features": 230,
        "labels": {str(k): int(v) for k,v in df.label.value_counts().sort_index().items()},
        "patient": "CHB-MIT subject chb03 only", "window_seconds": 5,
        "label_semantics": "at least 2.5 seconds of overlap with annotated seizure"},
        "split": {"train_files": sorted(TRAIN_FILES), "validation_files": sorted(VALIDATION_FILES),
            "test_files": sorted(TEST_FILES), "counts": {"train":len(train),"validation":len(validation),"test":len(test)},
            "positive_windows": {"train":int(train.label.sum()),"validation":int(validation.label.sum()),"test":int(test.label.sum())}},
        "ga": {"configuration": selector.config_, "selected_features": selector.selected_features_,
            "best_grouped_cv_fitness": selector.best_fitness_, "history": selector.history_},
        "anfis": model.training_, "validation_threshold": float(val_threshold),
        "test": {"ga_anfis": metric_row(test.label, test_score, val_threshold),
            "random_forest_same_features": metric_row(test.label, rf_test, rf_threshold),
            "event_level": {"ga_anfis":event_metrics(test, test_score, val_threshold),
                "random_forest":event_metrics(test, rf_test, rf_threshold)},
            "random_forest_validation_threshold": float(rf_threshold),
            "score_note": "Sigmoid outputs are model scores; no probability calibration was fitted.",
            "window_metrics_are_window_level": True}}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    if evaluation_only:
        print(json.dumps({"evaluation_report":str(REPORT), "test":report["test"]}, indent=2))
        return

    # Deployment fit: now use all available labeled windows, distinct from test report.
    deploy_selector, deploy_scaler, deploy_model = select_and_fit(df, selector_seed=SEED)
    artifact = {"format_version": 1, "model_id": "GA–ANFIS Sugeno window detector",
        "model": deploy_model, "scaler": deploy_scaler,
        "selected_features": deploy_selector.selected_features_,
        "feature_metadata": {"window_seconds":5, "gamma_band_hz":[30,100],
            "power_unit":"V^2 (EDF calibrated signal units)",
            "channel_order":"EDF channel order ch0...ch22",
            "channel_names":["FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3","C3-P3","P3-O1","FP2-F4","F4-C4","C4-P4","P4-O2","FP2-F8","F8-T8","T8-P8","P8-O2","FZ-CZ","CZ-PZ","P7-T7","T7-FT9","FT9-FT10","FT10-T8","T8-P8"]},
        "ga": {"configuration":deploy_selector.config_, "history":deploy_selector.history_,
            "fitness":deploy_selector.best_fitness_},
        "threshold": float(val_threshold), "training_rows":len(df),
        "warning":"Deployment fit includes all supplied rows; evaluation metrics use the held-out recording split above."}
    joblib.dump(artifact, OUT)
    print(json.dumps({"evaluation_report":str(REPORT), "artifact":str(OUT),
        "selected_features":selector.selected_features_, "validation_threshold":float(val_threshold),
        "test":report["test"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-only", action="store_true",
        help="run the recording-level train/validation/test evaluation without replacing the deployment artifact")
    main(evaluation_only=parser.parse_args().evaluation_only)
