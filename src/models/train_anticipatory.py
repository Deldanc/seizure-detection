"""Train and evaluate versioned anticipatory GA–ANFIS experiments.

GA fitness uses training-only grouped temporal folds. A separate complete
recording selects the alarm threshold and reports candidate-subset validation
metrics. The final test recording is untouched until the final scoring step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.ga_anfis import GeneticFeatureSelector
from src.models.anfis import SugenoANFIS
from src.preprocessing.anticipatory_pipeline import (FEATURE_PATH, CONFIG_PATH,
    PREDICTION_HORIZON_SEC, SPH_SEC, POSTICTAL_EXCLUSION_SEC,
    INTERICTAL_GUARD_SEC, WINDOW_SEC, STEP_SEC, TARGET_SFREQ, BANDS,
    FEATURE_SCHEMA_VERSION, CATALOG_PATH, parse_summary, feature_catalog)

ARTIFACT = ROOT / "models" / "anticipatory_ga_anfis_featureaudit_v4.joblib"
REPORT = ROOT / "reports" / "evaluation_report.md"
FEATURE_REPORT = ROOT / "reports" / "feature_selection_report.md"
OUTPUT = ROOT / "results" / "csv" / "anticipatory_featureaudit_v4_predictions.csv"
METRICS_JSON = ROOT / "results" / "csv" / "anticipatory_featureaudit_v4_evaluation.json"
TRAIN = {"chb03_34.edf", "chb03_15.edf", "chb03_16.edf", "chb03_17.edf",
         "chb03_02.edf", "chb03_03.edf", "chb03_04.edf"}
VALIDATION = {"chb03_35.edf"}
TEST = {"chb03_36.edf"}
SEED = 42
REFRACTORY_SEC = 30 * 60
CONFIRM_WINDOWS = 3
EXISTING_DETECTOR_FEATURES = ["ch1_energy", "ch6_rms", "ch14_gamma_power", "ch18_std"]


def split_groups(frame):
    """Use separate 5-minute recording blocks for training-only GA fitness."""
    return np.asarray([f"{name}:{int(start // 300)}" for name, start in zip(frame.file, frame.start)])


def candidate_names(frame):
    excluded = {"file", "start", "end", "prediction_time", "label", "eligible"}
    return [name for name in frame.columns if name not in excluded]


def get_features(frame, names):
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"Selected feature(s) are missing from the dataset: {missing}")
    values = frame[names].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Feature matrix contains missing or non-finite values.")
    return values


def get_xy(frame, names):
    return get_features(frame, names), frame.label.to_numpy(int)


def scores_metrics(y, score, threshold):
    pred = np.asarray(score) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n_windows": int(len(y)), "positive_windows": int(np.sum(y)),
        "threshold": float(threshold), "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(y, score)) if np.any(y) else None,
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)]}


def generate_alarms(times, scores, threshold, refractory=REFRACTORY_SEC,
                    confirm_windows=CONFIRM_WINDOWS, step_seconds=STEP_SEC):
    alarms, streak, next_allowed = [], 0, -np.inf
    previous = None
    for time, score in zip(times, scores):
        if previous is not None and time - previous > step_seconds * 1.5:
            streak = 0
        streak = streak + 1 if score >= threshold else 0
        if streak >= confirm_windows and time >= next_allowed:
            alarms.append(float(time))
            next_allowed = float(time) + refractory
            streak = 0
        previous = time
    return alarms


def event_evaluation(frame, score, threshold):
    annotations = parse_summary()
    filename = str(frame.file.iloc[0])
    scored = frame.copy()
    scored["__score"] = np.asarray(score)
    if "eligible" in scored:
        scored["__eligible"] = scored.eligible.to_numpy(bool)
    else:
        scored["__eligible"] = scored.label.notna().to_numpy()
    held = scored[scored.file == filename].sort_values("prediction_time").copy()
    times = held.prediction_time.to_numpy(float)
    alarms = generate_alarms(times, held["__score"].to_numpy(float), threshold,
        REFRACTORY_SEC, CONFIRM_WINDOWS, STEP_SEC)
    seizures = annotations.get(filename, [])
    matched_onsets, lead_times, false_alarms = set(), [], []
    for alarm in alarms:
        match = next((onset for onset, _ in seizures if onset not in matched_onsets
            and alarm + SPH_SEC <= onset <= alarm + SPH_SEC + PREDICTION_HORIZON_SEC), None)
        if match is None:
            false_alarms.append(alarm)
        else:
            matched_onsets.add(match)
            lead_times.append(match - alarm)
    eligible_times = held.loc[held["__eligible"], "prediction_time"].to_numpy(float)
    eligible_hours = len(eligible_times) * STEP_SEC / 3600
    monitoring_hours = len(held) * STEP_SEC / 3600
    warning_intervals = [(t + SPH_SEC, t + SPH_SEC + PREDICTION_HORIZON_SEC) for t in alarms]
    warned_seconds = sum(STEP_SEC for t in eligible_times
        if any(start <= t < end for start, end in warning_intervals))
    return {"seizure_events": len(seizures), "successfully_predicted_events": len(matched_onsets),
        "event_sensitivity": len(matched_onsets) / len(seizures) if seizures else None,
        "alarms_generated": alarms, "false_alarm_times": false_alarms,
        "matched_alarm_times": [t for t in alarms if t not in false_alarms],
        "false_alarm_count": len(false_alarms), "eligible_monitoring_hours": eligible_hours,
        "total_monitoring_hours": monitoring_hours,
        "false_alarms_per_hour": len(false_alarms) / monitoring_hours if monitoring_hours else None,
        "false_alarms_per_eligible_hour": len(false_alarms) / eligible_hours if eligible_hours else None,
        "warning_lead_times_seconds": lead_times,
        "active_warning_fraction_percent": min(100.0, warned_seconds / max(eligible_hours * 3600, 1) * 100),
        "alarm_policy": {"consecutive_windows": CONFIRM_WINDOWS, "refractory_seconds": REFRACTORY_SEC,
            "event_match": "onset lies in [alarm+SPH, alarm+SPH+horizon]"}}


def select_threshold(y, scores, validation_frame):
    thresholds = np.arange(0.05, 0.991, 0.01)
    eligible = validation_frame.eligible.to_numpy(bool) if "eligible" in validation_frame else validation_frame.label.notna().to_numpy()
    eligible_scores = np.asarray(scores)[eligible]
    def objective(threshold):
        event = event_evaluation(validation_frame, scores, threshold)
        metrics = scores_metrics(y, eligible_scores, threshold)
        return (event["event_sensitivity"] or 0.0, -event["false_alarm_count"],
                metrics["specificity"], metrics["precision"], metrics["f1"], threshold)
    return float(max(thresholds, key=objective))


def fit_candidate(train, validation, names):
    Xtr, ytr = get_xy(train, names)
    Xv = get_features(validation, names)
    validation_eligible = validation.eligible.to_numpy(bool) if "eligible" in validation else validation.label.notna().to_numpy()
    yv = validation.loc[validation_eligible, "label"].to_numpy(int)
    scaler = StandardScaler().fit(Xtr)
    model = SugenoANFIS().fit(scaler.transform(Xtr), ytr)
    validation_scores = model.predict_proba(scaler.transform(Xv))[:, 1]
    threshold = select_threshold(yv, validation_scores, validation)
    metrics = scores_metrics(yv, validation_scores[validation_eligible], threshold)
    metrics["event_level"] = event_evaluation(validation, validation_scores, threshold)
    return {"features": list(names), "feature_count": len(names), "scaler": scaler,
        "model": model, "validation_scores": validation_scores,
        "threshold": threshold, "validation_metrics": metrics}


def main(refractory_minutes=30.0, confirm_windows=3, max_features=4,
         population_size=12, generations=6):
    global PREDICTION_HORIZON_SEC, SPH_SEC, POSTICTAL_EXCLUSION_SEC
    global INTERICTAL_GUARD_SEC, WINDOW_SEC, STEP_SEC, REFRACTORY_SEC, CONFIRM_WINDOWS
    config = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    if config.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Feature dataset schema is stale. Rebuild it with prepare_anticipatory_data.py first.")
    WINDOW_SEC = float(config.get("window_seconds", WINDOW_SEC))
    STEP_SEC = float(config.get("step_seconds", STEP_SEC))
    PREDICTION_HORIZON_SEC = float(config.get("prediction_horizon_seconds", PREDICTION_HORIZON_SEC))
    SPH_SEC = float(config.get("sph_seconds", SPH_SEC))
    POSTICTAL_EXCLUSION_SEC = float(config.get("postictal_exclusion_seconds", POSTICTAL_EXCLUSION_SEC))
    INTERICTAL_GUARD_SEC = float(config.get("interictal_guard_seconds", INTERICTAL_GUARD_SEC))
    REFRACTORY_SEC = refractory_minutes * 60
    CONFIRM_WINDOWS = int(confirm_windows)
    if min(WINDOW_SEC, STEP_SEC, PREDICTION_HORIZON_SEC, SPH_SEC) <= 0 or CONFIRM_WINDOWS < 1:
        raise ValueError("Window, step, horizon, SPH, and confirmation-window settings must be positive.")
    if not 1 <= max_features <= 4:
        raise ValueError("ANFIS currently supports a selected subset of 1 to 4 features.")
    frame = pd.read_csv(FEATURE_PATH)
    available = candidate_names(frame)
    train_recordings = frame[frame.file.isin(TRAIN)].copy()
    validation = frame[frame.file.isin(VALIDATION)].copy()
    test = frame[frame.file.isin(TEST)].copy()
    train = train_recordings[train_recordings.eligible == 1].copy()
    validation_eligible = validation[validation.eligible == 1].copy()
    test_eligible = test[test.eligible == 1].copy()
    if any(f.empty for f in (train, validation_eligible, test_eligible)) or train.label.nunique() < 2 or validation_eligible.label.nunique() < 2:
        raise ValueError("Train/validation/test split must contain data and both train/validation labels.")
    if set(train.file) & set(validation.file) or set(train.file) & set(test.file) or set(validation.file) & set(test.file):
        raise ValueError("Recording leakage detected between train, validation and test.")
    X, y = get_xy(train, available)
    selector = GeneticFeatureSelector(population_size=population_size, generations=generations,
        max_features=max_features, mutation_rate=0.04, random_state=SEED, n_estimators=20)
    selector.fit(X, y, split_groups(train), available)
    selected = selector.selected_features_

    # Fixed comparison subsets are evaluated identically on validation. They do
    # not replace the training-CV GA winner or use the final test set.
    comparison_sets = {"ga_selected": selected,
        "existing_detector_four": EXISTING_DETECTOR_FEATURES,
        "theta_alpha_absolute_ch14": ["ch14_theta_power", "ch14_alpha_power"],
        "theta_alpha_relative_ch14": ["ch14_theta_relative_power", "ch14_alpha_relative_power"]}
    comparison = {}
    fitted = {}
    for label, subset in comparison_sets.items():
        if all(name in available for name in subset):
            fitted[label] = fit_candidate(train, validation, subset)
            comparison[label] = {"features": subset,
                "feature_count": len(subset),
                "threshold_selected_on_validation": fitted[label]["threshold"],
                "validation_metrics": fitted[label]["validation_metrics"]}

    selected_fit = fitted["ga_selected"]
    test_X = get_features(test, selected)
    test_mask = test.eligible.to_numpy(bool)
    yt = test.loc[test_mask, "label"].to_numpy(int)
    test_scores = selected_fit["model"].predict_proba(selected_fit["scaler"].transform(test_X))[:, 1]
    threshold = selected_fit["threshold"]
    test_metrics = scores_metrics(yt, test_scores[test_mask], threshold)
    test_metrics["event_level"] = event_evaluation(test, test_scores, threshold)
    alarms = test_metrics["event_level"]["alarms_generated"]
    test["risk_score"] = test_scores
    test["threshold_prediction"] = (test_scores >= threshold).astype(int)
    test["prediction_time"] = test.end
    test["warning_event"] = test.prediction_time.isin(alarms).astype(int)
    test["warning_active"] = test.prediction_time.map(lambda t: int(any(
        a + SPH_SEC <= t < a + SPH_SEC + PREDICTION_HORIZON_SEC for a in alarms)))
    test["future_risk_start"] = test.prediction_time.map(lambda t: t + SPH_SEC if t in alarms else np.nan)
    test["future_risk_end"] = test.prediction_time.map(lambda t: t + SPH_SEC + PREDICTION_HORIZON_SEC if t in alarms else np.nan)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    test[["file", "start", "end", "prediction_time", "eligible", "label", "risk_score", "threshold_prediction",
        "warning_event", "warning_active", "future_risk_start", "future_risk_end"]].to_csv(OUTPUT, index=False)

    baseline_metrics = {"always_interictal": scores_metrics(yt, np.zeros(len(yt)), 0.5)}
    Xtr, ytr = get_xy(train, selected)
    Xv = get_features(validation, selected)
    val_mask = validation.eligible.to_numpy(bool)
    yv = validation.loc[val_mask, "label"].to_numpy(int)
    scaler_baselines = StandardScaler().fit(Xtr)
    Xtr_s, Xv_s = scaler_baselines.transform(Xtr), scaler_baselines.transform(Xv)
    Xt_s = scaler_baselines.transform(test_X)
    for name, estimator in (("logistic_regression", LogisticRegression(class_weight="balanced",
            max_iter=1000, random_state=SEED)),
            ("random_forest", RandomForestClassifier(n_estimators=300, class_weight="balanced",
            min_samples_leaf=2, random_state=SEED, n_jobs=-1))):
        estimator.fit(Xtr_s, ytr)
        val_scores = estimator.predict_proba(Xv_s)[:, 1]
        val_threshold = select_threshold(yv, val_scores, validation)
        baseline_test_scores = estimator.predict_proba(Xt_s)[:, 1]
        baseline_metrics[name] = scores_metrics(yt, baseline_test_scores[test_mask], val_threshold)

    split = {"train": sorted(train.file.unique()), "validation": sorted(validation.file.unique()),
        "test": sorted(test.file.unique()), "training_recordings": train.file.nunique(),
        "validation_recordings": validation.file.nunique(), "test_recordings": test.file.nunique(),
        "training_rows": len(train), "validation_rows": len(validation_eligible), "test_rows": len(test_eligible),
        "full_recording_windows": {"train": len(train_recordings), "validation": len(validation), "test": len(test)},
        "positive_windows": {"train": int(ytr.sum()), "validation": int(validation_eligible.label.sum()), "test": int(yt.sum())}}
    full_catalog = feature_catalog(config["channel_names"])
    full_names = full_catalog.feature.tolist()
    artifact = {"format_version": 2, "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_id": "experimental anticipatory GA–ANFIS feature audit v4",
        "task": "anticipatory risk score", "model": selected_fit["model"],
        "scaler": selected_fit["scaler"], "selected_features": selected,
        "n_features_in": len(selected), "ga": {"configuration": selector.config_,
        "history": selector.history_, "fitness": selector.best_fitness_, "seed": SEED},
        "threshold": threshold, "prediction_horizon_seconds": PREDICTION_HORIZON_SEC,
        "sph_seconds": SPH_SEC, "postictal_exclusion_seconds": POSTICTAL_EXCLUSION_SEC,
        "interictal_guard_seconds": INTERICTAL_GUARD_SEC,
        "window_seconds": WINDOW_SEC, "step_seconds": STEP_SEC,
        "sampling_frequency_hz": float(config["target_sampling_frequency_hz"]),
        "alarm_refractory_seconds": REFRACTORY_SEC, "confirm_windows": CONFIRM_WINDOWS,
        "feature_names": full_names, "channel_names": config["channel_names"],
        "bands_hz": BANDS, "welch": config["welch"],
        "resampling": config["resampling"], "preprocessing": config["preprocessing"],
        "threshold_selection": "validation event sensitivity, then fewer full-timeline false alarms, then window specificity, precision, F1, and higher threshold",
        "warning": "One subject and one held-out event; research prototype only."}
    ARTIFACT.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, ARTIFACT)

    report = {"protocol": {"horizon_seconds": PREDICTION_HORIZON_SEC,
        "sph_seconds": SPH_SEC, "window_seconds": WINDOW_SEC, "split": split,
        "ga_features": selected, "threshold_selected_on_validation": threshold,
        "class_counts_test": {"negative": int((yt == 0).sum()), "positive": int((yt == 1).sum())}},
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "candidate_feature_count": len(full_names),
        "feature_catalog": str(CATALOG_PATH.relative_to(ROOT)),
        "feature_comparisons_validation": comparison, "ga_anfis_test": test_metrics,
        "baselines_test": baseline_metrics,
        "ga_search": {"configuration": selector.config_, "selected_features": selected,
            "fitness": selector.best_fitness_, "history": selector.history_, "seed": SEED},
        "model_training": selected_fit["model"].training_,
        "outputs": {"predictions_csv": str(OUTPUT.relative_to(ROOT)),
            "artifact": str(ARTIFACT.relative_to(ROOT)), "feature_catalog": str(CATALOG_PATH.relative_to(ROOT))},
        "limitations": ["Only one subject is locally available; three annotated chb03 EDFs are missing.",
            "Training contains one seizure event; training-only GA folds are five-minute temporal recording blocks, not independent-event folds.",
            "Validation and test each contain one annotated seizure event. Candidate validation metrics and selected thresholds are unstable.",
            "The test recording is not used for GA selection, feature comparisons, scaler/model fitting, or threshold selection.",
            "Scores are uncalibrated. Rates below 177.78 Hz and channel sets other than the saved montage are rejected; supported rates are causally resampled.",
            "The four-feature cap is an ANFIS implementation constraint (2^d fuzzy rules), not an empirically established optimum."]}
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(report, indent=2) + "\n")
    REPORT.write_text(render_report(report))
    FEATURE_REPORT.write_text(render_feature_report(report, full_catalog, selector))
    print(json.dumps({"evaluation_report": str(REPORT), "feature_report": str(FEATURE_REPORT),
        "artifact": str(ARTIFACT), "selected_features": selected, "threshold": threshold,
        "validation_comparison": comparison, "test": test_metrics, "baselines_test": baseline_metrics}, indent=2))


def render_feature_report(report, catalog, selector):
    lines = ["# Anticipatory EEG feature-selection audit", "",
        "## Saved detector quartet provenance", "",
        f"The existing seizure-activity detector artifact stores `{EXISTING_DETECTOR_FEATURES}` and its GA metadata records a six-generation, population-12, seed-42 search with max four features. These inputs were genuinely emitted by a GA for the older ictal-window detection task. Four was a configured maximum, not a demonstrated optimal count. The anticipatory GA is rerun separately against annotation-derived preictal labels.", "",
        "## Candidate set and signal processing", "",
        f"Schema v{report['feature_schema_version']} has {report['candidate_feature_count']} candidates: 15 per channel across {len(catalog.channel_name.unique())} saved channels (5 time-domain, 5 absolute band-power, and 5 relative band-power descriptors). Signals are in MNE calibrated volts. A one-sided causal polyphase FIR resamples supported inputs to 256 Hz, then a causal fourth-order Butterworth 0.5–100 Hz filter and causal 60 Hz notch are applied. Features use non-overlapping {report['protocol']['window_seconds']:g}-second windows at 256 Hz. Welch uses two-second Hann segments, 50% overlap, density scaling and 0.5 Hz resolution. Absolute integrated powers are V²; relative powers are unitless fractions of total 0.5–80 Hz power.", "",
        "| Band | Range | Present features |", "|---|---:|---|",
        "| Delta | 0.5–4 Hz | absolute and relative power |",
        "| Theta | 4–8 Hz | absolute and relative power |",
        "| Alpha | 8–13 Hz | absolute and relative power |",
        "| Beta | 13–30 Hz | absolute and relative power |",
        "| Gamma | 30–80 Hz | absolute and relative power |", "",
        "The band ranges are contiguous and non-overlapping at boundaries. Exact edge PSD values are interpolated before integration. Relative-band denominator is integrated 0.5–80 Hz power; the five relative bands sum to approximately 1 for nonzero signals. No band is forced into the GA subset.", "",
        "## Channel mapping", "",
        "Feature prefix `chN` maps to zero-based position N in the canonical channel table below. Inference accepts the same channel set in a different order and reorders by normalized channel name. Missing/extra/ambiguous channels are rejected. Input rates below 177.78 Hz are rejected because the causal 0.5-100 Hz bandpass uses a 0.45*sfreq Nyquist margin; higher supported rates are resampled to 256 Hz with a causal one-sided polyphase FIR and its delay is retained.", "",
        "| Index | EDF channel |", "|---:|---|"]
    for i, name in enumerate(catalog.drop_duplicates("channel_index").channel_name.tolist()):
        lines.append(f"| {i} | `{name}` |")
    lines += ["", "The complete 345-row feature-name/channel/formula/unit mapping is machine-readable at `data/processed/anticipatory_feature_catalog.csv`.", "",
        "## GA selection and subset comparisons", "",
        f"The anticipatory GA searched {report['candidate_feature_count']} candidates, allowing 1–{selector.config_['max_features']} features. Four is the ANFIS implementation limit because it creates 2^d rules; it is not claimed to be biologically or statistically optimal. The selector's fitness uses training-only five-minute grouped cross-validation with RF window F1 and a small feature-count penalty. It selected `{report['ga_search']['selected_features']}` (training grouped-CV fitness {report['ga_search']['fitness']:.3f}).", "",
        "Validation comparison uses the same train-fitted scaler/model approach and same event-aware threshold policy for each subset. Validation is one event and therefore exploratory; the final test was not used.", "",
        "| Candidate subset | Count | Features | Val threshold | Val precision | Val sensitivity | Val F1 | Val ROC-AUC | Val event hit | Val false alarms |", "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, item in report["feature_comparisons_validation"].items():
        m = item["validation_metrics"]
        e = m["event_level"]
        lines.append(f"| {key} | {item['feature_count']} | `{', '.join(item['features'])}` | {item['threshold_selected_on_validation']:.2f} | {m['precision']:.3f} | {m['sensitivity']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} | {e['successfully_predicted_events']}/{e['seizure_events']} | {e['false_alarm_count']} |")
    lines += ["", "The existing detector quartet is retrained using anticipatory labels only for this comparison; it is not reused as a trained predictor. Its ch14 gamma feature is recomputed with the corrected 30-80 Hz definition, so this is a feature-set comparison rather than byte-for-byte reproduction of the old detector input. Theta/alpha absolute and relative subsets are diagnostic comparisons, not forced selections. The GA winner is determined by training-only grouped CV, not by the validation ranking.", "",
        "## Limits", "", "- Only subject chb03 is available locally; EDFs 02, 03 and 04, which include additional annotated seizures, are missing.",
        "- Only one positive seizure event is present in train, validation and test respectively. Recording-independent event selection cannot be estimated reliably.",
        "- Training temporal blocks reduce leakage between nearby windows but are not independent events. A genuinely event-aware GA validation is blocked by too few labeled events.",
        "- The final test recording is used once for the selected GA subset after all comparisons and threshold selection.",
        "- Adding relative power makes the candidate features available; it does not establish that they improve prediction.", ""]
    return "\n".join(lines)


def render_report(report):
    p, test = report["protocol"], report["ga_anfis_test"]
    e = test["event_level"]
    lines = ["# Anticipatory prediction evaluation", "",
        "## Protocol and split", "",
        f"Prediction horizon {p['horizon_seconds']/60:.0f} min; SPH {p['sph_seconds']/60:.0f} min; windows {p['window_seconds']} s. Feature schema v{report['feature_schema_version']} has {report['candidate_feature_count']} features.",
        f"Training-only GA chose `{', '.join(p['ga_features'])}` using grouped five-minute blocks; validation selected threshold {p['threshold_selected_on_validation']:.2f}.",
        f"Train {p['split']['train']}; validation {p['split']['validation']}; untouched test {p['split']['test']}.",
        f"Test class counts: {p['class_counts_test']}.", "", "## Held-out test results", "",
        f"Window metrics: precision {test['precision']:.3f}, sensitivity {test['sensitivity']:.3f}, specificity {test['specificity']:.3f}, F1 {test['f1']:.3f}, AP {test['pr_auc_average_precision']:.3f}, ROC-AUC {test['roc_auc']:.3f}; TN/FP/FN/TP {test['confusion_matrix_tn_fp_fn_tp']}.",
        f"Event result: {e['successfully_predicted_events']}/{e['seizure_events']} events; {e['false_alarm_count']} false alarms over {e['total_monitoring_hours']:.3f} recording hours ({e['false_alarms_per_hour']:.3f}/recording hour; {e['false_alarms_per_eligible_hour']:.3f}/eligible hour); lead times in seconds {e['warning_lead_times_seconds']}.",
        f"Eligible monitoring time under warning: {e['active_warning_fraction_percent']:.1f}%.", "",
        "Alarms are generated on the full continuous recording timeline. Window metrics use only labeled eligible windows; false-alarm counts include alarms outside eligible windows. The test set is one annotated event in one patient; these are descriptive one-event estimates and do not support a performance claim.", "",
        "## Validation feature-subset comparison", "", "See `reports/feature_selection_report.md` for the same-protocol validation metrics for the GA subset, existing detector quartet, and theta/alpha absolute/relative subsets. Validation metrics are exploratory and were not used to select among the GA candidates.", "",
        "## Baselines", "", "Model | Precision | Sensitivity | Specificity | F1 | AP | ROC-AUC | TN/FP/FN/TP", "---|---:|---:|---:|---:|---:|---:|---|"]
    for name, m in report["baselines_test"].items():
        lines.append(f"{name} | {m['precision']:.3f} | {m['sensitivity']:.3f} | {m['specificity']:.3f} | {m['f1']:.3f} | {m['pr_auc_average_precision']:.3f} | {m['roc_auc']:.3f} | {m['confusion_matrix_tn_fp_fn_tp']}")
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in report["limitations"]] + [
        "- Threshold alarms require three consecutive crossings; matching requires onset inside the SPH-plus-horizon interval.",
        "- A one-event event-sensitivity result is not a dependable prediction estimate.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refractory-minutes", type=float, default=30)
    parser.add_argument("--confirm-windows", type=int, default=3)
    parser.add_argument("--max-features", type=int, choices=range(1, 5), default=4,
        help="maximum GA subset size; ANFIS is implemented for at most four inputs")
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--generations", type=int, default=6)
    args = parser.parse_args()
    main(args.refractory_minutes, args.confirm_windows, args.max_features,
         args.population_size, args.generations)
