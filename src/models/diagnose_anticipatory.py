"""Reproduce saved anticipatory metrics and write class/split diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.train_anticipatory import scores_metrics, TRAIN, VALIDATION, TEST
from src.preprocessing.anticipatory_pipeline import FEATURE_PATH, CONFIG_PATH, parse_summary, prediction_labels

ARTIFACT = ROOT / "models/anticipatory_ga_anfis_featureaudit_v4.joblib"
SAVED = ROOT / "results/csv/anticipatory_featureaudit_v4_predictions.csv"
OUT = ROOT / "results/diagnostics/anticipatory_v4"
PLOTS = ROOT / "reports/figures/anticipatory_v4"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(ARTIFACT)
    saved = pd.read_csv(SAVED)
    all_features = pd.read_csv(FEATURE_PATH)
    config = json.loads(CONFIG_PATH.read_text())
    annotations = parse_summary()

    # Reconstruct every label from annotation timing instead of trusting CSV labels.
    labels = []
    for row in all_features.itertuples(index=False):
        labels.append(prediction_labels(float(row.start), float(row.end),
            annotations.get(str(row.file).lower(), []), artifact["prediction_horizon_seconds"],
            artifact["sph_seconds"], artifact["postictal_exclusion_seconds"],
            artifact["interictal_guard_seconds"]))
    all_features["reconstructed_label"] = labels
    all_features["eligible"] = all_features.reconstructed_label.notna().astype(int)
    all_features["label"] = all_features.reconstructed_label

    split_defs = {"train": TRAIN, "validation": VALIDATION, "test": TEST}
    split_rows, score_rows = [], []
    for split_name, files in split_defs.items():
        subset = all_features[all_features.file.isin(files)].copy()
        eligible = subset[subset.eligible == 1].copy()
        X = eligible[artifact["selected_features"]].to_numpy(float)
        score = artifact["model"].predict_proba(artifact["scaler"].transform(X))[:, 1]
        eligible["risk_score"] = score
        for cls, group in eligible.groupby("label"):
            vals = group.risk_score.to_numpy()
            split_rows.append({"split": split_name, "n": len(group), "class": int(cls),
                "mean": float(vals.mean()), "std": float(vals.std()),
                "min": float(vals.min()), "q25": float(np.quantile(vals, .25)),
                "median": float(np.median(vals)), "q75": float(np.quantile(vals, .75)),
                "max": float(vals.max())})
        score_rows.append(eligible[["file", "start", "end", "label", "risk_score"]])
    score_frame = pd.concat(score_rows, ignore_index=True)
    score_frame.to_csv(OUT / "scores_by_split.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUT / "score_distribution_by_split_class.csv", index=False)

    test = all_features[all_features.file.isin(TEST)].sort_values("end").copy()
    test = test.merge(saved[["file", "end", "risk_score", "label"]].rename(
                      columns={"risk_score":"saved_risk_score", "label":"saved_label"}),
                      on=["file", "end"], how="left", validate="one_to_one")
    if test.saved_risk_score.isna().any():
        raise AssertionError("Saved test predictions do not cover the complete test recording.")
    test["reconstructed_label"] = test.label
    Xall = test[artifact["selected_features"]].to_numpy(float)
    recomputed = artifact["model"].predict_proba(artifact["scaler"].transform(Xall))[:, 1]
    score_delta = float(np.max(np.abs(recomputed - test.saved_risk_score.to_numpy())))
    elig = test.label.notna().to_numpy()
    y = test.loc[elig, "label"].to_numpy(int)
    s = test.loc[elig, "saved_risk_score"].to_numpy(float)
    threshold = float(artifact["threshold"])
    metrics = scores_metrics(y, s, threshold)
    pred = (s >= threshold).astype(int)
    metrics["reversed_score_roc_auc_diagnostic_only"] = float(roc_auc_score(y, -s))
    metrics["score_orientation"] = "class 1 is documented preictal; predict_proba column 1 is used"
    metrics["class_counts"] = {"negative": int((y == 0).sum()), "positive": int((y == 1).sum())}
    metrics["positive_prevalence"] = float(y.mean())
    metrics["score_reproduction_max_abs_delta"] = score_delta
    metrics["label_reconstruction_mismatches"] = int((test.label.fillna(-1).to_numpy() !=
        test.saved_label.fillna(-1).to_numpy()).sum())
    metrics["test_scores_by_label"] = {str(int(cls)): {"n": int(len(group)), "mean": float(group.mean()),
        "median": float(group.median()), "q25": float(group.quantile(.25)), "q75": float(group.quantile(.75))}
        for cls, group in test.loc[elig].groupby("label")["saved_risk_score"]}
    (OUT / "independent_reproduction.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # ROC, PR and confusion matrix for eligible held-out windows.
    fpr, tpr, _ = roc_curve(y, s)
    precision, recall, _ = precision_recall_curve(y, s)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(fpr, tpr, label=f"ROC-AUC={metrics['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray"); axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out ROC"); axes[0].legend()
    axes[1].plot(recall, precision, label=f"AP={metrics['pr_auc_average_precision']:.3f}")
    axes[1].axhline(y.mean(), linestyle="--", color="gray", label=f"Prevalence={y.mean():.3f}"); axes[1].set(xlabel="Recall", ylabel="Precision", title="Held-out precision–recall"); axes[1].legend()
    cm = confusion_matrix(y, pred, labels=[0, 1])
    axes[2].imshow(cm, cmap="Blues"); axes[2].set(xticks=[0,1], yticks=[0,1], xticklabels=["0", "1"], yticklabels=["0", "1"], xlabel="Predicted", ylabel="Actual", title="Confusion matrix")
    for (i,j), value in np.ndenumerate(cm): axes[2].text(j, i, str(value), ha="center", va="center")
    fig.tight_layout(); fig.savefig(PLOTS / "test_window_diagnostics.png", dpi=160); plt.close(fig)

    # Full timeline includes excluded windows; mark target preictal range, onset,
    # offset and emitted warnings. This makes event timing distinct from window labels.
    test["risk_score"] = test.saved_risk_score
    onset, offset = annotations[str(test.file.iloc[0]).lower()][0]
    alarms = saved.loc[saved.warning_event == 1, "prediction_time"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(test.end, test.risk_score, linewidth=1.2, label="Risk score")
    ax.axhline(threshold, color="tab:red", linestyle="--", label=f"Threshold {threshold:.2f}")
    ax.axvspan(onset-artifact["sph_seconds"]-artifact["prediction_horizon_seconds"], onset-artifact["sph_seconds"], alpha=.15, color="green", label="Documented preictal target interval")
    ax.axvline(onset, color="black", linestyle="--", label="Annotated onset")
    ax.axvline(offset, color="gray", linestyle=":", label="Annotated offset")
    for i, alarm in enumerate(alarms): ax.axvline(alarm, color="orange", alpha=.8, label="Emitted warning" if i == 0 else None)
    ax.set(xlabel="Seconds from recording start", ylabel="Uncalibrated score", title="Held-out event warning timeline", ylim=(0, 1))
    ax.legend(loc="upper right", ncol=2, fontsize=8); fig.tight_layout(); fig.savefig(PLOTS / "test_event_timeline.png", dpi=160); plt.close(fig)
    print(json.dumps({"diagnostics": str(OUT.relative_to(ROOT)), "plots": str(PLOTS.relative_to(ROOT)),
        "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
