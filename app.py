from pathlib import Path
import csv
import io
import json
import uuid
from flask import Flask, render_template, request, jsonify, send_file
from inference.predictor import predict_edf, validate_edf, get_default_threshold, get_model_info
from inference.anticipatory_predictor import predict_anticipatory_edf

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
SAMPLE_EDF = BASE_DIR / "sample" / "chb03_36_sample.edf"
CACHE_DIR = BASE_DIR / ".result_cache"
CACHE_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024


def _save_result(result):
    job_id = result["job_id"]
    (CACHE_DIR / f"{job_id}.json").write_text(json.dumps(result))
    return job_id


def _load_result(job_id):
    path = CACHE_DIR / f"{job_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@app.route("/")
def dashboard():
    report_path = BASE_DIR / "results" / "csv" / "ga_anfis_evaluation.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    anticipatory_path = BASE_DIR / "results" / "csv" / "anticipatory_featureaudit_v4_evaluation.json"
    anticipatory = json.loads(anticipatory_path.read_text()) if anticipatory_path.exists() else None
    try:
        model_info = get_model_info()
    except (FileNotFoundError, ValueError):
        model_info = None
    return render_template("dashboard.html", evaluation=report,
        anticipatory_evaluation=anticipatory, model_info=model_info)


@app.route("/predict", methods=["GET"])
def predict_page():
    try:
        default_threshold = get_default_threshold()
        model_error = None
    except (FileNotFoundError, ValueError) as exc:
        default_threshold = 0.50
        model_error = str(exc)
    return render_template("predict.html", default_threshold=default_threshold, model_error=model_error)


@app.route("/anticipate")
def anticipate_page():
    try:
        from inference.anticipatory_predictor import load_artifact
        artifact = load_artifact()
        model_info = {"id": artifact.get("model_id"), "threshold": artifact["threshold"],
            "window": artifact.get("window_seconds", 5), "step": artifact.get("step_seconds", 5),
            "horizon": artifact.get("prediction_horizon_seconds", 1200) / 60,
            "sph": artifact.get("sph_seconds", 300) / 60,
            "confirm": artifact.get("confirm_windows", 3),
            "refractory": artifact.get("alarm_refractory_seconds", 1800) / 60}
        model_error = None
    except (FileNotFoundError, ValueError) as exc:
        model_info, model_error = None, str(exc)
    return render_template("anticipate.html", model_info=model_info, model_error=model_error)


@app.route("/api/anticipate", methods=["POST"])
def api_anticipate():
    uploaded = request.files.get("edf_file")
    use_sample = request.form.get("use_sample") == "1"
    if use_sample:
        if not SAMPLE_EDF.exists():
            return jsonify(ok=False, error="Sample EDF is not bundled. Add sample/chb03_36_sample.edf."), 400
        path = SAMPLE_EDF
    else:
        if not uploaded or not uploaded.filename:
            return jsonify(ok=False, error="Please select an EDF file or use the bundled sample."), 400
        if not uploaded.filename.lower().endswith(".edf"):
            return jsonify(ok=False, error="Only .edf files are supported."), 400
        path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{Path(uploaded.filename).name}"
    try:
        if not use_sample:
            uploaded.save(path)
        threshold_raw = request.form.get("threshold")
        result = predict_anticipatory_edf(path, float(threshold_raw) if threshold_raw else None)
        result["source"] = "sample" if use_sample else "upload"
        _save_result(result)
        return jsonify(ok=True, result_url=f"/anticipatory-result/{result['job_id']}")
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    except FileNotFoundError as exc:
        return jsonify(ok=False, error=str(exc)), 503
    except Exception:
        return jsonify(ok=False, error="Anticipatory analysis failed. Check that the EDF is readable and the trained model artifacts are compatible."), 400
    finally:
        if not use_sample:
            path.unlink(missing_ok=True)


@app.route("/anticipatory-result/<job_id>")
def anticipatory_result(job_id):
    data = _load_result(job_id)
    if not data or "prediction_horizon_seconds" not in data:
        return render_template("anticipate.html", error="This result has expired or could not be found."), 404
    return render_template("anticipatory_result.html", result=data)


@app.route("/api/validate", methods=["POST"])
def api_validate():
    uploaded = request.files.get("edf_file")
    if not uploaded or not uploaded.filename:
        return jsonify(ok=False, error="Please select an EDF file."), 400
    if not uploaded.filename.lower().endswith(".edf"):
        return jsonify(ok=False, error="Only .edf files are supported."), 400
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{Path(uploaded.filename).name}"
    try:
        uploaded.save(path)
        info = validate_edf(path)
        return jsonify(ok=True, filename=Path(uploaded.filename).name, size_bytes=path.stat().st_size, **info)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    except FileNotFoundError as exc:
        return jsonify(ok=False, error=str(exc)), 503
    except Exception:
        return jsonify(
            ok=False,
            error="NeuroGuard could not read this EDF recording. Please try a standard EDF/EDF+ file with a compatible EEG channel layout."
        ), 400
    finally:
        path.unlink(missing_ok=True)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    uploaded = request.files.get("edf_file")
    use_sample = request.form.get("use_sample") == "1"
    threshold = request.form.get("threshold")
    merge_gap = request.form.get("merge_gap", "30")
    min_duration = request.form.get("min_duration", "10")

    if use_sample:
        if not SAMPLE_EDF.exists():
            return jsonify(ok=False, error="Sample EDF is not bundled. Add sample/chb03_36_sample.edf."), 400
        path = SAMPLE_EDF
        cleanup = False
    else:
        if not uploaded or not uploaded.filename:
            return jsonify(ok=False, error="Please select an EDF file."), 400
        if not uploaded.filename.lower().endswith(".edf"):
            return jsonify(ok=False, error="Only .edf files are supported."), 400
        path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{Path(uploaded.filename).name}"
        uploaded.save(path)
        cleanup = True

    try:
        result = predict_edf(path, float(threshold) if threshold else None,
                             float(merge_gap), float(min_duration))
        result["source"] = "sample" if use_sample else "upload"
        _save_result(result)
        return jsonify(ok=True, result=result)
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    except FileNotFoundError as exc:
        return jsonify(ok=False, error=str(exc)), 503
    except Exception:
        return jsonify(
            ok=False,
            error="NeuroGuard analysis failed. Check that the EDF is readable and the GA–ANFIS model artifacts are compatible."
        ), 400
    finally:
        if cleanup:
            path.unlink(missing_ok=True)


@app.route("/result/<job_id>")
def result(job_id):
    data = _load_result(job_id)
    if not data:
        return render_template("predict.html", error="This result has expired or could not be found."), 404
    return render_template("result.html", result=data)


@app.route("/download/<job_id>.csv")
def download_csv(job_id):
    result = _load_result(job_id)
    if not result:
        return "Result not found", 404
    out = io.StringIO()
    writer = csv.writer(out)
    if "prediction_horizon_seconds" in result:
        writer.writerow(["window_start", "window_end", "prediction_time", "risk_score", "prediction", "warning_active", "warning_event", "warning_generated_at", "future_risk_start", "future_risk_end"])
        for row in result["windows"]:
            writer.writerow([row["start"], row["end"], row["prediction_time"], row["risk_score"], row["prediction"], row["warning_active"], row["warning_event"], row.get("warning_generated_at"), row.get("future_risk_start"), row.get("future_risk_end")])
        suffix = "anticipatory_risk"
    else:
        writer.writerow(["window_start", "window_end", "score", "prediction"])
        for row in result["windows"]:
            writer.writerow([row["start"], row["end"], row["score"], row["prediction"]])
        suffix = "neuroguard_predictions"
    return send_file(io.BytesIO(out.getvalue().encode()), mimetype="text/csv", as_attachment=True, download_name=f"{Path(result['filename']).stem}_{suffix}.csv")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)
