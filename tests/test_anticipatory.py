import tempfile
import unittest
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import mne
import pandas as pd
import joblib

from src.preprocessing.anticipatory_pipeline import (
    causal_filter, feature_row, feature_catalog, parse_summary, prediction_labels,
    _channel_indices, causal_resample, BANDS, TARGET_SFREQ, FEATURE_SCHEMA_VERSION,
    PREDICTION_HORIZON_SEC, SPH_SEC,
)
from inference.anticipatory_predictor import (generate_alarms, validate_anticipatory_edf,
    load_artifact)
from src.models.anfis import SugenoANFIS
from sklearn.preprocessing import StandardScaler
from src.models.train_anticipatory import event_evaluation
from src.models.train_anticipatory import TRAIN, VALIDATION, TEST


class AnticipatoryProtocolTests(unittest.TestCase):
    def test_dashboard_loads_separate_v4_prediction_evaluation(self):
        from app import app
        response = app.test_client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"EXPERIMENTAL FUTURE-RISK PREDICTION", response.data)
        self.assertIn(b"GA-selected inputs", response.data)
        self.assertIn("30–80".encode(), response.data)

    def test_summary_annotations_are_recording_relative_and_correct(self):
        annotations = parse_summary()
        self.assertEqual(annotations["chb03_34.edf"], [(1982.0, 2029.0)])
        self.assertEqual(annotations["chb03_36.edf"], [(1725.0, 1778.0)])

    def test_real_edf_opens_and_has_expected_sampling_and_montage(self):
        path = Path("data/raw/chb03_36.edf")
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        try:
            self.assertEqual(raw.info["sfreq"], 256.0)
            self.assertEqual(len(raw.ch_names), 23)
            self.assertEqual(len(raw.get_data(start=0, stop=256)), 23)
        finally:
            raw.close()

    def test_anticipatory_inference_checks_layout_and_rejects_synthetic_generic_channels(self):
        info = validate_anticipatory_edf(Path("data/raw/chb03_36.edf"))
        self.assertTrue(info["compatible"])
        with self.assertRaisesRegex(ValueError, "channel layout mismatch"):
            validate_anticipatory_edf(Path("sample/chb03_36_sample.edf"))

    def test_preictal_boundary_sph_and_ictal_never_positive(self):
        onset, offset = 1982.0, 2029.0
        start = onset - SPH_SEC - PREDICTION_HORIZON_SEC
        end = onset - SPH_SEC
        self.assertEqual(prediction_labels(start, start + 5, [(onset, offset)]), 1)
        self.assertEqual(prediction_labels(end - 5, end, [(onset, offset)]), 1)
        self.assertIsNone(prediction_labels(end, end + 5, [(onset, offset)]))
        self.assertIsNone(prediction_labels(onset, onset + 5, [(onset, offset)]))
        self.assertIsNone(prediction_labels(offset + 5, offset + 10, [(onset, offset)]))

    def test_short_history_event_is_not_fabricated_as_preictal(self):
        self.assertIsNone(prediction_labels(0, 5, [(362, 414)]))
        self.assertEqual(prediction_labels(0, 5, []), 0)

    def test_recording_split_has_no_file_overlap(self):
        self.assertTrue(TRAIN.isdisjoint(VALIDATION))
        self.assertTrue(TRAIN.isdisjoint(TEST))
        self.assertTrue(VALIDATION.isdisjoint(TEST))
        self.assertEqual(TEST, {"chb03_36.edf"})

    def test_window_alignment_and_features(self):
        sf = 256
        signal = np.vstack([np.sin(np.arange(sf * 5) * (2 * np.pi * 10 / sf))] * 23)
        values = feature_row(signal, sf)
        self.assertEqual(len(values), 345)
        self.assertTrue(np.isfinite(list(values.values())).all())
        self.assertGreater(values["ch0_alpha_power"], values["ch0_gamma_power"])
        relative = [values[f"ch0_{band}_relative_power"] for band in BANDS]
        self.assertAlmostEqual(sum(relative), 1.0, places=6)
        self.assertAlmostEqual(values["ch0_alpha_relative_power"], 1.0, places=2)
        self.assertEqual(BANDS["gamma"], (30.0, 80.0))

    def test_each_documented_band_has_a_distinct_candidate_and_dominant_tone(self):
        bounds = list(BANDS.values())
        self.assertEqual(bounds[0][0], 0.5)
        self.assertEqual(bounds[-1][1], 80.0)
        self.assertTrue(all(bounds[i][1] == bounds[i + 1][0]
                            for i in range(len(bounds) - 1)))
        self.assertEqual(len(BANDS), len(set(BANDS)))
        t = np.arange(256 * 5) / 256
        for band, (low, high) in BANDS.items():
            frequency = (low + high) / 2
            wave = np.sin(2 * np.pi * frequency * t)
            row = feature_row(np.tile(wave, (23, 1)), 256)
            powers = {name: row[f"ch0_{name}_power"] for name in BANDS}
            self.assertEqual(max(powers, key=powers.get), band)

    def test_feature_catalog_maps_each_channel_and_documents_units(self):
        names = ["FP1-F7", "F7-T7"]
        catalog = feature_catalog(names)
        self.assertEqual(len(catalog), 30)
        row = catalog[catalog.feature == "ch1_theta_relative_power"].iloc[0]
        self.assertEqual(row.channel_name, "F7-T7")
        self.assertEqual(row.frequency_range_hz, "4-8")
        self.assertEqual(row.unit, "unitless fraction")

    def test_inference_reorders_channels_by_name_and_rejects_ambiguous_mapping(self):
        self.assertEqual(_channel_indices(["B", "A"], ["A", "B"]), [1, 0])
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _channel_indices(["A", "a"], ["A", "B"])

    def test_saved_feature_list_model_dimensions_and_roundtrip_predictions(self):
        rng = np.random.default_rng(17)
        X = rng.normal(size=(40, 2))
        y = (X[:, 0] > 0).astype(int)
        scaler = StandardScaler().fit(X)
        model = SugenoANFIS(max_iter=40).fit(scaler.transform(X), y)
        artifact = {"format_version": 2, "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "model": model, "scaler": scaler, "selected_features": ["ch0_std", "ch1_alpha_power"],
            "n_features_in": 2, "feature_names": ["ch0_std", "ch1_alpha_power", "ch1_theta_power"],
            "channel_names": ["FP1-F7", "F7-T7"], "sampling_frequency_hz": TARGET_SFREQ,
            "threshold": .5, "prediction_horizon_seconds": 1200, "sph_seconds": 300}
        expected = model.predict_proba(scaler.transform(X[:5]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.joblib"
            joblib.dump(artifact, path)
            with patch("inference.anticipatory_predictor.MODEL_PATH", path):
                loaded = load_artifact()
            actual = loaded["model"].predict_proba(loaded["scaler"].transform(X[:5]))
        self.assertEqual(len(artifact["selected_features"]), model.n_features_in_)
        self.assertTrue(np.allclose(expected, actual))

    def test_trained_artifact_features_exist_in_inference_schema(self):
        artifact = load_artifact()
        signal = np.zeros((len(artifact["channel_names"]),
                           int(artifact["window_seconds"] * artifact["sampling_frequency_hz"])))
        available = feature_row(signal, artifact["sampling_frequency_hz"])
        self.assertTrue(set(artifact["selected_features"]).issubset(available))
        self.assertEqual(len(artifact["selected_features"]), artifact["model"].n_features_in_)
        self.assertTrue(artifact["model"].training_["converged"])
        from src.preprocessing.anticipatory_pipeline import FEATURE_PATH
        frame = pd.read_csv(FEATURE_PATH)
        train = frame[(frame.file.isin(TRAIN)) & (frame.eligible == 1)]
        expected_means = train[artifact["selected_features"]].mean().to_numpy()
        self.assertTrue(np.allclose(artifact["scaler"].mean_, expected_means))

    def test_causal_filter_prefix_does_not_depend_on_future(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(2, 2000))
        full_prefix = causal_filter(x, 256)[:, :1000]
        short_prefix = causal_filter(x[:, :1000], 256)
        self.assertTrue(np.array_equal(full_prefix, short_prefix))

    def test_causal_resampling_supports_input_rate_difference_without_future_dependency(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(2, 1000))
        full = causal_resample(x, 250.0)
        prefix = causal_resample(x[:, :500], 250.0)
        self.assertEqual(full.shape[-1], 1024)
        self.assertTrue(np.array_equal(full[:, :prefix.shape[-1]], prefix))
        with self.assertRaisesRegex(ValueError, "177.78"):
            causal_resample(x, 170.0)

    def test_alarm_confirmation_and_refractory(self):
        times = np.arange(0, 3600, 5)
        scores = np.zeros(len(times))
        scores[2:8] = 0.8
        scores[20:23] = 0.8
        scores[370:373] = 0.8
        alarms = generate_alarms(times, scores, 0.5, confirm_windows=3, refractory_sec=1800)
        self.assertEqual(alarms, [20.0, 1860.0])

    def test_alarm_streak_resets_across_excluded_time_gaps(self):
        alarms = generate_alarms([0, 5, 10, 100, 105, 110], [0.9] * 6,
                                 0.5, confirm_windows=3, refractory_sec=0)
        self.assertEqual(alarms, [10.0, 110.0])

    def test_event_matching_uses_future_sph_and_horizon_interval(self):
        times = np.arange(5, 3600, 5, dtype=float)
        frame = pd.DataFrame({"file": "chb03_36.edf", "start": times - 5,
            "end": times, "prediction_time": times, "label": 0})
        scores = np.zeros(len(frame))
        scores[(times >= 535) & (times <= 545)] = 0.99
        result = event_evaluation(frame, scores, 0.9)
        self.assertEqual(result["successfully_predicted_events"], 1)
        self.assertEqual(result["warning_lead_times_seconds"], [1180.0])

    def test_event_evaluation_counts_live_alarm_on_unlabeled_full_timeline(self):
        times = np.arange(5, 3600, 5, dtype=float)
        eligible = np.ones(len(times), dtype=int)
        eligible[(times >= 1990) & (times <= 2010)] = 0
        frame = pd.DataFrame({"file": "chb03_36.edf", "start": times - 5,
            "end": times, "prediction_time": times, "label": np.where(eligible, 0, np.nan),
            "eligible": eligible})
        scores = np.zeros(len(frame))
        scores[(times >= 2000) & (times <= 2010)] = .99
        result = event_evaluation(frame, scores, .9)
        self.assertEqual(result["false_alarm_count"], 1)
        self.assertGreater(result["total_monitoring_hours"], result["eligible_monitoring_hours"])

    def test_flask_prediction_endpoint_uses_model_output_and_csv_fields(self):
        from app import app
        mocked = {"job_id": "unit-test-anticipatory", "filename": "sample.edf",
            "model_id": "test model", "prediction_horizon_seconds": 1200,
            "sph_seconds": 300, "threshold": 0.5, "validated_threshold": 0.5,
            "threshold_source": "validation-selected artifact threshold",
            "window_sec": 5, "step_sec": 5, "alarm_refractory_seconds": 1800,
            "peak_score": 0.8, "warning_active_at_end": False,
            "warning_count": 1, "warning_events": [], "windows_analyzed": 1,
            "duration_sec": 5, "channels": 23, "sampling_frequency": 256,
            "windows": [{"start": 0, "end": 5, "prediction_time": 5,
                "risk_score": .8, "prediction": 1, "warning_active": 0,
                "warning_event": 1}]}
        app.config["TESTING"] = True
        with patch("app.predict_anticipatory_edf", return_value=mocked) as predictor:
            response = app.test_client().post("/api/anticipate", data={
                "edf_file": (io.BytesIO(b"test"), "sample.edf"),
                "threshold": "0.94"})
        predictor.assert_called_once()
        self.assertEqual(predictor.call_args.args[1], 0.94)
        self.assertEqual(response.status_code, 200)
        url = response.get_json()["result_url"]
        result_page = app.test_client().get(url)
        self.assertEqual(result_page.status_code, 200)
        csv_response = app.test_client().get("/download/unit-test-anticipatory.csv")
        self.assertIn(b"prediction_time,risk_score,prediction,warning_active,warning_event,warning_generated_at,future_risk_start,future_risk_end", csv_response.data)
        from app import CACHE_DIR
        (CACHE_DIR / "unit-test-anticipatory.json").unlink(missing_ok=True)

    def test_real_edf_upload_runs_active_anticipatory_model_and_csv_export(self):
        from app import app
        source = Path("data/raw/chb03_36.edf")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "uploads").mkdir()
            (root / "cache").mkdir()
            app.config["TESTING"] = True
            with patch("app.UPLOAD_DIR", root / "uploads"), patch("app.CACHE_DIR", root / "cache"):
                with source.open("rb") as edf:
                    response = app.test_client().post("/api/anticipate", data={
                        "edf_file": (edf, source.name)}, content_type="multipart/form-data")
                self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
                result_url = response.json["result_url"]
                page = app.test_client().get(result_url)
                self.assertEqual(page.status_code, 200)
                self.assertIn(b"Prediction horizon", page.data)
                csv_url = result_url.replace("/anticipatory-result/", "/download/") + ".csv"
                csv_response = app.test_client().get(csv_url)
                self.assertEqual(csv_response.status_code, 200)
                lines = csv_response.get_data(as_text=True).splitlines()
                self.assertEqual(len(lines), 721)
                self.assertIn("risk_score", lines[0])

    def test_anticipatory_upload_rejects_bad_files_and_missing_model_without_fallback(self):
        from app import app
        import inference.anticipatory_predictor as anticipatory
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "uploads").mkdir()
            (root / "cache").mkdir()
            app.config["TESTING"] = True
            client = app.test_client()
            with patch("app.UPLOAD_DIR", root / "uploads"), patch("app.CACHE_DIR", root / "cache"):
                wrong_suffix = client.post("/api/anticipate", data={
                    "edf_file": (io.BytesIO(b"not an EDF"), "recording.txt")})
                self.assertEqual(wrong_suffix.status_code, 400)
                malformed = client.post("/api/anticipate", data={
                    "edf_file": (io.BytesIO(b"not an EDF"), "recording.edf")})
                self.assertEqual(malformed.status_code, 400)
                with patch.object(anticipatory, "MODEL_PATH", root / "missing.joblib"):
                    missing_model = client.post("/api/anticipate", data={
                        "edf_file": (io.BytesIO(b"not an EDF"), "recording.edf")})
                self.assertEqual(missing_model.status_code, 503)
                self.assertIn(b"not trained", missing_model.data)


if __name__ == "__main__":
    unittest.main()
