import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.models.ga_anfis import GeneticFeatureSelector, SugenoANFIS
from src.models.anfis import ANFIS


class HybridComponentsTest(unittest.TestCase):
    def test_public_anfis_module_uses_canonical_model(self):
        self.assertIs(ANFIS, SugenoANFIS)

    def test_ga_runs_grouped_fitness_and_respects_subset_bounds(self):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(64, 8))
        y = np.tile([0, 1], 32)
        X[:, 3] += y * 1.5
        groups = np.repeat(["r1", "r2", "r3", "r4"], 16)
        ga = GeneticFeatureSelector(population_size=6, generations=3,
            max_features=4, random_state=9, n_estimators=3)
        ga.fit(X, y, groups, [f"f{i}" for i in range(X.shape[1])])
        self.assertGreaterEqual(len(ga.selected_features_), 1)
        self.assertLessEqual(len(ga.selected_features_), 4)
        self.assertTrue(np.isfinite(ga.best_fitness_))
        self.assertLessEqual(len(ga.history_), 3)
        self.assertIn("grouped-CV", ga.config_["fitness"])
        repeat = GeneticFeatureSelector(population_size=6, generations=3,
            max_features=4, random_state=9, n_estimators=3).fit(
                X, y, groups, [f"f{i}" for i in range(X.shape[1])])
        self.assertEqual(ga.selected_features_, repeat.selected_features_)
        self.assertEqual(ga.history_, repeat.history_)

    def test_anfis_trains_rules_handles_invalid_values_and_roundtrips(self):
        rng = np.random.default_rng(23)
        X = rng.normal(size=(80, 2))
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
        model = SugenoANFIS(max_iter=80).fit(X, y)
        self.assertEqual(model.training_["rule_count"], 4)
        self.assertIn("premise and consequent", model.training_["optimizer"])
        strengths, assignments = model._firing(X[:5])
        self.assertEqual(strengths.shape, (5, 4))
        self.assertEqual(assignments.shape, (4, 2))
        self.assertTrue(np.allclose(strengths.sum(axis=1), 1.0))
        self.assertTrue(np.all(strengths >= 0.0))
        membership = np.exp(-0.5 * ((X[:5, 0, None] - model.centers_[:, 0]) /
                                    model.widths_[0]) ** 2)
        self.assertTrue(np.all((membership >= 0.0) & (membership <= 1.0)))
        output = model.predict_proba(np.array([[np.nan, np.inf], [-np.inf, 0.0]]))
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(np.all((output >= 0) & (output <= 1)))
        self.assertTrue(np.allclose(output.sum(axis=1), 1.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            self.assertTrue(np.allclose(output, restored.predict_proba(
                np.array([[np.nan, np.inf], [-np.inf, 0.0]]))))


if __name__ == "__main__":
    unittest.main()
