"""Small, reproducible GA feature selector and first-order Sugeno ANFIS.

Inputs are scaled features. The ANFIS uses two Gaussian membership functions
per input (low/high), a Cartesian rule base, normalized product firing
strengths, and learned first-order Sugeno consequents. Consequents are fit by
regularized logistic loss; membership centers and widths are initialized from
training data only and then jointly adapted with consequents to keep the
bounded rule system tractable on this small dataset.
"""
from __future__ import annotations

from itertools import product

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


class GeneticFeatureSelector:
    """Binary-chromosome GA evaluated by caller-supplied grouped CV."""

    def __init__(self, population_size=12, generations=6, max_features=4,
                 mutation_rate=0.04, random_state=42, n_estimators=30):
        self.population_size = population_size
        self.generations = generations
        self.max_features = max_features
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.n_estimators = n_estimators

    def fit(self, X, y, groups, feature_names):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        groups = np.asarray(groups)
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        k = min(self.max_features, n_features)
        if len(np.unique(groups)) < 2:
            raise ValueError("GA fitness needs at least two distinct recording groups.")

        def chromosome(indices):
            row = np.zeros(n_features, dtype=np.uint8)
            row[indices] = 1
            return row

        population = [chromosome(rng.choice(n_features,
                            int(rng.integers(1, k + 1)), replace=False))
                      for _ in range(self.population_size)]
        folds = list(GroupKFold(n_splits=min(4, len(np.unique(groups)))).split(X, y, groups))
        cache = {}

        def fitness(bits):
            key = bits.tobytes()
            if key in cache:
                return cache[key]
            chosen = np.flatnonzero(bits)
            if not len(chosen):
                return -1.0
            scores = []
            for train_idx, val_idx in folds:
                if len(np.unique(y[train_idx])) < 2:
                    continue
                clf = RandomForestClassifier(n_estimators=self.n_estimators,
                    class_weight="balanced", max_features="sqrt",
                    min_samples_leaf=2, random_state=self.random_state, n_jobs=1)
                clf.fit(X[train_idx][:, chosen], y[train_idx])
                prob = clf.predict_proba(X[val_idx][:, chosen])[:, 1]
                pred = prob >= 0.5
                # F1 rewards recall without accepting an all-positive detector.
                scores.append(f1_score(y[val_idx], pred, zero_division=0))
            mean_f1 = float(np.mean(scores)) if scores else 0.0
            score = mean_f1 - 0.002 * len(chosen)
            cache[key] = score
            return score

        best_bits, best_score, stale = None, -np.inf, 0
        history = []
        for generation in range(self.generations):
            values = np.asarray([fitness(bits) for bits in population])
            order = np.argsort(values)[::-1]
            if values[order[0]] > best_score:
                best_bits, best_score, stale = population[order[0]].copy(), float(values[order[0]]), 0
            else:
                stale += 1
            history.append({"generation": generation + 1, "fitness": best_score,
                            "selected_count": int(best_bits.sum())})
            if stale >= 3:
                break
            parents = [population[i] for i in order[:max(2, self.population_size // 3)]]
            new_population = [best_bits.copy()]
            while len(new_population) < self.population_size:
                p1, p2 = (parents[rng.integers(len(parents))] for _ in range(2))
                mask = rng.random(n_features) < 0.5
                child = np.where(mask, p1, p2).astype(np.uint8)
                child ^= (rng.random(n_features) < self.mutation_rate).astype(np.uint8)
                ones = np.flatnonzero(child)
                if len(ones) > k:
                    child[rng.choice(ones, len(ones)-k, replace=False)] = 0
                elif len(ones) == 0:
                    zeros = np.flatnonzero(child == 0)
                    child[rng.choice(zeros)] = 1
                new_population.append(child)
            population = new_population

        self.selected_indices_ = np.flatnonzero(best_bits)
        self.selected_features_ = [feature_names[i] for i in self.selected_indices_]
        self.best_fitness_ = best_score
        self.history_ = history
        self.config_ = {"population_size": self.population_size,
            "generations": self.generations, "max_features": k,
            "mutation_rate": self.mutation_rate, "random_state": self.random_state,
            "selection": "uniform sampling from the top third", "crossover": "uniform",
            "stopping": "configured generations or 3 stagnant generations",
            "fitness": "mean grouped-CV window F1 minus 0.002 per feature",
            "grouping": "groups supplied by caller; may be recordings or temporal blocks"}
        return self


class SugenoANFIS:
    """Binary first-order Sugeno ANFIS with stable normalized rule firing."""

    def __init__(self, l2=1e-3, max_iter=250):
        self.l2 = float(l2)
        self.max_iter = int(max_iter)

    def _firing(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        # One binary membership choice per input gives 2^d bounded rules.
        assignments = np.asarray(list(product((0, 1), repeat=d)), dtype=int)
        rule_centers = self.centers_[assignments, np.arange(d)[None, :]]
        log_mu = -0.5 * ((X[:, None, :] - rule_centers[None, :, :]) /
                         self.widths_[None, None, :]) ** 2
        log_strength = np.sum(log_mu, axis=2)
        log_strength -= np.max(log_strength, axis=1, keepdims=True)
        strengths = np.exp(log_strength)
        strengths /= np.maximum(strengths.sum(axis=1, keepdims=True), 1e-12)
        # Sugeno linear consequent for every rule: b + sum_j a_j*x_j.
        return strengths, assignments

    def _basis(self, X):
        strengths, _ = self._firing(X)
        local = np.concatenate([np.ones((len(X), 1)), X], axis=1)
        return (strengths[:, :, None] * local[:, None, :]).reshape(len(X), -1)

    def fit(self, X, y):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or X.shape[1] > 4:
            raise ValueError("SugenoANFIS expects a 2D array with at most 4 selected inputs.")
        if len(np.unique(y)) < 2:
            raise ValueError("ANFIS training requires both classes.")
        self.n_features_in_ = X.shape[1]
        self.centers_ = np.quantile(X, [0.25, 0.75], axis=0)
        self.widths_ = np.maximum(np.std(X, axis=0), 1e-3)
        initial_centers = self.centers_.copy()
        initial_log_widths = np.log(self.widths_)
        assignments = np.asarray(list(product((0, 1), repeat=X.shape[1])), dtype=int)
        n_rules, n_features = len(assignments), X.shape[1]
        local = np.concatenate([np.ones((len(X), 1)), X], axis=1)
        positive = max(float(y.sum()), 1.0)
        negative = max(float(len(y) - y.sum()), 1.0)
        weights = np.where(y > 0, len(y)/(2*positive), len(y)/(2*negative))
        def objective(theta):
            split = 2*n_features
            centers = theta[:split].reshape(2, n_features)
            log_widths = theta[split:split+n_features]
            widths = np.exp(np.clip(log_widths, -5, 5))
            consequents = theta[split+n_features:].reshape(n_rules, n_features+1)
            rule_centers = centers[assignments, np.arange(n_features)[None, :]]
            delta = X[:, None, :] - rule_centers[None, :, :]
            log_rule = -0.5 * np.sum((delta / widths[None, None, :]) ** 2, axis=2)
            log_rule -= np.max(log_rule, axis=1, keepdims=True)
            firing = np.exp(log_rule)
            firing /= np.maximum(firing.sum(axis=1, keepdims=True), 1e-12)
            rule_outputs = local @ consequents.T
            z = np.clip(np.sum(firing * rule_outputs, axis=1), -35, 35)
            loss = np.sum(weights * (np.logaddexp(0, z) - y*z)) / len(y)
            d_z = weights * (expit(z) - y) / len(y)
            grad_consequents = (firing * d_z[:, None]).T @ local + self.l2 * consequents
            d_log_rule = d_z[:, None] * firing * (rule_outputs - z[:, None])
            grad_centers = np.zeros_like(centers)
            grad_log_widths = np.zeros_like(log_widths)
            for rule_idx, membership_ids in enumerate(assignments):
                contrib = d_log_rule[:, rule_idx, None]
                for feature_idx, membership_id in enumerate(membership_ids):
                    diff = delta[:, rule_idx, feature_idx]
                    grad_centers[membership_id, feature_idx] += np.sum(
                        contrib[:, 0] * diff / widths[feature_idx]**2)
                    grad_log_widths[feature_idx] += np.sum(
                        contrib[:, 0] * diff**2 / widths[feature_idx]**2)
            # Small prior keeps premise terms from collapsing on this tiny dataset.
            center_delta = centers - initial_centers
            width_delta = log_widths - initial_log_widths
            loss += 0.005 * (np.sum(center_delta**2) + np.sum(width_delta**2))
            grad_centers += 0.01 * center_delta
            grad_log_widths += 0.01 * width_delta
            # Keep the scalar objective consistent with the L2 gradient above;
            # otherwise L-BFGS line search sees a different function than the
            # analytic gradient and can stop with an abnormal line-search exit.
            loss += 0.5 * self.l2 * np.sum(consequents**2)
            grad = np.concatenate([grad_centers.ravel(), grad_log_widths,
                                   grad_consequents.ravel()])
            return loss, grad
        initial = np.concatenate([self.centers_.ravel(), initial_log_widths,
                                  np.zeros(n_rules * (n_features+1))])
        result = minimize(objective, initial, jac=True,
                          method="L-BFGS-B", options={"maxiter": self.max_iter})
        if not np.all(np.isfinite(result.x)):
            raise ValueError("ANFIS optimization produced invalid parameters.")
        split = 2*n_features
        self.centers_ = result.x[:split].reshape(2, n_features)
        self.widths_ = np.exp(np.clip(result.x[split:split+n_features], -5, 5))
        self.consequents_ = result.x[split+n_features:]
        self.training_ = {"objective": "class-balanced binary cross entropy + L2",
                          "optimizer": "L-BFGS (joint premise and consequent gradient updates)", "iterations": int(result.nit),
                          "converged": bool(result.success), "status_message": str(result.message),
                          "gradient_norm": float(np.linalg.norm(result.jac)), "loss": float(result.fun),
                          "membership_functions_per_feature": 2,
                          "rule_count": 2 ** X.shape[1],
                          "consequent": "first-order linear Sugeno"}
        return self

    def predict_proba(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("ANFIS input feature count does not match training metadata.")
        p = expit(np.clip(self._basis(X) @ self.consequents_, -35, 35))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
