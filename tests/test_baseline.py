"""Tests del baseline (M1, §5.1)."""

from __future__ import annotations

import numpy as np

from src.models import baseline


def _separable_data(n: int = 300):
    """Dataset binario linealmente separable con 3 features."""
    rng = np.random.default_rng(0)
    x_pos = rng.normal(2.0, 0.5, size=(n // 2, 3))
    x_neg = rng.normal(-2.0, 0.5, size=(n // 2, 3))
    x = np.vstack([x_pos, x_neg])
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    idx = rng.permutation(n)
    return x[idx], y[idx]


def test_train_baseline_returns_best_model():
    x, y = _separable_data()
    result = baseline.train_baseline(
        x[:200], y[:200], x[200:], y[200:], feature_names=["a", "b", "c"]
    )
    assert result["name"] in {"logreg", "random_forest"}
    # El modelo elegido debe poder predecir probabilidades.
    proba = result["model"].predict_proba(x[200:])
    assert proba.shape == (100, 2)
    # En datos separables el F1 de validación debe ser alto.
    assert result["f1"] > 0.9


def test_train_baseline_reports_importances():
    x, y = _separable_data()
    result = baseline.train_baseline(
        x[:200], y[:200], x[200:], y[200:], feature_names=["a", "b", "c"]
    )
    assert len(result["importances"]) == 3
    assert set(result["importances"].keys()) == {"a", "b", "c"}
