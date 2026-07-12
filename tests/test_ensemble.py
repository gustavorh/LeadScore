"""Tests del ensamble híbrido (M4, §5.5)."""

from __future__ import annotations

import numpy as np

from src.models import ensemble


def test_recommend_action_bands() -> None:
    assert ensemble.recommend_action(0.85)[0] == "caliente"
    assert ensemble.recommend_action(0.55)[0] == "tibio"
    assert ensemble.recommend_action(0.20)[0] == "frío"
    # cada banda trae una acción no vacía.
    for p in (0.85, 0.55, 0.20):
        assert ensemble.recommend_action(p)[1]


def test_calibrate_threshold_maximizes_f1() -> None:
    # Separación perfecta en 0.5: el umbral óptimo debe caer entre las clases.
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = ensemble.calibrate_threshold(y, p)
    assert 0.3 < thr <= 0.7


def test_combine_returns_probabilities() -> None:
    ens = ensemble.HybridEnsemble("mean", [1 / 3, 1 / 3, 1 / 3], 0.0, 0.5)
    probs = {
        "baseline": np.array([0.2, 0.8]),
        "gru": np.array([0.3, 0.9]),
        "transformer": np.array([0.1, 0.7]),
    }
    out = ens.combine(probs)
    assert out.shape == (2,)
    assert np.all((out >= 0) & (out <= 1))


def test_fit_ensemble_learns_and_sets_threshold() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0] * 100 + [1] * 100)
    # gru muy informativo; los otros ruidosos.
    val_probs = {
        "baseline": rng.uniform(0, 1, 200),
        "gru": np.clip(y * 0.6 + rng.uniform(0, 0.4, 200), 0, 1),
        "transformer": rng.uniform(0, 1, 200),
    }
    ens = ensemble.fit_ensemble(val_probs, y)
    assert ens.method in {"stacking", "mean"}
    assert 0.0 < ens.threshold < 1.0
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, ens.combine(val_probs)) > 0.9


def test_json_roundtrip() -> None:
    ens = ensemble.HybridEnsemble("stacking", [0.5, 1.2, 0.8], -0.3, 0.42)
    restored = ensemble.HybridEnsemble.from_dict(ens.to_dict())
    assert restored.method == ens.method
    assert restored.coef == ens.coef
    assert restored.threshold == ens.threshold
