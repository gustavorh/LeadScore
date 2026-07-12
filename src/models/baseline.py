"""Baseline supervisado clásico (§5.1).

Entrena LogisticRegression y RandomForest sobre features tabulares
estandarizadas, elige el mejor por F1 de validación y expone la importancia
de variables para la sección de interpretación.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.config import SEED


def _build_models() -> dict[str, Any]:
    return {
        "logreg": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
    }


def _importances(model: Any, feature_names: list[str]) -> dict[str, float]:
    """Importancia por variable: |coef| para LogReg, feature_importances_ para RF."""
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:  # LogisticRegression
        values = np.abs(model.coef_.ravel())
    return {name: float(v) for name, v in zip(feature_names, values)}


def train_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """Entrena ambos candidatos y devuelve el mejor por F1 de validación.

    Retorna un dict con: `name`, `model`, `f1`, `all_f1` e `importances`.
    """
    all_f1: dict[str, float] = {}
    fitted: dict[str, Any] = {}
    for name, model in _build_models().items():
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        all_f1[name] = float(f1_score(y_val, preds))
        fitted[name] = model

    best_name = max(all_f1, key=all_f1.__getitem__)
    best_model = fitted[best_name]
    return {
        "name": best_name,
        "model": best_model,
        "f1": all_f1[best_name],
        "all_f1": all_f1,
        "importances": _importances(best_model, feature_names),
    }
