"""Ensamble híbrido: stacking de baseline + GRU + Transformer (§5.5).

`p_final = stacking(p_baseline, p_gru, p_transformer)` con una regresión
logística aprendida en validación (fallback a promedio simple). El umbral de
decisión se calibra en validación maximizando F1. Las bandas de acción
(caliente/tibio/frío) son independientes del umbral binario.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.config import ACTION_HOT, ACTION_WARM, SEED

_MODEL_ORDER: tuple[str, ...] = ("baseline", "gru", "transformer")

_ACTIONS: dict[str, str] = {
    "caliente": "Priorizar contacto comercial; alta intención, no quemar descuento.",
    "tibio": "Ofrecer descuento o recordatorio de carrito.",
    "frío": "No invertir; retargeting de bajo costo.",
}


def _stack(probs: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([np.asarray(probs[m], dtype=float) for m in _MODEL_ORDER])


@dataclass
class HybridEnsemble:
    """Combina las 3 probabilidades y aplica el umbral calibrado."""

    method: str            # "stacking" | "mean"
    coef: list[float]      # pesos por modelo (orden: baseline, gru, transformer)
    intercept: float
    threshold: float

    def combine(self, probs: dict[str, np.ndarray]) -> np.ndarray:
        """Devuelve p_final ∈ [0, 1] para cada sesión."""
        x = _stack(probs)
        if self.method == "stacking":
            z = x @ np.asarray(self.coef) + self.intercept
            return 1.0 / (1.0 + np.exp(-z))
        return x.mean(axis=1)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "coef": list(self.coef),
            "intercept": self.intercept,
            "threshold": self.threshold,
            "model_order": list(_MODEL_ORDER),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HybridEnsemble":
        return cls(
            method=data["method"],
            coef=list(data["coef"]),
            intercept=float(data["intercept"]),
            threshold=float(data["threshold"]),
        )


def calibrate_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    """Umbral que maximiza F1 sobre (y_true, p) barriendo candidatos."""
    candidates = np.unique(np.concatenate([p, [0.5]]))
    best_thr, best_f1 = 0.5, -1.0
    for thr in candidates:
        f1 = f1_score(y_true, (p >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def fit_ensemble(val_probs: dict[str, np.ndarray], y_val: np.ndarray) -> HybridEnsemble:
    """Ajusta el stacking en validación; fallback a promedio si algo falla."""
    x = _stack(val_probs)
    try:
        lr = LogisticRegression(max_iter=1000, random_state=SEED)
        lr.fit(x, y_val)
        ens = HybridEnsemble(
            method="stacking",
            coef=lr.coef_.ravel().tolist(),
            intercept=float(lr.intercept_[0]),
            threshold=0.5,
        )
    except Exception:  # pragma: no cover - fallback defensivo
        ens = HybridEnsemble("mean", [1 / 3] * 3, 0.0, 0.5)

    ens.threshold = calibrate_threshold(y_val, ens.combine(val_probs))
    return ens


def recommend_action(p_final: float, segment: str | None = None) -> tuple[str, str]:
    """Etiqueta (caliente/tibio/frío) y acción recomendada según p_final (§5.5)."""
    if p_final >= ACTION_HOT:
        label = "caliente"
    elif p_final >= ACTION_WARM:
        label = "tibio"
    else:
        label = "frío"
    return label, _ACTIONS[label]
