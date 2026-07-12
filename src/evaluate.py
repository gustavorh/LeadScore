"""CLI de evaluación: métricas comparativas sobre el test set → results/.

M1 evalúa el baseline sobre el test set persistido de UCI. Se amplía en M3/M4
con GRU, Transformer e híbrido (mismo test set, §10).

    python -m src.evaluate
"""

from __future__ import annotations

import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")  # sin display; guardamos PNG
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Calcula accuracy, precision, recall, F1 y AUC."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


def _save_confusion(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["no conv.", "conv."])
    ax.set_yticks([0, 1], labels=["no conv.", "conv."])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(config.RESULTS / f"confusion_{name}.png", dpi=120)
    plt.close(fig)


def _evaluate_uci() -> dict:
    config.ensure_dirs()
    df = pd.read_parquet(config.DATA_PROCESSED / "uci.parquet")
    test = df[df["split"] == "test"]
    y_true = test["target"].to_numpy()
    x_test = test.drop(columns=["target", "split"]).to_numpy(dtype=float)

    scaler = joblib.load(config.ARTIFACTS / "scaler.joblib")
    model = joblib.load(config.ARTIFACTS / "baseline.joblib")
    y_prob = model.predict_proba(scaler.transform(x_test))[:, 1]

    metrics = compute_metrics(y_true, y_prob)
    _save_confusion(y_true, (y_prob >= 0.5).astype(int), "baseline")

    return {
        "dataset": "uci",
        "n_test": int(len(test)),
        "models": {"baseline": metrics},
    }


def main() -> None:
    results = _evaluate_uci()
    (config.RESULTS / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    m = results["models"]["baseline"]
    print(
        f"[evaluate] baseline UCI (n_test={results['n_test']}): "
        f"F1={m['f1']:.4f} AUC={m['auc']:.4f} acc={m['accuracy']:.4f} "
        f"→ {config.RESULTS / 'metrics.json'}"
    )


if __name__ == "__main__":
    main()
