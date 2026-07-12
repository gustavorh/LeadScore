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
from src.data import features  # noqa: E402


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


def _save_attention_heatmap(model, sessions_test, vocab: dict, device: str) -> None:
    """Guarda el heatmap de atención de la última capa para 2-3 ejemplos (§5.3)."""
    from src.data import sequence

    examples = sessions_test[sessions_test["converted"] == 1].head(3)
    if examples.empty:
        examples = sessions_test.head(3)
    ds = sequence.SessionDataset(examples, vocab["delta_mean"], vocab["delta_std"])
    tok, dlt, _, pad_mask, _ = sequence.collate([ds[i] for i in range(len(ds))])
    attn = model.attention(tok.to(device), dlt.to(device), pad_mask.to(device)).cpu()

    label = {1: "view", 2: "cart"}
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    axes = np.atleast_1d(axes)
    for ax, (_, row), i in zip(axes, examples.iterrows(), range(n)):
        length = int(row["length"])
        a = attn[i, :length, :length]
        ax.imshow(a, cmap="viridis")
        labels = [label.get(t, "?") for t in row["event_types"][:length]]
        ax.set_xticks(range(length), labels=labels, rotation=90, fontsize=7)
        ax.set_yticks(range(length), labels=labels, fontsize=7)
        ax.set_title(f"sesión {row['session_id']} (conv={row['converted']})", fontsize=9)
    fig.suptitle("Atención — última capa del Transformer")
    fig.tight_layout()
    fig.savefig(config.RESULTS / "attention_heatmap.png", dpi=120)
    plt.close(fig)


def _evaluate_retailrocket() -> dict:
    import torch

    from src.data import sequence
    from src.models.gru import GRUClassifier
    from src.models.transformer import TransformerClassifier

    config.ensure_dirs()
    device = "cpu"  # inferencia en CPU (coincide con la imagen de la API)
    vocab = json.loads((config.ARTIFACTS / "vocab.json").read_text())

    sessions = pd.read_parquet(config.DATA_PROCESSED / "sessions.parquet")
    test = sessions[sessions["split"] == "test"].reset_index(drop=True)
    y_true = test["converted"].to_numpy()
    models_metrics: dict[str, dict] = {}

    # --- Baseline tabular ---
    x_tab = features.tabular_from_sessions(test).to_numpy(dtype=float)
    scaler = joblib.load(config.ARTIFACTS / "scaler.joblib")
    base = joblib.load(config.ARTIFACTS / "baseline.joblib")
    p_base = base.predict_proba(scaler.transform(x_tab))[:, 1]
    models_metrics["baseline"] = compute_metrics(y_true, p_base)
    _save_confusion(y_true, (p_base >= 0.5).astype(int), "baseline")

    # --- Modelos secuenciales ---
    test_ds = sequence.SessionDataset(test, vocab["delta_mean"], vocab["delta_std"])
    test_loader = sequence.make_loader(test_ds, config.BATCH_SIZE, shuffle=False)
    for name, model in (("gru", GRUClassifier()), ("transformer", TransformerClassifier())):
        model.load_state_dict(torch.load(config.ARTIFACTS / f"{name}.pt", map_location=device))
        model.to(device)
        prob = sequence.predict_proba(model, test_loader, device)
        models_metrics[name] = compute_metrics(y_true, prob)
        _save_confusion(y_true, (prob >= 0.5).astype(int), name)
        if name == "transformer":
            _save_attention_heatmap(model, test, vocab, device)

    return {"dataset": "retailrocket", "n_test": int(len(test)), "models": models_metrics}


def main() -> None:
    meta = json.loads((config.ARTIFACTS / "metadata.json").read_text())
    if meta.get("dataset") == "retailrocket":
        results = _evaluate_retailrocket()
    else:
        results = _evaluate_uci()

    (config.RESULTS / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    print(f"[evaluate] {results['dataset']} (n_test={results['n_test']}):")
    for name, m in results["models"].items():
        print(f"    {name:12s} F1={m['f1']:.4f} AUC={m['auc']:.4f} acc={m['accuracy']:.4f}")
    print(f"    → {config.RESULTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
