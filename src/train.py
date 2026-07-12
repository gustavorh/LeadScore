"""CLI de entrenamiento: entrena los modelos y guarda artefactos en `artifacts/`.

M1 cubre el baseline sobre UCI. Los modelos secuenciales y el híbrido
(RetailRocket) se añaden en M3/M4.

    python -m src.train --dataset uci
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from src import config
from src.data import features
from src.data.download import download_uci
from src.models import baseline


def _train_uci() -> None:
    """Pipeline completo de baseline sobre el dataset UCI."""
    config.set_seed()
    config.ensure_dirs()

    df = download_uci()
    x, y = features.build_uci_features(df)
    feature_names = list(x.columns)

    splits = features.make_splits(y.to_numpy(), seed=config.SEED)

    x_np = x.to_numpy(dtype=float)
    y_np = y.to_numpy()
    scaler = features.fit_scaler(x_np[splits["train"]])
    x_scaled = scaler.transform(x_np)

    result = baseline.train_baseline(
        x_scaled[splits["train"]], y_np[splits["train"]],
        x_scaled[splits["val"]], y_np[splits["val"]],
        feature_names=feature_names,
    )

    # Persistir dataset procesado + split (mismo test set para evaluate).
    processed = x.copy()
    processed["target"] = y_np
    split_col = pd.Series("train", index=processed.index)
    split_col.iloc[splits["val"]] = "val"
    split_col.iloc[splits["test"]] = "test"
    processed["split"] = split_col.to_numpy()
    processed.to_parquet(config.DATA_PROCESSED / "uci.parquet", index=False)

    # Artefactos que consume la API / evaluate.
    joblib.dump(result["model"], config.ARTIFACTS / "baseline.joblib", compress=3)
    joblib.dump(scaler, config.ARTIFACTS / "scaler.joblib", compress=3)

    metadata = {
        "dataset": "uci",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": config.VERSION,
        "seed": config.SEED,
        "baseline": {
            "best": result["name"],
            "val_f1": result["f1"],
            "all_val_f1": result["all_f1"],
        },
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "splits": {k: int(len(v)) for k, v in splits.items()},
        "importances": result["importances"],
    }
    (config.ARTIFACTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )

    print(
        f"[train] UCI listo. Baseline='{result['name']}' "
        f"val_F1={result['f1']:.4f} | artefactos en {config.ARTIFACTS}"
    )


def _load_sessions() -> "pd.DataFrame":
    """Carga sessions.parquet (lo genera M2) o lo construye si falta."""
    path = config.DATA_PROCESSED / "sessions.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from src.data.sessionize import build_sessions

    return build_sessions()


def _train_retailrocket() -> None:
    """Entrena baseline tabular + GRU + Transformer sobre las sesiones (M3)."""
    import torch

    from src.data import sequence
    from src.models.gru import GRUClassifier
    from src.models.transformer import TransformerClassifier

    config.set_seed()
    config.ensure_dirs()
    # CPU: para estos modelos pequeños el overhead de dispatch de MPS supera al
    # cómputo (empíricamente ~5x más lento). Además coincide con la imagen
    # CPU-only de la API, garantizando que los .pt cargan en el contenedor.
    device = "cpu"

    sessions = features.add_visitor_split(_load_sessions())
    sessions.to_parquet(config.DATA_PROCESSED / "sessions.parquet", index=False)
    train_df = sessions[sessions["split"] == "train"]
    val_df = sessions[sessions["split"] == "val"]

    # --- Baseline tabular sobre sesiones (§4.4/§5.1) ---
    x_tab = features.tabular_from_sessions(sessions)
    feature_names = list(x_tab.columns)
    x_np = x_tab.to_numpy(dtype=float)
    y_np = sessions["converted"].to_numpy()
    tr = (sessions["split"] == "train").to_numpy()
    va = (sessions["split"] == "val").to_numpy()
    scaler = features.fit_scaler(x_np[tr])
    x_scaled = scaler.transform(x_np)
    baseline_res = baseline.train_baseline(
        x_scaled[tr], y_np[tr], x_scaled[va], y_np[va], feature_names=feature_names
    )
    joblib.dump(baseline_res["model"], config.ARTIFACTS / "baseline.joblib", compress=3)
    joblib.dump(scaler, config.ARTIFACTS / "scaler.joblib", compress=3)

    # --- Modelos secuenciales (§5.2/§5.3) ---
    delta_mean, delta_std = sequence.delta_stats(train_df)
    train_sub = sequence.subsample_negatives(train_df)
    pos = int((train_sub["converted"] == 1).sum())
    neg = int((train_sub["converted"] == 0).sum())
    pos_weight = neg / max(pos, 1)

    train_ds = sequence.SessionDataset(train_sub, delta_mean, delta_std)
    val_ds = sequence.SessionDataset(val_df, delta_mean, delta_std)
    train_loader = sequence.make_loader(train_ds, config.BATCH_SIZE, shuffle=True)
    val_loader = sequence.make_loader(val_ds, config.BATCH_SIZE, shuffle=False)

    seq_metrics = {}
    for name, model in (("gru", GRUClassifier()), ("transformer", TransformerClassifier())):
        config.set_seed()  # misma inicialización de datos entre modelos
        info = sequence.fit_sequence_model(
            model, train_loader, val_loader, pos_weight, device, label=name
        )
        model.to("cpu")
        torch.save(model.state_dict(), config.ARTIFACTS / f"{name}.pt")
        seq_metrics[name] = info
        print(f"[train] {name}: val_AUC={info['val_auc']:.4f} ({info['epochs']} épocas)")

    # --- Contratos de artefactos ---
    vocab = {
        "pad": config.PAD_TOKEN,
        "event_vocab": config.EVENT_VOCAB,
        "category_oov": config.CATEGORY_OOV,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "dims": {
            "embed_dim": config.EMBED_DIM,
            "delta_proj_dim": config.DELTA_PROJ_DIM,
            "gru_hidden": config.GRU_HIDDEN,
            "d_model": config.TRANSFORMER_DMODEL,
            "nhead": config.TRANSFORMER_NHEAD,
            "ff": config.TRANSFORMER_FF,
            "layers": config.TRANSFORMER_LAYERS,
            "max_len": config.MAX_LEN,
        },
    }
    (config.ARTIFACTS / "vocab.json").write_text(
        json.dumps(vocab, indent=2, ensure_ascii=False)
    )

    metadata = {
        "dataset": "retailrocket",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": config.VERSION,
        "seed": config.SEED,
        "baseline": {
            "best": baseline_res["name"],
            "val_f1": baseline_res["f1"],
            "all_val_f1": baseline_res["all_f1"],
        },
        "sequential": {k: v["val_auc"] for k, v in seq_metrics.items()},
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "splits": {
            k: int((sessions["split"] == k).sum()) for k in ("train", "val", "test")
        },
        "pos_weight": pos_weight,
        "importances": baseline_res["importances"],
    }
    (config.ARTIFACTS / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    print(f"[train] RetailRocket listo. Artefactos en {config.ARTIFACTS}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena los modelos de LeadScore.")
    parser.add_argument(
        "--dataset", choices=["uci", "retailrocket"], default="uci",
        help="Dataset a entrenar (uci = M1; retailrocket = M3/M4).",
    )
    args = parser.parse_args()

    if args.dataset == "uci":
        _train_uci()
    else:
        _train_retailrocket()


if __name__ == "__main__":
    main()
