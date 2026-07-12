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
        raise NotImplementedError(
            "El entrenamiento sobre RetailRocket se implementa en M3/M4."
        )


if __name__ == "__main__":
    main()
