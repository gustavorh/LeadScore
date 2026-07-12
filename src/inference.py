"""Inferencia reutilizable: carga de artefactos y scoring de sesiones/leads.

La API (M5) es un wrapper HTTP sobre estas funciones. Reutiliza el mismo código
de features y encoding que el entrenamiento para evitar train/serve skew.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src import config
from src.data import features, sequence
from src.models import clustering
from src.models.ensemble import HybridEnsemble, recommend_action
from src.models.gru import GRUClassifier
from src.models.transformer import TransformerClassifier


@dataclass
class Artifacts:
    baseline: object
    scaler: object
    kmeans: object
    gru: torch.nn.Module
    transformer: torch.nn.Module
    ensemble: HybridEnsemble
    vocab: dict
    segment_names: dict[int, str]
    metadata: dict


def load_artifacts(path: Path = config.ARTIFACTS) -> Artifacts:
    """Carga todos los artefactos entrenados desde `artifacts/`."""
    vocab = json.loads((path / "vocab.json").read_text())
    metadata = json.loads((path / "metadata.json").read_text())
    gru = GRUClassifier()
    gru.load_state_dict(torch.load(path / "gru.pt", map_location="cpu"))
    gru.eval()
    transformer = TransformerClassifier()
    transformer.load_state_dict(torch.load(path / "transformer.pt", map_location="cpu"))
    transformer.eval()
    kmeans = joblib.load(path / "kmeans.joblib")
    return Artifacts(
        baseline=joblib.load(path / "baseline.joblib"),
        scaler=joblib.load(path / "scaler.joblib"),
        kmeans=kmeans,
        gru=gru,
        transformer=transformer,
        ensemble=HybridEnsemble.from_dict(
            json.loads((path / "ensemble.json").read_text())
        ),
        vocab=vocab,
        segment_names=clustering.name_clusters(
            kmeans.cluster_centers_, metadata["feature_names"]
        ),
        metadata=metadata,
    )


def _sequence_probs(art: Artifacts, event_types: list[str], seconds: list[float]):
    """Probabilidades de GRU y Transformer para una sola sesión."""
    tokens = np.array([art.vocab["event_vocab"][t] for t in event_types], dtype="int64")
    dlog = np.log1p(np.asarray(seconds, dtype="float64"))
    deltas = ((dlog - art.vocab["delta_mean"]) / art.vocab["delta_std"]).astype("float32")
    tok, dlt, lengths, pad_mask, _ = sequence.collate([(tokens, deltas, 0.0)])
    with torch.no_grad():
        p_gru = float(torch.sigmoid(art.gru(tok, dlt, lengths, pad_mask)).item())
        p_trans = float(torch.sigmoid(art.transformer(tok, dlt, lengths, pad_mask)).item())
    return p_gru, p_trans


def score_session(
    art: Artifacts, events: list[dict], hour_of_day: int, day_of_week: int
) -> dict:
    """Scorea una sesión (secuencia de eventos) → respuesta del contrato §6."""
    types = [e["type"] for e in events]
    cats = [e.get("item_category", "<oov>") for e in events]
    seconds = [float(e["seconds_since_prev"]) for e in events]

    p_gru, p_trans = _sequence_probs(art, types, seconds)
    x_tab = features.tabular_from_events(types, cats, seconds, hour_of_day, day_of_week)
    x_scaled = art.scaler.transform(x_tab.to_numpy(dtype=float))
    p_base = float(art.baseline.predict_proba(x_scaled)[0, 1])
    segment = art.segment_names[int(art.kmeans.predict(x_scaled)[0])]

    probs = {
        "baseline": np.array([p_base]),
        "gru": np.array([p_gru]),
        "transformer": np.array([p_trans]),
    }
    p_final = float(art.ensemble.combine(probs)[0])
    label, action = recommend_action(p_final, segment)
    return {
        "conversion_probability": round(p_final, 4),
        "label": label,
        "recommended_action": action,
        "segment": segment,
        "model_breakdown": {
            "baseline": round(p_base, 4),
            "gru": round(p_gru, 4),
            "transformer": round(p_trans, 4),
            "ensemble": round(p_final, 4),
        },
        "threshold": round(art.ensemble.threshold, 4),
    }


def score_batch(art: Artifacts, df: pd.DataFrame) -> list[dict]:
    """Scorea un CSV de leads con solo el baseline tabular + segmento (§6).

    Deriva n_events, mean_delta_t y addtocart_rate desde las columnas del CSV.
    Devuelve la lista ordenada por probabilidad descendente.
    """
    n_events = (df["n_views"] + df["n_addtocart"]).astype(float).clip(lower=1)
    tab = pd.DataFrame(
        {
            "n_events": n_events,
            "n_views": df["n_views"].astype(float),
            "n_addtocart": df["n_addtocart"].astype(float),
            "n_unique_items": df["n_unique_items"].astype(float),
            "duration_sec": df["duration_sec"].astype(float),
            "mean_delta_t": df["duration_sec"].astype(float) / n_events,
            "hour_of_day": df["hour_of_day"].astype(float),
            "day_of_week": df["day_of_week"].astype(float),
            "addtocart_rate": df["n_addtocart"].astype(float) / n_events,
        }
    )[config.TABULAR_FEATURES]

    x_scaled = art.scaler.transform(tab.to_numpy(dtype=float))
    p_base = art.baseline.predict_proba(x_scaled)[:, 1]
    seg_ids = art.kmeans.predict(x_scaled)

    rows = []
    for i, lead_id in enumerate(df["lead_id"]):
        segment = art.segment_names[int(seg_ids[i])]
        label, action = recommend_action(float(p_base[i]), segment)
        rows.append(
            {
                "lead_id": lead_id,
                "probability": round(float(p_base[i]), 4),
                "label": label,
                "segment": segment,
                "recommended_action": action,
            }
        )
    return sorted(rows, key=lambda r: r["probability"], reverse=True)
