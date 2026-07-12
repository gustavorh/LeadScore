"""Tests del clustering K-means y el nombrado de segmentos (M4, §5.4)."""

from __future__ import annotations

import numpy as np

from src.config import TABULAR_FEATURES
from src.models import clustering


def test_name_clusters_by_centroid_heuristic() -> None:
    # columnas: n_events,n_views,n_addtocart,n_unique_items,duration,mean_delta,
    #           hour,dow,addtocart_rate
    centroids = np.array([
        [3, 2, 1, 1, 10, 5, 12, 3, 0.9],   # decidido: mayor addtocart_rate
        [8, 7, 1, 6, 50, 6, 12, 3, 0.1],   # comparador: más ítems únicos
        [10, 10, 0, 2, 60, 6, 12, 3, 0.0],  # explorador: más vistas
        [6, 4, 3, 2, 40, 6, 12, 3, 0.4],   # carrito abandonado: resto
    ])
    names = clustering.name_clusters(centroids, TABULAR_FEATURES)
    assert names == {
        0: "decidido",
        1: "comparador",
        2: "explorador",
        3: "carrito abandonado",
    }


def test_name_clusters_are_unique_and_deterministic() -> None:
    rng = np.random.default_rng(0)
    centroids = rng.normal(size=(4, len(TABULAR_FEATURES)))
    a = clustering.name_clusters(centroids, TABULAR_FEATURES)
    b = clustering.name_clusters(centroids, TABULAR_FEATURES)
    assert a == b
    assert len(set(a.values())) == 4  # los 4 nombres, sin repetir
