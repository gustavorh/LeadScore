"""Segmentación no supervisada de sesiones con K-means (§5.4).

Los clusters se nombran con una heurística documentada sobre sus centroides
(no a mano por índice): a cada nombre se le asigna, en orden de prioridad, el
cluster aún no asignado con mayor valor en su señal característica.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from src.config import K_CLUSTERS, SEED

# (nombre, feature característica): se asignan en este orden por valor máximo.
_NAMING_PRIORITY: list[tuple[str, str]] = [
    ("decidido", "addtocart_rate"),      # mayor intención de compra
    ("comparador", "n_unique_items"),    # explora muchos productos distintos
    ("explorador", "n_views"),           # navega mucho, sin decidir
    ("carrito abandonado", "n_addtocart"),  # el cluster restante
]


def fit_kmeans(x_scaled_train: np.ndarray) -> KMeans:
    """Ajusta K-means (k=4) sobre features tabulares estandarizadas."""
    km = KMeans(n_clusters=K_CLUSTERS, random_state=SEED, n_init=10)
    km.fit(x_scaled_train)
    return km


def name_clusters(centroids: np.ndarray, feature_names: list[str]) -> dict[int, str]:
    """Asigna un nombre a cada cluster según su centroide (asignación greedy)."""
    col = {name: i for i, name in enumerate(feature_names)}
    assigned: dict[int, str] = {}
    for name, feature in _NAMING_PRIORITY:
        values = centroids[:, col[feature]]
        # ordenar clusters por la señal, de mayor a menor, saltando los ya asignados.
        for cluster in np.argsort(-values):
            cluster = int(cluster)
            if cluster not in assigned:
                assigned[cluster] = name
                break
    return assigned
