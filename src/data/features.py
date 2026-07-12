"""Features tabulares y utilidades de split.

En M1 alimenta el baseline sobre el dataset UCI. `make_splits` soporta dos
modos: estratificado plano (UCI) y agrupado por visitante (RetailRocket, §4.5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import SEED, TEST_FRAC, TRAIN_FRAC, VAL_FRAC

_UCI_CATEGORICAL: tuple[str, ...] = ("Month", "VisitorType")


def build_uci_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Construye (X, y) desde el DataFrame crudo de UCI.

    Codifica `Month` y `VisitorType` con one-hot, `Weekend` a entero, y devuelve
    el target `Revenue` como 0/1. Todas las columnas de X quedan numéricas.
    """
    df = df.copy()
    y = df["Revenue"].astype(int)
    X = df.drop(columns=["Revenue"])

    if "Weekend" in X.columns:
        X["Weekend"] = X["Weekend"].astype(int)

    cat_cols = [c for c in _UCI_CATEGORICAL if c in X.columns]
    X = pd.get_dummies(X, columns=cat_cols, prefix=cat_cols)

    # get_dummies devuelve columnas bool en pandas ≥2 → castear a entero.
    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    return X, y


def make_splits(
    labels: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series | None = None,
    seed: int = SEED,
    fracs: tuple[float, float, float] = (TRAIN_FRAC, VAL_FRAC, TEST_FRAC),
) -> dict[str, np.ndarray]:
    """Split estratificado 70/15/15 en dos pasos.

    - `groups=None`: estratifica por `labels` (UCI).
    - `groups` dado: divide a nivel de grupo (ningún grupo en dos splits) y
      estratifica por la etiqueta de grupo (1 si el grupo tiene algún positivo).

    Devuelve índices posicionales para 'train', 'val' y 'test'.
    """
    labels = np.asarray(labels)
    n = len(labels)
    all_idx = np.arange(n)
    train_frac, val_frac, test_frac = fracs
    temp_frac = val_frac + test_frac
    rel_test = test_frac / temp_frac  # proporción de test dentro de val+test

    if groups is None:
        idx_train, idx_temp = train_test_split(
            all_idx, test_size=temp_frac, stratify=labels, random_state=seed
        )
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=rel_test, stratify=labels[idx_temp], random_state=seed
        )
    else:
        groups = np.asarray(groups)
        # Etiqueta por grupo = máximo de la etiqueta dentro del grupo.
        group_label = pd.Series(labels).groupby(groups).max()
        uniq = group_label.index.to_numpy()
        glabel = group_label.to_numpy()

        g_train, g_temp = train_test_split(
            uniq, test_size=temp_frac, stratify=glabel, random_state=seed
        )
        glabel_temp = group_label.loc[g_temp].to_numpy()
        g_val, g_test = train_test_split(
            g_temp, test_size=rel_test, stratify=glabel_temp, random_state=seed
        )
        idx_train = all_idx[np.isin(groups, g_train)]
        idx_val = all_idx[np.isin(groups, g_val)]
        idx_test = all_idx[np.isin(groups, g_test)]

    return {
        "train": np.sort(idx_train),
        "val": np.sort(idx_val),
        "test": np.sort(idx_test),
    }


def fit_scaler(x_train: np.ndarray | pd.DataFrame) -> StandardScaler:
    """Ajusta un StandardScaler sobre el conjunto de entrenamiento."""
    scaler = StandardScaler()
    scaler.fit(np.asarray(x_train, dtype=float))
    return scaler


def add_visitor_split(sessions: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Añade la columna `split` (train/val/test) agrupando por visitante (§4.5).

    Ningún visitante queda en dos splits; se estratifica por `converted`.
    """
    splits = make_splits(
        sessions["converted"].to_numpy(),
        groups=sessions["visitor_id"].to_numpy(),
        seed=seed,
    )
    split_col = np.empty(len(sessions), dtype=object)
    for name, idx in splits.items():
        split_col[idx] = name
    out = sessions.copy()
    out["split"] = split_col
    return out


def tabular_from_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """Deriva las features tabulares por sesión (§4.4) desde el parquet de sesiones.

    Usadas por el baseline y K-means sobre RetailRocket. `event_types` es la
    secuencia ya truncada (anti-leakage), con view=1 y addtocart=2.
    """
    ev = sessions["event_types"]
    n_events = sessions["length"].astype(float)
    n_views = ev.map(lambda s: sum(1 for t in s if t == 1)).astype(float)
    n_addtocart = ev.map(lambda s: sum(1 for t in s if t == 2)).astype(float)
    # deltas_t son log1p(segundos); duración ≈ suma de segundos reconstruidos.
    duration = sessions["deltas_t"].map(lambda d: float(np.expm1(np.asarray(d)).sum()))
    mean_delta = duration / n_events.clip(lower=1)
    return pd.DataFrame(
        {
            "n_events": n_events,
            "n_views": n_views,
            "n_addtocart": n_addtocart,
            "n_unique_items": sessions["n_unique_items"].astype(float),
            "duration_sec": duration,
            "mean_delta_t": mean_delta,
            "hour_of_day": sessions["hour_of_day"].astype(float),
            "day_of_week": sessions["day_of_week"].astype(float),
            "addtocart_rate": (n_addtocart / n_events.clip(lower=1)),
        }
    )
