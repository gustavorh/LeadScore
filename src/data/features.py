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
