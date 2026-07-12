"""Configuración global: rutas, seed y todos los hiperparámetros.

Única fuente de verdad del proyecto. Cualquier otro módulo importa desde aquí;
nunca se hardcodean rutas ni hiperparámetros fuera de este archivo.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
ARTIFACTS: Path = PROJECT_ROOT / "artifacts"
RESULTS: Path = PROJECT_ROOT / "results"


def ensure_dirs() -> None:
    """Crea los directorios de salida si no existen (idempotente)."""
    for path in (DATA_RAW, DATA_PROCESSED, ARTIFACTS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #
SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """Fija la semilla en random, numpy y torch (si está disponible)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        # torch aún no es necesario en M1 (baseline sklearn).
        pass


def get_device() -> str:
    """Devuelve 'mps' si hay GPU Apple disponible, si no 'cpu'.

    Los artefactos siempre se guardan y cargan de forma agnóstica al device.
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# --------------------------------------------------------------------------- #
# Split (§4.5) — estratificado 70/15/15, por visitante en RetailRocket
# --------------------------------------------------------------------------- #
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.15

# --------------------------------------------------------------------------- #
# Sesionización (§4.3)
# --------------------------------------------------------------------------- #
SESSION_GAP_MIN: int = 30       # minutos de inactividad que cortan una sesión
MIN_EVENTS: int = 3             # sesiones con menos eventos se descartan
MAX_LEN: int = 50               # truncado máximo de la secuencia

# Vocabulario de eventos (§4.3). 'transaction' NO tiene token: se elimina de la
# entrada (anti-leakage) y solo sirve para etiquetar `converted`.
PAD_TOKEN: int = 0
EVENT_VOCAB: dict[str, int] = {"view": 1, "addtocart": 2}
EVENT_VOCAB_SIZE: int = 1 + len(EVENT_VOCAB)  # +1 por PAD

# Categorías de ítem: diferidas a v1 (todas → OOV). Se conserva el token para
# compatibilidad futura del contrato de la API.
CATEGORY_OOV: int = 1
CATEGORY_VOCAB_SIZE: int = 2  # PAD + OOV

# --------------------------------------------------------------------------- #
# Features tabulares por sesión (§4.4) — baseline y K-means
# --------------------------------------------------------------------------- #
TABULAR_FEATURES: list[str] = [
    "n_events",
    "n_views",
    "n_addtocart",
    "n_unique_items",
    "duration_sec",
    "mean_delta_t",
    "hour_of_day",
    "day_of_week",
    "addtocart_rate",
]

# --------------------------------------------------------------------------- #
# Modelos secuenciales (§5.2, §5.3) — pequeños, entrenables en CPU/MPS
# --------------------------------------------------------------------------- #
EMBED_DIM: int = 32
DELTA_PROJ_DIM: int = 16   # proyección del delta_t escalar
GRU_HIDDEN: int = 64
GRU_LAYERS: int = 1

TRANSFORMER_DMODEL: int = 64
TRANSFORMER_NHEAD: int = 4
TRANSFORMER_FF: int = 128
TRANSFORMER_LAYERS: int = 2
TRANSFORMER_DROPOUT: float = 0.1

# Entrenamiento
LEARNING_RATE: float = 1e-3
MAX_EPOCHS: int = 10
BATCH_SIZE: int = 128
EARLY_STOPPING_PATIENCE: int = 3
NEG_SUBSAMPLE_RATIO: int = 10  # negativos:positivos en TRAIN (nunca val/test)

# --------------------------------------------------------------------------- #
# Clustering (§5.4) y bandas de acción (§5.5)
# --------------------------------------------------------------------------- #
K_CLUSTERS: int = 4

# Bandas de acción sobre p_final (distintas del umbral binario calibrado).
ACTION_HOT: float = 0.70   # p >= 0.70 → "caliente"
ACTION_WARM: float = 0.40  # 0.40 <= p < 0.70 → "tibio"; p < 0.40 → "frío"

# --------------------------------------------------------------------------- #
# Metadatos
# --------------------------------------------------------------------------- #
VERSION: str = "1.0.0"
