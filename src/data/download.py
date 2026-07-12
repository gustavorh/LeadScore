"""Descarga de datasets.

- UCI Online Shoppers Purchasing Intention (§4.2): descarga directa sin
  credenciales vía `ucimlrepo`. Es el dataset del pipeline de M1.
- RetailRocket (§4.1): descarga vía Kaggle API. Se implementa/verifica en M2.

Todo se cachea en `data/raw/` para no re-descargar.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from src import config

# --------------------------------------------------------------------------- #
# UCI Online Shoppers (§4.2)
# --------------------------------------------------------------------------- #
UCI_ID: int = 468  # id del dataset en el repositorio UCI
RAW_UCI: Path = config.DATA_RAW / "uci"
UCI_CSV: Path = RAW_UCI / "online_shoppers_intention.csv"


def download_uci(force: bool = False) -> pd.DataFrame:
    """Descarga (o carga de caché) el dataset UCI y lo devuelve completo.

    El DataFrame incluye las 17 features y la columna target `Revenue` (bool).
    """
    if UCI_CSV.exists() and not force:
        return pd.read_csv(UCI_CSV)

    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=UCI_ID)
    features = dataset.data.features
    targets = dataset.data.targets  # columna 'Revenue'
    df = pd.concat([features, targets], axis=1)

    RAW_UCI.mkdir(parents=True, exist_ok=True)
    df.to_csv(UCI_CSV, index=False)
    return df


# --------------------------------------------------------------------------- #
# RetailRocket (§4.1) — se completa en M2
# --------------------------------------------------------------------------- #
RAW_RETAILROCKET: Path = config.DATA_RAW / "retailrocket"
KAGGLE_SLUG: str = "retailrocket/ecommerce-dataset"


def download_retailrocket() -> Path:
    """Descarga RetailRocket vía Kaggle CLI; si falla, instruye descarga manual.

    Devuelve el directorio con los CSV (`events.csv`, `category_tree.csv`, ...).
    """
    events = RAW_RETAILROCKET / "events.csv"
    if events.exists():
        return RAW_RETAILROCKET

    RAW_RETAILROCKET.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
             "-p", str(RAW_RETAILROCKET), "--unzip"],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            "No se pudo descargar RetailRocket vía Kaggle CLI. "
            "Configura credenciales (~/.kaggle/kaggle.json o KAGGLE_KEY) o "
            f"descarga manualmente '{KAGGLE_SLUG}' en {RAW_RETAILROCKET}."
        ) from exc
    return RAW_RETAILROCKET


if __name__ == "__main__":
    df = download_uci()
    print(f"UCI descargado: {df.shape[0]} filas, {df.shape[1]} columnas → {UCI_CSV}")
