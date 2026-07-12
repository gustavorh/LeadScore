"""Descarga de datasets.

- UCI Online Shoppers Purchasing Intention (§4.2): descarga directa sin
  credenciales vía `ucimlrepo`. Es el dataset del pipeline de M1.
- RetailRocket (§4.1): descarga vía Kaggle API. Se implementa/verifica en M2.

Todo se cachea en `data/raw/` para no re-descargar.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from src import config


def _kaggle_cmd() -> str:
    """Ruta del ejecutable `kaggle`, preferentemente el del venv actual."""
    candidate = Path(sys.executable).with_name("kaggle")
    return str(candidate) if candidate.exists() else "kaggle"

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


EVENTS_CSV: Path = RAW_RETAILROCKET / "events.csv"


def download_retailrocket() -> Path:
    """Descarga solo `events.csv` de RetailRocket vía Kaggle CLI.

    Las categorías se difieren a v1 (no se necesita `category_tree.csv`), así que
    basta con el archivo de eventos (~90 MB). Si falla, instruye descarga manual.
    Devuelve la ruta de `events.csv`.
    """
    if EVENTS_CSV.exists():
        return EVENTS_CSV

    RAW_RETAILROCKET.mkdir(parents=True, exist_ok=True)
    try:
        # Con `-f` (archivo único) Kaggle descarga un .zip aunque se pase
        # --unzip, así que lo extraemos manualmente después.
        subprocess.run(
            [_kaggle_cmd(), "datasets", "download", "-d", KAGGLE_SLUG,
             "-f", "events.csv", "-p", str(RAW_RETAILROCKET)],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            "No se pudo descargar RetailRocket vía Kaggle CLI. "
            "Configura credenciales (~/.kaggle/access_token o kaggle.json) o "
            f"descarga manualmente 'events.csv' de '{KAGGLE_SLUG}' en "
            f"{RAW_RETAILROCKET}."
        ) from exc

    zip_path = RAW_RETAILROCKET / "events.csv.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_RETAILROCKET)
        zip_path.unlink()
    return EVENTS_CSV


if __name__ == "__main__":
    df = download_uci()
    print(f"UCI descargado: {df.shape[0]} filas, {df.shape[1]} columnas → {UCI_CSV}")
