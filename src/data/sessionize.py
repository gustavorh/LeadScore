"""Sesionización de RetailRocket (§4.3): eventos → sesiones etiquetadas.

Regla anti-leakage (§4.3, §10): la etiqueta `converted` se calcula sobre la
sesión completa, pero la SECUENCIA DE ENTRADA elimina el primer `transaction` y
todo lo posterior. El modelo predice conversión desde el comportamiento previo,
nunca desde la compra misma. El paso crítico está vectorizado y verificado por
`tests/test_sessionize.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.config import CATEGORY_OOV, EVENT_VOCAB, MAX_LEN, MIN_EVENTS, SESSION_GAP_MIN
from src.data.download import EVENTS_CSV

_GAP_MS: int = SESSION_GAP_MIN * 60 * 1000  # corte de sesión en milisegundos


def sessionize(events: pd.DataFrame) -> pd.DataFrame:
    """Convierte el DataFrame de eventos crudos en sesiones etiquetadas.

    Devuelve una fila por sesión con: `session_id, visitor_id, event_types,
    item_cats, deltas_t, length, converted, n_unique_items, hour_of_day,
    day_of_week`.
    """
    df = events.loc[:, ["timestamp", "visitorid", "event", "itemid"]].copy()
    df.sort_values(["visitorid", "timestamp"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 1) Cortar sesiones: cambio de visitante o gap > 30 min.
    new_session = (df["visitorid"] != df["visitorid"].shift()) | (
        df["timestamp"].diff() > _GAP_MS
    )
    df["session_id"] = new_session.cumsum()

    # 2) Etiqueta a partir de la sesión CRUDA (antes de truncar).
    is_tx = df["event"].eq("transaction")
    df["converted"] = (
        is_tx.groupby(df["session_id"]).transform("max").astype("int64")
    )

    # 3) Anti-leakage: quitar la transacción y todo evento en/después de ella.
    first_tx_ts = df["timestamp"].where(is_tx).groupby(df["session_id"]).transform("min")
    keep = (~is_tx) & (first_tx_ts.isna() | (df["timestamp"] < first_tx_ts))
    kept = df.loc[keep].copy()

    kept["token"] = kept["event"].map(EVENT_VOCAB).astype("int64")
    delta_sec = kept.groupby("session_id")["timestamp"].diff().fillna(0.0) / 1000.0
    kept["delta_log"] = np.log1p(delta_sec)

    # 4) Agregar por sesión.
    grouped = kept.groupby("session_id", sort=True)
    sessions = grouped.agg(
        visitor_id=("visitorid", "first"),
        event_types=("token", list),
        deltas_t=("delta_log", list),
        length=("token", "size"),
        converted=("converted", "first"),
        start_ts=("timestamp", "first"),
        n_unique_items=("itemid", "nunique"),
    ).reset_index()

    # 5) Filtrar sesiones sin señal (sobre la secuencia ya truncada).
    sessions = sessions[sessions["length"] >= MIN_EVENTS].reset_index(drop=True)

    # 6) Truncar a MAX_LEN (últimos eventos) y resetear el primer delta.
    def _cap(row: pd.Series) -> pd.Series:
        seq = row["event_types"]
        deltas = list(row["deltas_t"])
        if len(seq) > MAX_LEN:
            seq = seq[-MAX_LEN:]
            deltas = deltas[-MAX_LEN:]
        if deltas:
            deltas[0] = 0.0
        return pd.Series({"event_types": seq, "deltas_t": deltas})

    capped = sessions.apply(_cap, axis=1)
    sessions["event_types"] = capped["event_types"]
    sessions["deltas_t"] = capped["deltas_t"]
    sessions["length"] = sessions["event_types"].map(len)
    sessions["item_cats"] = sessions["length"].map(lambda n: [CATEGORY_OOV] * n)

    # 7) Features temporales desde el inicio de la sesión.
    start = pd.to_datetime(sessions["start_ts"], unit="ms")
    sessions["hour_of_day"] = start.dt.hour.astype("int64")
    sessions["day_of_week"] = start.dt.dayofweek.astype("int64")

    return sessions[
        [
            "session_id", "visitor_id", "event_types", "item_cats", "deltas_t",
            "length", "converted", "n_unique_items", "hour_of_day", "day_of_week",
        ]
    ]


def build_sessions() -> pd.DataFrame:
    """Lee events.csv, sesiona y guarda `data/processed/sessions.parquet`."""
    config.ensure_dirs()
    events = pd.read_csv(EVENTS_CSV)
    sessions = sessionize(events)
    out = config.DATA_PROCESSED / "sessions.parquet"
    sessions.to_parquet(out, index=False)
    return sessions


if __name__ == "__main__":
    s = build_sessions()
    print(
        f"[sessionize] {len(s)} sesiones | "
        f"conversión={s['converted'].mean():.4f} | "
        f"largo medio={s['length'].mean():.2f}"
    )
