"""Tests de sesionización (M2, §4.3) — con foco en anti-leakage (§10)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import sessionize

SEC = 1000                 # 1 segundo en milisegundos (timestamps de RetailRocket)
GAP = 31 * 60 * 1000       # 31 min: supera el corte de sesión de 30 min


def make_events(rows: list[tuple[int, str, int, int]]) -> pd.DataFrame:
    """Construye un DataFrame de eventos. Cada fila: (visitorid, event, itemid, t_ms)."""
    return pd.DataFrame(
        {
            "timestamp": [r[3] for r in rows],
            "visitorid": [r[0] for r in rows],
            "event": [r[1] for r in rows],
            "itemid": [r[2] for r in rows],
            "transactionid": [
                1.0 if r[1] == "transaction" else np.nan for r in rows
            ],
        }
    )


def test_truncate_removes_transaction_and_after() -> None:
    # Sesión convertida: la transacción y lo posterior NO deben entrar a la entrada.
    df = make_events([
        (1, "view", 10, 0 * SEC),
        (1, "view", 10, 1 * SEC),
        (1, "addtocart", 10, 2 * SEC),
        (1, "transaction", 10, 3 * SEC),
        (1, "view", 20, 4 * SEC),
    ])
    out = sessionize.sessionize(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["converted"] == 1
    assert list(row["event_types"]) == [1, 1, 2]  # view, view, addtocart
    assert row["length"] == 3


def test_converted_false_without_transaction() -> None:
    df = make_events([
        (2, "view", 10, 0 * SEC),
        (2, "view", 11, 1 * SEC),
        (2, "view", 12, 2 * SEC),
    ])
    out = sessionize.sessionize(df)
    assert len(out) == 1
    assert out.iloc[0]["converted"] == 0
    assert list(out.iloc[0]["event_types"]) == [1, 1, 1]


def test_min_events_filter_drops_short_sessions() -> None:
    # (a) convertida pero con 1 solo evento previo a la transacción → se descarta.
    # (b) no convertida con 2 eventos → se descarta.
    df = make_events([
        (3, "view", 10, 0 * SEC),
        (3, "transaction", 10, 1 * SEC),
        (4, "view", 10, 0 * SEC),
        (4, "view", 11, 1 * SEC),
    ])
    out = sessionize.sessionize(df)
    assert len(out) == 0


def test_max_len_truncation() -> None:
    rows = [(5, "view", i, i * SEC) for i in range(60)]  # 60 views en <30 min
    out = sessionize.sessionize(make_events(rows))
    assert len(out) == 1
    assert out.iloc[0]["length"] == 50


def test_gap_splits_into_two_sessions() -> None:
    df = make_events([
        (6, "view", 1, 0 * SEC),
        (6, "view", 2, 1 * SEC),
        (6, "view", 3, 2 * SEC),
        (6, "view", 4, 2 * SEC + GAP),      # tras 31 min → nueva sesión
        (6, "view", 5, 2 * SEC + GAP + SEC),
        (6, "view", 6, 2 * SEC + GAP + 2 * SEC),
    ])
    out = sessionize.sessionize(df)
    assert len(out) == 2
    assert all(out["length"] == 3)


def test_no_transaction_token_in_any_sequence() -> None:
    # Invariante global: las secuencias solo contienen tokens de view(1)/addtocart(2).
    df = make_events([
        (7, "view", 1, 0 * SEC),
        (7, "addtocart", 1, 1 * SEC),
        (7, "view", 1, 2 * SEC),
        (7, "transaction", 1, 3 * SEC),
        (8, "view", 2, 0 * SEC),
        (8, "view", 2, 1 * SEC),
        (8, "addtocart", 2, 2 * SEC),
    ])
    out = sessionize.sessionize(df)
    all_tokens = {t for seq in out["event_types"] for t in seq}
    assert all_tokens <= {1, 2}
