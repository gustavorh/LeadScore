"""Tests de features tabulares y del split estratificado (M1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import features


# --------------------------------------------------------------------------- #
# Fixture: mini-DataFrame con el esquema real de UCI (§4.2)
# --------------------------------------------------------------------------- #
@pytest.fixture
def uci_like() -> pd.DataFrame:
    """Seis filas con las columnas y tipos del dataset UCI."""
    return pd.DataFrame(
        {
            "Administrative": [0, 1, 2, 0, 3, 1],
            "Administrative_Duration": [0.0, 10.0, 20.0, 0.0, 5.0, 3.0],
            "Informational": [0, 0, 1, 0, 2, 0],
            "Informational_Duration": [0.0, 0.0, 4.0, 0.0, 8.0, 0.0],
            "ProductRelated": [1, 2, 5, 1, 9, 3],
            "ProductRelated_Duration": [0.0, 64.0, 100.0, 0.0, 300.0, 40.0],
            "BounceRates": [0.2, 0.0, 0.01, 0.2, 0.0, 0.05],
            "ExitRates": [0.2, 0.1, 0.02, 0.2, 0.01, 0.08],
            "PageValues": [0.0, 0.0, 12.0, 0.0, 30.0, 0.0],
            "SpecialDay": [0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
            "Month": ["Feb", "Feb", "Nov", "Feb", "Nov", "May"],
            "OperatingSystems": [1, 2, 4, 1, 3, 2],
            "Browser": [1, 2, 1, 1, 4, 2],
            "Region": [1, 1, 9, 1, 3, 4],
            "TrafficType": [1, 2, 3, 1, 8, 2],
            "VisitorType": [
                "Returning_Visitor", "Returning_Visitor", "New_Visitor",
                "Returning_Visitor", "New_Visitor", "Other",
            ],
            "Weekend": [False, False, True, False, True, False],
            "Revenue": [False, False, True, False, True, False],
        }
    )


# --------------------------------------------------------------------------- #
# build_uci_features
# --------------------------------------------------------------------------- #
def test_build_uci_features_returns_numeric_matrix(uci_like: pd.DataFrame) -> None:
    X, y = features.build_uci_features(uci_like)
    assert len(X) == len(uci_like)
    assert len(y) == len(uci_like)
    # Todas las columnas deben ser numéricas (categóricas ya codificadas).
    assert all(np.issubdtype(dt, np.number) for dt in X.dtypes)
    assert not X.isna().any().any()


def test_build_uci_features_target_is_binary(uci_like: pd.DataFrame) -> None:
    _, y = features.build_uci_features(uci_like)
    assert set(np.unique(y)) <= {0, 1}
    assert int(y.sum()) == 2  # dos Revenue=True en el fixture


def test_build_uci_features_encodes_categoricals(uci_like: pd.DataFrame) -> None:
    X, _ = features.build_uci_features(uci_like)
    # Month y VisitorType no deben quedar como texto crudo en las columnas.
    assert "Month" not in X.columns
    assert "VisitorType" not in X.columns
    # Debe existir alguna columna derivada de la codificación de VisitorType.
    assert any("VisitorType" in c for c in X.columns)


# --------------------------------------------------------------------------- #
# make_splits — modo estratificado plano (UCI)
# --------------------------------------------------------------------------- #
def test_make_splits_disjoint_and_complete() -> None:
    labels = np.array([0, 1] * 100)  # 200 muestras, balanceadas
    splits = features.make_splits(labels, seed=42)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    # Cobertura exacta y sin solapamiento.
    assert len(set(tr) | set(va) | set(te)) == len(labels)
    assert set(tr).isdisjoint(va)
    assert set(tr).isdisjoint(te)
    assert set(va).isdisjoint(te)


def test_make_splits_fractions_approx() -> None:
    labels = np.array([0, 1] * 500)  # 1000 muestras
    splits = features.make_splits(labels, seed=42)
    n = len(labels)
    assert abs(len(splits["train"]) / n - 0.70) < 0.03
    assert abs(len(splits["val"]) / n - 0.15) < 0.03
    assert abs(len(splits["test"]) / n - 0.15) < 0.03


def test_make_splits_stratified() -> None:
    # 20% positivos; cada split debe preservar la tasa aproximadamente.
    labels = np.array([1] * 200 + [0] * 800)
    splits = features.make_splits(labels, seed=42)
    for key in ("train", "val", "test"):
        rate = labels[splits[key]].mean()
        assert abs(rate - 0.20) < 0.05


def test_make_splits_deterministic() -> None:
    labels = np.array([0, 1] * 100)
    a = features.make_splits(labels, seed=42)
    b = features.make_splits(labels, seed=42)
    for key in ("train", "val", "test"):
        assert np.array_equal(a[key], b[key])


# --------------------------------------------------------------------------- #
# make_splits — modo agrupado por visitante (RetailRocket, §4.5)
# --------------------------------------------------------------------------- #
def test_make_splits_grouped_no_visitor_leakage() -> None:
    # 300 muestras repartidas en 100 visitantes (3 sesiones c/u).
    groups = np.repeat(np.arange(100), 3)
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=300)
    splits = features.make_splits(labels, groups=groups, seed=42)
    g_tr = set(groups[splits["train"]])
    g_va = set(groups[splits["val"]])
    g_te = set(groups[splits["test"]])
    # Ningún visitante puede estar en más de un split.
    assert g_tr.isdisjoint(g_va)
    assert g_tr.isdisjoint(g_te)
    assert g_va.isdisjoint(g_te)
    # Cobertura completa de las muestras.
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(labels)
