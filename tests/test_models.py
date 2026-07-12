"""Tests de los modelos secuenciales y del pipeline PyTorch (M3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src import config
from src.data import sequence
from src.models.gru import GRUClassifier
from src.models.transformer import TransformerClassifier

DEVICE = "cpu"  # tests deterministas en CPU


def _batch(sequences: list[list[int]], deltas: list[list[float]]):
    """Empaqueta secuencias en tensores con padding y máscara vía collate."""
    items = [
        (np.asarray(t, dtype="int64"), np.asarray(d, dtype="float32"), 0.0)
        for t, d in zip(sequences, deltas)
    ]
    tok, dlt, lengths, pad_mask, _ = sequence.collate(items)
    return tok, dlt, lengths, pad_mask


@pytest.fixture(autouse=True)
def _seed():
    config.set_seed(0)


# --------------------------------------------------------------------------- #
# collate
# --------------------------------------------------------------------------- #
def test_collate_pads_and_masks():
    tok, dlt, lengths, pad_mask = _batch([[1, 2, 1], [2]], [[0.0, 0.5, 0.3], [0.0]])
    assert tok.shape == (2, 3)
    assert lengths.tolist() == [3, 1]
    # máscara: True en posiciones PAD.
    assert pad_mask[0].tolist() == [False, False, False]
    assert pad_mask[1].tolist() == [False, True, True]
    # PAD token = 0 en las posiciones rellenadas.
    assert tok[1, 1:].tolist() == [0, 0]


# --------------------------------------------------------------------------- #
# forward shapes
# --------------------------------------------------------------------------- #
def test_gru_forward_shape():
    model = GRUClassifier().eval()
    tok, dlt, lengths, pad_mask = _batch([[1, 2, 1], [2, 1]], [[0.0, 0.5, 0.3], [0.0, 0.2]])
    out = model(tok, dlt, lengths, pad_mask)
    assert out.shape == (2,)


def test_transformer_forward_shape():
    model = TransformerClassifier().eval()
    tok, dlt, lengths, pad_mask = _batch([[1, 2, 1], [2, 1]], [[0.0, 0.5, 0.3], [0.0, 0.2]])
    out = model(tok, dlt, lengths, pad_mask)
    assert out.shape == (2,)


# --------------------------------------------------------------------------- #
# invariancia al padding (valida la máscara)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_cls", [GRUClassifier, TransformerClassifier])
def test_padding_invariance(model_cls):
    torch.manual_seed(0)
    model = model_cls().eval()
    seq, delt = [1, 2, 1], [0.0, 0.5, 0.3]

    tok1, dlt1, len1, mask1 = _batch([seq], [delt])
    # mismo ejemplo con 2 posiciones PAD extra.
    tok2, dlt2, len2, mask2 = _batch([seq, [1, 1, 1, 1, 1]], [delt, [0.0] * 5])

    with torch.no_grad():
        o1 = model(tok1, dlt1, len1, mask1)[0]
        o2 = model(tok2, dlt2, len2, mask2)[0]  # primer ejemplo, ahora padeado a 5
    assert torch.allclose(o1, o2, atol=1e-5)


# --------------------------------------------------------------------------- #
# atención del Transformer
# --------------------------------------------------------------------------- #
def test_transformer_attention_shape_and_rows_sum_to_one():
    model = TransformerClassifier().eval()
    tok, dlt, _, pad_mask = _batch([[1, 2, 1]], [[0.0, 0.5, 0.3]])
    attn = model.attention(tok, dlt, pad_mask)
    assert attn.shape == (1, 3, 3)
    # cada fila es una distribución de atención (suma ≈ 1).
    assert torch.allclose(attn.sum(-1), torch.ones(1, 3), atol=1e-4)


# --------------------------------------------------------------------------- #
# submuestreo de negativos (solo train)
# --------------------------------------------------------------------------- #
def test_subsample_negatives_ratio():
    df = pd.DataFrame(
        {
            "converted": [1] * 20 + [0] * 1000,
            "event_types": [[1, 2, 1]] * 1020,
            "deltas_t": [[0.0, 0.1, 0.2]] * 1020,
            "visitor_id": range(1020),
        }
    )
    out = sequence.subsample_negatives(df, ratio=10, seed=42)
    n_pos = int((out["converted"] == 1).sum())
    n_neg = int((out["converted"] == 0).sum())
    assert n_pos == 20
    assert n_neg == 200  # 10:1
