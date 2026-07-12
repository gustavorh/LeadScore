"""Clasificador GRU sobre secuencias de eventos (§5.2).

Entrada por paso: embedding del evento (32) concatenado con el delta_t
proyectado. El último estado oculto alimenta una capa lineal → logit.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from src import config


class GRUClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = config.EVENT_VOCAB_SIZE,
        embed_dim: int = config.EMBED_DIM,
        delta_dim: int = config.DELTA_PROJ_DIM,
        hidden: int = config.GRU_HIDDEN,
        layers: int = config.GRU_LAYERS,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_TOKEN)
        self.delta_proj = nn.Linear(1, delta_dim)
        self.gru = nn.GRU(embed_dim + delta_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(
        self,
        tokens: torch.Tensor,      # (B, L) long
        deltas: torch.Tensor,      # (B, L) float
        lengths: torch.Tensor,     # (B,) long
        pad_mask: torch.Tensor,    # (B, L) bool — no usado por la GRU (usa lengths)
    ) -> torch.Tensor:
        e = self.embed(tokens)
        d = self.delta_proj(deltas.unsqueeze(-1))
        x = torch.cat([e, d], dim=-1)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)          # hidden: (layers, B, H)
        return self.head(hidden[-1]).squeeze(-1)  # (B,)
