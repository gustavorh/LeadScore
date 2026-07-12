"""Clasificador Transformer sobre secuencias de eventos (§5.3).

Embeddings de entrada + positional encoding sinusoidal → encoder de 2 capas con
máscara de padding → mean pooling enmascarado → logit. Cada capa expone los
pesos de atención de forma que se puede guardar un heatmap (§5.3).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from src import config


class SinusoidalPositionalEncoding(nn.Module):
    """Positional encoding sinusoidal clásico (Vaswani et al.)."""

    def __init__(self, d_model: int, max_len: int = config.MAX_LEN + 1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, d_model)
        return x + self.pe[: x.size(1)].unsqueeze(0)


class _EncoderLayer(nn.Module):
    """Capa de encoder post-norm que guarda su atención (need_weights=True)."""

    def __init__(self, d_model: int, nhead: int, ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.lin1 = nn.Linear(d_model, ff)
        self.lin2 = nn.Linear(ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        attn_out, weights = self.attn(
            x, x, x, key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=True,
        )
        self.last_attn = weights.detach()
        x = self.norm1(x + self.drop(attn_out))
        ff = self.lin2(self.drop(self.act(self.lin1(x))))
        return self.norm2(x + self.drop(ff))


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = config.EVENT_VOCAB_SIZE,
        embed_dim: int = config.EMBED_DIM,
        delta_dim: int = config.DELTA_PROJ_DIM,
        d_model: int = config.TRANSFORMER_DMODEL,
        nhead: int = config.TRANSFORMER_NHEAD,
        ff: int = config.TRANSFORMER_FF,
        num_layers: int = config.TRANSFORMER_LAYERS,
        dropout: float = config.TRANSFORMER_DROPOUT,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_TOKEN)
        self.delta_proj = nn.Linear(1, delta_dim)
        self.input_proj = nn.Linear(embed_dim + delta_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [_EncoderLayer(d_model, nhead, ff, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Linear(d_model, 1)

    def _encode(self, tokens, deltas, pad_mask):
        e = self.embed(tokens)
        d = self.delta_proj(deltas.unsqueeze(-1))
        x = self.input_proj(torch.cat([e, d], dim=-1))
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask)
        return x  # (B, L, d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        deltas: torch.Tensor,
        lengths: torch.Tensor,     # no usado (el Transformer usa la máscara)
        pad_mask: torch.Tensor,    # (B, L) bool — True = PAD
    ) -> torch.Tensor:
        h = self._encode(tokens, deltas, pad_mask)
        mask = (~pad_mask).unsqueeze(-1).float()          # (B, L, 1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)

    @torch.no_grad()
    def attention(self, tokens, deltas, pad_mask) -> torch.Tensor:
        """Pesos de atención de la última capa (B, L, L) para interpretación."""
        self.eval()
        self._encode(tokens, deltas, pad_mask)
        return self.layers[-1].last_attn
