"""Pipeline secuencial en PyTorch: Dataset, padding/máscara y entrenamiento.

Compartido por GRU y Transformer (mismo esquema de entrenamiento, §5.2/§5.3).
El encoding de una sola sesión (`encode_session`) lo reutiliza la API (M5) para
evitar train/serve skew.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src import config


def delta_stats(train_sessions: pd.DataFrame) -> tuple[float, float]:
    """Media y desviación de los delta_t (log1p) sobre el train set."""
    all_deltas = np.concatenate(
        [np.asarray(d, dtype="float64") for d in train_sessions["deltas_t"]]
    )
    mean = float(all_deltas.mean())
    std = float(all_deltas.std()) or 1.0
    return mean, std


def subsample_negatives(
    train_sessions: pd.DataFrame, ratio: int = config.NEG_SUBSAMPLE_RATIO,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Submuestrea negativos a `ratio`:1 (solo train; nunca val/test)."""
    pos = train_sessions[train_sessions["converted"] == 1]
    neg = train_sessions[train_sessions["converted"] == 0]
    keep_neg = min(len(neg), ratio * len(pos))
    neg = neg.sample(n=keep_neg, random_state=seed)
    return (
        pd.concat([pos, neg])
        .sample(frac=1.0, random_state=seed)  # barajar
        .reset_index(drop=True)
    )


class SessionDataset(Dataset):
    """Sesiones como (tokens, deltas estandarizados, label)."""

    def __init__(self, df: pd.DataFrame, delta_mean: float = 0.0, delta_std: float = 1.0):
        self.tokens = list(df["event_types"])
        self.deltas = list(df["deltas_t"])
        self.labels = df["converted"].to_numpy(dtype="float32")
        self.mean, self.std = delta_mean, delta_std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        d = (np.asarray(self.deltas[i], dtype="float32") - self.mean) / self.std
        return np.asarray(self.tokens[i], dtype="int64"), d, self.labels[i]


def collate(batch):
    """Empaqueta un batch con padding (PAD=0) y máscara (True = posición PAD)."""
    toks, dels, labs = zip(*batch)
    lengths = torch.tensor([len(t) for t in toks], dtype=torch.long)
    maxlen = int(lengths.max())
    n = len(toks)
    tok = torch.zeros(n, maxlen, dtype=torch.long)
    dlt = torch.zeros(n, maxlen, dtype=torch.float32)
    pad_mask = torch.ones(n, maxlen, dtype=torch.bool)
    for i, (t, d) in enumerate(zip(toks, dels)):
        length = len(t)
        tok[i, :length] = torch.from_numpy(t)
        dlt[i, :length] = torch.from_numpy(d)
        pad_mask[i, :length] = False
    return tok, dlt, lengths, pad_mask, torch.tensor(labs, dtype=torch.float32)


def make_loader(dataset: SessionDataset, batch_size: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.SEED)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate, generator=generator if shuffle else None,
    )


@torch.no_grad()
def predict_proba(model: torch.nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Probabilidades (sigmoide de los logits) para todo el loader."""
    model.eval()
    probs: list[np.ndarray] = []
    for tok, dlt, lengths, pad_mask, _ in loader:
        logits = model(tok.to(device), dlt.to(device), lengths, pad_mask.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def _labels_of(loader: DataLoader) -> np.ndarray:
    return np.concatenate([labs.numpy() for *_, labs in loader])


def fit_sequence_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weight: float,
    device: str,
    max_epochs: int = config.MAX_EPOCHS,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    lr: float = config.LEARNING_RATE,
    label: str = "",
) -> dict:
    """Entrena con BCE ponderada + AdamW; early stopping por AUC de validación.

    Devuelve el mejor `val_auc` y deja los mejores pesos cargados en `model`.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device)
    )
    y_val = _labels_of(val_loader)

    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for tok, dlt, lengths, pad_mask, labs in train_loader:
            optimizer.zero_grad()
            logits = model(tok.to(device), dlt.to(device), lengths, pad_mask.to(device))
            loss = criterion(logits, labs.to(device))
            loss.backward()
            optimizer.step()

        val_auc = float(roc_auc_score(y_val, predict_proba(model, val_loader, device)))
        print(f"[{label}] época {epoch}: val_AUC={val_auc:.4f}", flush=True)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                break

    model.load_state_dict(best_state)
    return {"val_auc": best_auc, "epochs": epoch}
