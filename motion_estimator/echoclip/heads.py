"""Trainable heads over frozen EchoCLIP embeddings.

The trunk stays frozen (see the package docstring), so these are the only
parameters that ever move. They are deliberately tiny: ~60 patients cannot
support much capacity, and the ridge baseline in :mod:`probe` is the bar.

``MeanPoolHead`` reproduces ridge's inductive bias in torch -- if it does not
land near ridge, the training loop is wrong, not the model. ``AttnPoolHead``
adds the one thing ridge cannot do: weight frames. That matters because the
target is *end-systolic* strain and the in-vivo data carries no end-systolic
frame index (``scar_struct`` gives only the cycle window), so the head has to
find the relevant part of the cycle for itself.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "MeanPoolHead",
    "AttnPoolHead",
    "TargetScaler",
    "masked_mse",
    "build_head",
]

N_TARGETS = 7


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Mean squared error over the entries where ground truth exists.

    23% of clips have a NaN somewhere in their target. Averaging over the mask
    rather than over all elements keeps a clip with four of six segments
    contributing those four at full weight, instead of being dropped or having
    its missing entries silently count as zero error.
    """
    valid = mask.sum()
    if valid == 0:
        return prediction.sum() * 0.0
    squared = (prediction - torch.nan_to_num(target)) ** 2
    return (squared * mask).sum() / valid


class TargetScaler:
    """Per-target standardisation, fitted on the training fold only.

    Segment 3's strain spans ~25 percentage points while segment 6 spans ~10,
    so unscaled MSE would quietly optimise the wide targets and ignore the
    narrow ones. Fitted with nan-aware statistics since the targets are holey.
    """

    def __init__(self) -> None:
        self.mean = None
        self.scale = None

    def fit(self, targets: np.ndarray) -> "TargetScaler":
        # A column can be entirely NaN in a given fold, which nanmean/nanstd
        # warn about before returning NaN. The NaN is handled below; silence
        # the warning so it does not look like something went wrong.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean = np.nanmean(targets, axis=0)
            scale = np.nanstd(targets, axis=0)
        # A target that is constant or entirely missing must not divide by zero.
        self.scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
        return self

    def transform(self, targets: np.ndarray) -> np.ndarray:
        return (targets - self.mean) / self.scale

    def inverse_transform(self, targets: np.ndarray) -> np.ndarray:
        return targets * self.scale + self.mean


class MeanPoolHead(nn.Module):
    """Mean over frames, then a linear map. The torch analogue of ridge."""

    def __init__(self, embed_dim: int, n_views: int = 2, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(embed_dim + n_views, N_TARGETS)

    def pool(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings.mean(dim=1)

    def forward(self, embeddings: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(embeddings)
        return self.linear(self.dropout(torch.cat([pooled, views], dim=-1)))


class AttnPoolHead(MeanPoolHead):
    """Learned attention over frames, then the same linear map.

    A single scalar score per frame, softmaxed across the cycle -- one hidden
    layer, a few thousand parameters. Anything larger would out-capacity the
    dataset. :meth:`attention` exposes the weights so what the head selects can
    be checked against where end-systole actually falls.
    """

    def __init__(
        self,
        embed_dim: int,
        n_views: int = 2,
        hidden: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__(embed_dim, n_views, dropout)
        self.score = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def attention(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.score(embeddings).squeeze(-1), dim=-1)

    def pool(self, embeddings: torch.Tensor) -> torch.Tensor:
        weights = self.attention(embeddings)
        return torch.einsum("bf,bfd->bd", weights, embeddings)


def build_head(name: str, embed_dim: int, **kwargs) -> nn.Module:
    if name == "mean":
        return MeanPoolHead(embed_dim, dropout=kwargs.get("dropout", 0.0))
    if name == "attn":
        return AttnPoolHead(
            embed_dim,
            hidden=kwargs.get("hidden", 64),
            dropout=kwargs.get("dropout", 0.0),
        )
    raise ValueError(f"unknown head {name!r}; expected 'mean' or 'attn'")
