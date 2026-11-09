"""Grouped cross-validation training loop for the heads.

Three places this could leak, all closed deliberately:

* the target scaler is fitted on the training fold only;
* early stopping watches an *inner* validation split carved out of the
  training fold by group, never the test fold;
* the inner split is grouped too, so a patient never straddles it.

Out-of-fold predictions come back in the same :class:`ProbeResult` the ridge
baseline uses, so the numbers are directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .heads import TargetScaler, build_head, masked_mse
from .metrics import TARGET_LABELS, masked_metrics
from .probe import FoldResult, ProbeResult

__all__ = ["TrainConfig", "train_cv"]


@dataclass
class TrainConfig:
    head: str = "attn"
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-2
    hidden: int = 64
    dropout: float = 0.1
    patience: int = 40
    val_fraction: float = 0.2
    seed: int = 0
    device: str = "cpu"

    def describe(self) -> str:
        return (
            f"{self.head} head, lr={self.lr}, wd={self.weight_decay}, "
            f"dropout={self.dropout}, hidden={self.hidden}, "
            f"max {self.epochs} epochs, patience {self.patience}"
        )


def _view_one_hot(views: np.ndarray) -> np.ndarray:
    return np.stack([(views == "2ch").astype(np.float32),
                     (views == "4ch").astype(np.float32)], axis=-1)


def _inner_split(groups: np.ndarray, fraction: float, rng: np.random.Generator):
    """Hold out whole groups for early stopping, never individual clips."""
    unique = np.unique(groups)
    n_validation = max(1, int(round(len(unique) * fraction)))
    validation_groups = set(rng.choice(unique, size=n_validation, replace=False))
    is_validation = np.array([group in validation_groups for group in groups])
    return ~is_validation, is_validation


def _train_one_fold(embeddings, views, targets, groups, config, rng):
    import torch

    device = torch.device(config.device)
    train_mask, validation_mask = _inner_split(groups, config.val_fraction, rng)
    if not validation_mask.any() or not train_mask.any():
        train_mask = np.ones(len(groups), dtype=bool)
        validation_mask = train_mask

    scaler = TargetScaler().fit(targets[train_mask])
    scaled = scaler.transform(targets)

    tensors = {
        "x": torch.tensor(embeddings, dtype=torch.float32, device=device),
        "v": torch.tensor(views, dtype=torch.float32, device=device),
        "y": torch.tensor(scaled, dtype=torch.float32, device=device),
        "m": torch.tensor(np.isfinite(targets), dtype=torch.float32, device=device),
    }

    torch.manual_seed(config.seed)
    model = build_head(
        config.head,
        embeddings.shape[-1],
        hidden=config.hidden,
        dropout=config.dropout,
    ).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    train_index = torch.tensor(np.flatnonzero(train_mask), device=device)
    validation_index = torch.tensor(np.flatnonzero(validation_mask), device=device)

    best_loss, best_state, waited = float("inf"), None, 0
    for _ in range(config.epochs):
        model.train()
        optimiser.zero_grad()
        prediction = model(tensors["x"][train_index], tensors["v"][train_index])
        loss = masked_mse(
            prediction, tensors["y"][train_index], tensors["m"][train_index]
        )
        loss.backward()
        optimiser.step()

        model.eval()
        with torch.no_grad():
            validation_loss = masked_mse(
                model(tensors["x"][validation_index], tensors["v"][validation_index]),
                tensors["y"][validation_index],
                tensors["m"][validation_index],
            ).item()

        if validation_loss < best_loss - 1e-6:
            best_loss, waited = validation_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler


def train_cv(data, config: TrainConfig | None = None, n_splits: int = 5) -> ProbeResult:
    """Train a head under grouped CV and return out-of-fold predictions."""
    import torch
    from sklearn.model_selection import GroupKFold

    config = config or TrainConfig()
    rng = np.random.default_rng(config.seed)

    embeddings = data.embeddings.astype(np.float32)
    views = _view_one_hot(data.views)
    targets = data.targets
    predictions = np.full_like(targets, np.nan)

    n_groups = len(np.unique(data.groups))
    splitter = GroupKFold(n_splits=min(n_splits, n_groups))

    folds = []
    for index, (train, test) in enumerate(
        splitter.split(embeddings, targets[:, -1], groups=data.groups)
    ):
        model, scaler = _train_one_fold(
            embeddings[train], views[train], targets[train], data.groups[train],
            config, rng,
        )
        model.eval()
        with torch.no_grad():
            scaled = model(
                torch.tensor(embeddings[test], dtype=torch.float32, device=config.device),
                torch.tensor(views[test], dtype=torch.float32, device=config.device),
            ).cpu().numpy()
        predictions[test] = scaler.inverse_transform(scaled)

        folds.append(
            FoldResult(
                fold=index,
                train_groups=len(np.unique(data.groups[train])),
                test_groups=len(np.unique(data.groups[test])),
                indices=test,
            )
        )

    # A target with no training signal anywhere must not be scored as if the
    # head had predicted it; blank those columns out.
    never_observed = ~np.isfinite(targets).any(axis=0)
    predictions[:, never_observed] = np.nan

    return ProbeResult(
        name=f"{config.head}-head",
        predictions=predictions,
        truth=targets,
        metrics=masked_metrics(targets, predictions, TARGET_LABELS),
        folds=folds,
    )
