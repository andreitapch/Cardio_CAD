"""Ridge and mean-only baselines on frozen EchoCLIP embeddings.

These are the floor everything else has to clear. The mean-only predictor in
particular is not a formality: at ~60 patients with a target whose spread is a
few percent strain, "predict the training mean" is a genuinely hard bar, and a
model that fails to beat it has learned nothing regardless of its correlation.

Every fit here is per-target and masked, because 23% of clips have a NaN
somewhere: a clip missing seg5 still trains seg1-4 and GLS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import TARGET_LABELS, TargetMetrics, masked_metrics

__all__ = ["FoldResult", "ProbeResult", "pool_frames", "build_features", "run_cv"]


def pool_frames(embeddings: np.ndarray) -> np.ndarray:
    """``(n, frames, dim)`` -> ``(n, dim)``, mean over frames then renormalised.

    Mean pooling is what EchoCLIP's own zero-shot code does across frames, so
    it is the honest starting point; a learned temporal pooling has to beat
    this to justify its parameters.
    """
    pooled = embeddings.mean(axis=1)
    norms = np.linalg.norm(pooled, axis=-1, keepdims=True)
    return pooled / np.where(norms == 0, 1.0, norms)


def build_features(data, pooled: np.ndarray | None = None) -> np.ndarray:
    """Pooled embedding with the cardiac view appended as a one-hot.

    One model spans both views -- it doubles the sample size, which dominates
    at this n -- so the view has to be an input rather than a separate model.
    """
    if pooled is None:
        pooled = pool_frames(data.embeddings)
    view_is_4ch = (data.views == "4ch").astype(float)[:, None]
    return np.hstack([pooled, view_is_4ch])


@dataclass
class FoldResult:
    fold: int
    train_groups: int
    test_groups: int
    indices: np.ndarray


@dataclass
class ProbeResult:
    """Out-of-fold predictions and the scores derived from them."""

    name: str
    predictions: np.ndarray
    truth: np.ndarray
    metrics: list[TargetMetrics] = field(default_factory=list)
    folds: list[FoldResult] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"{self.name}  ({len(self.folds)} folds)"]
        lines.extend("  " + metric.row() for metric in self.metrics)
        return "\n".join(lines)


def _fit_masked_ridge(features, targets, alpha):
    """One ridge per target column, fitted on that column's finite rows.

    sklearn's multi-output Ridge cannot skip NaNs, and imputing them would
    invent ground truth, so the columns are fitted independently.
    """
    from sklearn.linear_model import Ridge

    models = []
    for column in range(targets.shape[1]):
        mask = np.isfinite(targets[:, column])
        if mask.sum() < 2:
            models.append(None)
            continue
        models.append(Ridge(alpha=alpha).fit(features[mask], targets[mask, column]))
    return models


def _predict_masked(models, features, n_targets):
    prediction = np.full((len(features), n_targets), np.nan)
    for column, model in enumerate(models):
        if model is not None:
            prediction[:, column] = model.predict(features)
    return prediction


def run_cv(
    data,
    estimator: str = "ridge",
    alpha: float = 10.0,
    n_splits: int = 5,
    seed: int = 0,
) -> ProbeResult:
    """Grouped cross-validation producing out-of-fold predictions.

    Folds are grouped on ``data.groups``, so every clip belonging to one group
    -- both views, and both scans when grouped by patient -- is tested by a
    model that never saw any of it.
    """
    from sklearn.model_selection import GroupKFold

    features = build_features(data)
    targets = data.targets
    n_targets = targets.shape[1]
    predictions = np.full_like(targets, np.nan)

    n_groups = len(np.unique(data.groups))
    splits = min(n_splits, n_groups) if n_splits > 0 else n_groups
    splitter = GroupKFold(n_splits=splits)

    folds = []
    for index, (train, test) in enumerate(
        splitter.split(features, targets[:, -1], groups=data.groups)
    ):
        if estimator == "ridge":
            models = _fit_masked_ridge(features[train], targets[train], alpha)
            predictions[test] = _predict_masked(models, features[test], n_targets)
        elif estimator == "mean":
            # The floor: training-fold mean per target, ignoring the features.
            with np.errstate(invalid="ignore"):
                column_means = np.nanmean(targets[train], axis=0)
            predictions[test] = np.tile(column_means, (len(test), 1))
        else:
            raise ValueError(f"unknown estimator {estimator!r}")

        folds.append(
            FoldResult(
                fold=index,
                train_groups=len(np.unique(data.groups[train])),
                test_groups=len(np.unique(data.groups[test])),
                indices=test,
            )
        )

    return ProbeResult(
        name=f"{estimator}(alpha={alpha})" if estimator == "ridge" else estimator,
        predictions=predictions,
        truth=targets,
        metrics=masked_metrics(targets, predictions, TARGET_LABELS),
        folds=folds,
    )
