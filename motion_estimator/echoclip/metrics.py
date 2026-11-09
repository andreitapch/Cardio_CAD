"""Regression metrics that cope with missing ground truth.

23% of clips carry a NaN somewhere in their 7-element target, so every metric
here is masked and every metric reports its own ``n``. Support is uneven and
non-random -- GE failed to track segments 4-6 far more often than 1-3 -- so a
segmental number quoted without its ``n`` is not interpretable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TargetMetrics", "masked_metrics", "TARGET_LABELS"]

TARGET_LABELS = ("seg1", "seg2", "seg3", "seg4", "seg5", "seg6", "GLS")


@dataclass
class TargetMetrics:
    """Per-target scores, each over however many clips had a real value."""

    label: str
    n: int
    mae: float
    rmse: float
    bias: float
    pearson: float
    loa_lower: float
    loa_upper: float

    def row(self) -> str:
        return (
            f"{self.label:5} n={self.n:4d}  MAE {self.mae:6.2f}  "
            f"RMSE {self.rmse:6.2f}  bias {self.bias:+6.2f}  "
            f"r {self.pearson:+.3f}  LoA [{self.loa_lower:+6.2f}, {self.loa_upper:+6.2f}]"
        )


def masked_metrics(
    truth: np.ndarray, prediction: np.ndarray, labels=TARGET_LABELS
) -> list[TargetMetrics]:
    """Score each target column over the clips where truth is finite.

    ``bias`` and the limits of agreement are the Bland-Altman quantities:
    mean difference, and mean +/- 1.96 SD. Predictions are allowed to be NaN
    where truth is NaN; those entries are simply not scored.
    """
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape:
        raise ValueError(f"shape mismatch: truth {truth.shape} vs pred {prediction.shape}")

    results = []
    for index, label in enumerate(labels):
        mask = np.isfinite(truth[:, index]) & np.isfinite(prediction[:, index])
        n = int(mask.sum())
        if n == 0:
            results.append(TargetMetrics(label, 0, *([float("nan")] * 6)))
            continue

        actual = truth[mask, index]
        predicted = prediction[mask, index]
        difference = predicted - actual

        # Pearson is undefined when either side is constant, which happens for
        # the mean-only baseline; report NaN rather than a divide warning.
        if n < 2 or actual.std() == 0 or predicted.std() == 0:
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(actual, predicted)[0, 1])

        spread = difference.std(ddof=1) if n > 1 else 0.0
        results.append(
            TargetMetrics(
                label=label,
                n=n,
                mae=float(np.abs(difference).mean()),
                rmse=float(np.sqrt((difference**2).mean())),
                bias=float(difference.mean()),
                pearson=correlation,
                loa_lower=float(difference.mean() - 1.96 * spread),
                loa_upper=float(difference.mean() + 1.96 * spread),
            )
        )
    return results


def format_metrics(results: list[TargetMetrics], title: str = "") -> str:
    lines = [title] if title else []
    lines.extend(result.row() for result in results)
    return "\n".join(lines)
