"""What to measure, and the traps that made earlier numbers misleading.

Every function here takes a ``TrackResult`` and an :class:`EvalClip`, so it
works for either method and either cohort.

Three lessons are encoded rather than left to whoever writes the next script,
because each one produced a wrong conclusion first:

1. **Report end-systole separately from the cycle median.** A median over all
   frames is dominated by frames near end-diastole, where motion is small and
   matching is easy. On IV02/4ch the median read 3.65 px while end-systole --
   the frame strain is read at -- was 7.48 px with 8 of 72 points surviving.
2. **Report coverage next to every error.** An error computed only over the
   points that survived flatters a method that loses the hard ones. That is
   exactly how ``strided`` appeared to beat ``reference`` in vivo (3.08 vs
   3.22 px) before its NaN propagation was fixed; measured fairly it was 5.07.
3. **Gain is not accuracy.** The fraction of true motion reproduced determines
   the scale of the strain curve, and a tracker can have small position error
   and still read strain 15% low. It needs true correspondence, so it is only
   available where ``EvalClip.truth`` holds.
"""
from __future__ import annotations

import numpy as np

__all__ = ["evaluate", "summarise", "gain", "coverage"]


def coverage(result) -> float:
    """Fraction of tracked positions that came back finite."""
    finite = np.isfinite(result.x) & np.isfinite(result.y)
    return float(finite.mean())


def gain(result, clip) -> float:
    """Fraction of the true displacement reproduced, along its own direction.

    Projected onto the true direction so perpendicular noise neither inflates
    nor deflates it, and measured only where the truth has real motion -- a
    stationary point says nothing about scale. NaN unless the clip carries true
    correspondence.
    """
    if not clip.truth:
        return float("nan")
    origin = clip.start_frame
    predicted = np.stack(
        [result.x - clip.reference_x[origin], result.y - clip.reference_y[origin]],
        axis=-1,
    )
    true = np.stack(
        [
            clip.reference_x - clip.reference_x[origin],
            clip.reference_y - clip.reference_y[origin],
        ],
        axis=-1,
    )
    magnitude = np.linalg.norm(true, axis=-1)
    moving = np.isfinite(predicted).all(axis=-1) & (magnitude > 2.0)
    if not moving.any():
        return float("nan")
    direction = true[moving] / magnitude[moving][:, None]
    return float(
        np.median((predicted[moving] * direction).sum(-1) / magnitude[moving])
    )


def evaluate(result, clip, method: str = "", pairing: str = "") -> dict:
    """Every measurement for one clip and one tracking run.

    ``error_px_es`` and ``coverage_es`` are the pair to read first: they
    describe the frame strain is read at. ``error_px_median`` is kept because
    it is comparable with the literature, not because it is the number that
    matters.
    """
    error = np.hypot(result.x - clip.reference_x, result.y - clip.reference_y)
    finite = np.isfinite(result.x) & np.isfinite(result.y)
    closure = np.hypot(result.x[-1] - result.x[0], result.y[-1] - result.y[0])
    es = int(np.clip(clip.es_frame - 1, 0, result.n_frames - 1))

    return {
        "key": clip.key,
        "method": method or getattr(result, "pairing", ""),
        "pairing": pairing or getattr(result, "pairing", ""),
        "truth": bool(clip.truth),
        "error_px_es": float(np.nanmedian(error[es])),
        "coverage_es": float(finite[es].mean()),
        "error_px_median": float(np.nanmedian(error)),
        "error_px_p90": float(np.nanpercentile(error, 90)),
        "coverage": coverage(result),
        "closure_px_mean": float(np.nanmean(closure)),
        "gain": gain(result, clip),
        "es_frame": es + 1,
        "n_points": int(result.n_points),
        "n_frames": int(result.n_frames),
    }


def summarise(rows, by: str = "pairing") -> dict:
    """Medians per group, with end-systole and coverage kept beside the rest.

    A group whose ``coverage_es`` is low has an ``error_px_es`` computed over
    few points; the two are reported together so that cannot be read in
    isolation.
    """
    out = {}
    for group in sorted({row[by] for row in rows}):
        subset = [row for row in rows if row[by] == group]

        def median(field):
            values = np.array([r[field] for r in subset], dtype=float)
            return float(np.nanmedian(values)) if values.size else float("nan")

        out[group] = {
            "n_clips": len(subset),
            "error_px_es": median("error_px_es"),
            "coverage_es": median("coverage_es"),
            "error_px_median": median("error_px_median"),
            "error_px_p90": median("error_px_p90"),
            "coverage": median("coverage"),
            "closure_px_mean": median("closure_px_mean"),
            "gain": median("gain"),
        }
    return out


def format_summary(summary, title: str = "") -> str:
    """A fixed-width table, end-systole first because it is the point."""
    lines = []
    if title:
        lines.append(title)
    lines.append(
        f"{'group':16s} {'ES err':>8s} {'ES cov':>7s} {'med err':>8s} "
        f"{'p90':>7s} {'cov':>6s} {'closure':>8s} {'gain':>6s} {'clips':>6s}"
    )
    for group, stats in summary.items():
        lines.append(
            f"{group:16s} {stats['error_px_es']:8.2f} "
            f"{100 * stats['coverage_es']:6.0f}% {stats['error_px_median']:8.2f} "
            f"{stats['error_px_p90']:7.2f} {100 * stats['coverage']:5.0f}% "
            f"{stats['closure_px_mean']:8.2f} {stats['gain']:6.3f} "
            f"{stats['n_clips']:6d}"
        )
    lines.append("ES err/cov describe the frame strain is read at -- read them first.")
    return "\n".join(lines)
