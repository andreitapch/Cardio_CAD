"""Chaining an estimator across a cardiac cycle: four strategies.

Written against :class:`~motion_estimator.methods.base.MotionEstimator`, so it
is shared by every method rather than reimplemented per method. All four are
kept because which one wins has already surprised us once: for LoFTR
``strided`` was best on simulated data (gain 0.987) and worst in vivo
(5.07 px), because in vivo the matcher fails often enough that chaining
compounds the losses. A method that never fails has nothing to compound, so the
ordering can invert -- and the only way to know is to be able to run all four.

Frame gaps matter for a measured reason. Both LoFTR and pretrained RAFT shrink
small displacements (|gain| ~0.79 at 1 px, ~0.99 at 12 px) and these clips move
about 1 px per frame, so ``consecutive`` asks for exactly the displacements a
pretrained estimator is worst at. ``strided`` exists to avoid that.
"""
from __future__ import annotations

import numpy as np

from .result import TrackResult

__all__ = ["chain"]


def chain(
    frames,
    start_x,
    start_y,
    estimator,
    pairing: str = "consecutive",
    reference_frame: int = 0,
    stride: int = 6,
    progress: bool = False,
):
    """Track points across *frames* with a dense-flow *estimator*.

    *pairing* is ``"consecutive"``, ``"reference"``, ``"strided"`` or
    ``"bidirectional"``, matching the LoFTR side so evaluation can compare
    like with like.

    Returns a :class:`TrackResult`, the same type the LoFTR path returns, so
    strain and the metrics do not need to know which method ran.
    """
    if pairing not in ("consecutive", "reference", "strided", "bidirectional"):
        raise ValueError(f"unknown pairing {pairing!r}")

    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise ValueError(f"expected (n_frames, rows, cols), got {frames.shape}")
    start_x = np.asarray(start_x, dtype=float).ravel()
    start_y = np.asarray(start_y, dtype=float).ravel()
    n_frames = int(frames.shape[0])
    if not 0 <= reference_frame < n_frames:
        raise ValueError(f"reference_frame {reference_frame} outside the clip")

    # One range for the clip, so the uint8 conversion inside the estimator does
    # not rescale the image differently from frame to frame.
    finite = np.isfinite(frames)
    low = float(frames[finite].min()) if finite.any() else 0.0
    high = float(frames[finite].max()) if finite.any() else 1.0
    vrange = (low, high if high > low else low + 1.0)

    if pairing == "bidirectional":
        forward = chain(
            frames, start_x, start_y, estimator, "consecutive", 0, stride, progress
        )
        backward = chain(
            frames[::-1], forward.x[0], forward.y[0], estimator,
            "consecutive", 0, stride, progress,
        )  # a reversed clip, so a multi-frame model still sees a real sequence
        weight = (
            np.linspace(0.0, 1.0, n_frames)[:, None]
            if n_frames > 1 else np.zeros((1, 1))
        )
        reverse_x, reverse_y = backward.x[::-1], backward.y[::-1]
        return TrackResult(
            x=_blend(forward.x, reverse_x, weight),
            y=_blend(forward.y, reverse_y, weight),
            support=np.minimum(forward.support, backward.support[::-1]),
            n_matches=forward.n_matches,
            pairing=f"{estimator.name}:bidirectional",
            meta={"estimator": estimator.name},
        )

    x = np.full((n_frames, start_x.size), np.nan)
    y = np.full((n_frames, start_x.size), np.nan)
    support = np.zeros((n_frames, start_x.size), dtype=int)
    x[reference_frame] = start_x
    y[reference_frame] = start_y
    support[reference_frame] = -1

    if pairing == "strided":
        anchors = sorted(
            {reference_frame, n_frames - 1}
            | set(range(reference_frame, n_frames, stride))
            | set(range(reference_frame, -1, -stride))
        )
        sources = _plan_strided(anchors, n_frames, reference_frame, stride)
    elif pairing == "reference":
        sources = {k: reference_frame for k in range(n_frames) if k != reference_frame}
    else:
        forward_steps = {k: k - 1 for k in range(reference_frame + 1, n_frames)}
        backward_steps = {k: k + 1 for k in range(reference_frame - 1, -1, -1)}
        sources = {**forward_steps, **backward_steps}

    order = sorted(sources, key=lambda k: abs(k - reference_frame))
    iterator = order
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(order, desc=f"{estimator.name}:{pairing}", unit="frame")
        except ImportError:
            pass

    for target in iterator:
        source = sources[target]
        dx, dy, used = estimator.displacement(
            frames, source, target, x[source], y[source], vrange
        )
        x[target] = x[source] + dx
        y[target] = y[source] + dy
        support[target] = used

    return TrackResult(
        x=x,
        y=y,
        support=support,
        n_matches=np.zeros(n_frames, dtype=int),  # dense flow has no match count
        pairing=f"{estimator.name}:{pairing}",
        meta={
            "estimator": estimator.name,
            "reference_frame": reference_frame,
            "stride": stride if pairing == "strided" else None,
        },
    )


def _plan_strided(anchors, n_frames, reference_frame, stride):
    """Which frame each target is estimated from, for the strided plan.

    Anchors chain to their neighbour in the chain; every other frame takes one
    hop from the nearest anchor that is at least *stride* away, which is the
    point of the strategy -- never measure a displacement near the matcher's
    resolution floor.
    """
    sources = {}
    forward = [a for a in anchors if a >= reference_frame]
    backward = [a for a in anchors if a <= reference_frame][::-1]
    for chain in (forward, backward):
        for source, target in zip(chain, chain[1:]):
            sources[target] = source

    anchor_set = set(anchors)
    for frame in range(n_frames):
        if frame in anchor_set:
            continue
        far = [a for a in anchors if abs(a - frame) >= stride]
        pool = far or list(anchors)
        sources[frame] = min(pool, key=lambda a: abs(a - frame))
    return sources


def _blend(forward, backward, weight):
    """Weighted mean tolerating NaN on either side; see the LoFTR equivalent."""
    forward_ok = np.isfinite(forward)
    backward_ok = np.isfinite(backward)
    weight_f = np.where(forward_ok, 1.0 - weight, 0.0)
    weight_b = np.where(backward_ok, weight, 0.0)
    only_f = forward_ok & ~backward_ok
    only_b = backward_ok & ~forward_ok
    weight_f = np.where(only_f, 1.0, weight_f)
    weight_b = np.where(only_b, 1.0, weight_b)
    total = weight_f + weight_b
    degenerate = (total == 0) & (forward_ok | backward_ok)
    weight_f = np.where(degenerate & forward_ok, 1.0, weight_f)
    weight_b = np.where(degenerate & backward_ok, 1.0, weight_b)
    total = weight_f + weight_b
    blended = np.nan_to_num(forward) * weight_f + np.nan_to_num(backward) * weight_b
    return np.where(total > 0, blended / np.where(total > 0, total, 1.0), np.nan)
