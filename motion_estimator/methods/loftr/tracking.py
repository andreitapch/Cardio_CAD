"""Correspondences to trajectories -- tracking a given set of points.

The gap this closes: LoFTR is detector-free, so it will not track a point you
name. It returns matches on its own 1/8-resolution grid (~8 px apart at native
resolution, refined to subpixel), and the contour we care about does not sit on
that grid. So the correspondence field is *interpolated* at the query points.

The interpolant is a **confidence-weighted local affine fit**. For each query
point, matches within ``radius`` are weighted by their LoFTR confidence times a
Gaussian in distance, and a 6-parameter affine map is fitted by weighted least
squares; the displacement is read off at the query point. Why not something
simpler:

* Nearest match is quantised to LoFTR's grid and throws away the subpixel
  refinement that makes strain numbers move.
* Inverse-distance weighted averaging of displacements cannot represent a
  gradient, and a strain measurement *is* a displacement gradient -- averaging
  over a neighbourhood systematically underestimates it.
* A global affine or homography cannot represent myocardial deformation at all;
  it is the thing we are trying to measure.

The fit is made robust by one reweighting pass: residuals larger than
``reject_sigma`` robust standard deviations are dropped and the fit repeated.
Echo frames produce confident-but-wrong matches in speckle and in the blood
pool, and a single such match near a contour point drags it visibly.

**Two chaining modes**, and the difference matters more than the interpolant:

``pairing="consecutive"``
    Match frame *k* to *k+1* and advect the current estimate. Small
    inter-frame motion, so matching is easy, but error compounds over the
    cycle -- the same behaviour as ``propagate.propagate_points`` with
    ``propagate_error=True``, and what the MATLAB in-vivo pipeline does.

``pairing="reference"``
    Match frame 0 to frame *k* and displace the *original* points every time.
    No accumulated drift and no need for drift compensation, at the cost of a
    wide baseline at end-systole -- which is what LoFTR was built for and
    speckle decorrelation is worst at. Worth measuring both ways rather than
    assuming.

Coordinates are **1-based** ``(x=column, y=row)``, matching MATLAB and the rest
of the package, and points with no usable support return NaN rather than a
clamped or last-known value -- a point that has lost tracking should be
visibly missing. :class:`Matches` holds LoFTR's 0-based coordinates; the +1
happens here, once, at the boundary.
"""
from __future__ import annotations

import numpy as np

from ...tracking.result import TrackResult
from .matcher import Matches

__all__ = [
    "displacement_at",
    "track_bidirectional",
    "track_points",
    "track_strided",
    "to_dense_flow",
]


MAX_CONDITION = 1e4


def _weighted_mean(source, target, weights) -> np.ndarray | None:
    total = weights.sum()
    if total <= 0:
        return None
    return ((target - source) * weights[:, None]).sum(0) / total


def _weighted_affine(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    query: np.ndarray,
    model: str,
) -> np.ndarray | None:
    """Fit *model* mapping ``source -> target`` and evaluate it at *query*.

    Returns the ``(2,)`` displacement, or None when the fit is not determined:
    zero total weight, or a design matrix too ill-conditioned to fix a
    gradient. The conditioning check is not decoration -- LoFTR's matches lie
    on a grid, and where support thins to a few nearly collinear cells the
    least-squares solution is dominated by the direction with no spread and
    extrapolates by tens of pixels. ``MAX_CONDITION`` is the cutoff, past
    which the caller falls back to a weighted mean.
    """
    if model == "translation":
        return _weighted_mean(source, target, weights)

    design = np.column_stack([source, np.ones(len(source))])
    root_w = np.sqrt(weights)[:, None]
    weighted = design * root_w
    if not np.isfinite(weighted).all() or weights.sum() <= 0:
        return None
    # Centre before conditioning: absolute pixel coordinates of a few hundred
    # make the constant column look tiny next to x and y, which inflates the
    # condition number for reasons that have nothing to do with the support.
    centred = np.column_stack(
        [source - source.mean(0), np.ones(len(source))]
    ) * root_w
    singular = np.linalg.svd(centred, compute_uv=False)
    if singular[-1] <= 0 or singular[0] / singular[-1] > MAX_CONDITION:
        return None
    try:
        solution, *_ = np.linalg.lstsq(weighted, target * root_w, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(solution).all():
        return None
    predicted = np.concatenate([query, [1.0]]) @ solution
    return predicted - query


def displacement_at(
    matches: Matches,
    x,
    y,
    radius: float = 32.0,
    sigma: float | None = None,
    model: str = "affine",
    min_neighbours: int = 6,
    min_support: int = 8,
    reject_sigma: float = 2.5,
    robust_passes: int = 1,
    deviation_tolerance: float = 3.0,
    one_based: bool = True,
):
    """Interpolate a correspondence field at ``(x, y)``.

    Parameters
    ----------
    matches:
        LoFTR output for one frame pair, in 0-based coordinates.
    x, y:
        Query positions, any matching shape. 1-based unless *one_based* is
        False.
    radius:
        Matches further than this (pixels) are ignored. At the native 500x500
        geometry, ~0.3 mm/px, 32 px is about 10 mm of myocardium -- large
        enough that LoFTR's 8 px grid gives a well-posed affine fit, small
        enough that one affine map is a fair local description.
    sigma:
        Gaussian weighting scale; defaults to ``radius / 2``.
    model:
        ``"affine"`` (6 parameters, represents a deformation gradient) or
        ``"translation"`` (a weighted mean displacement, which cannot).
    min_neighbours:
        Fewer neighbours than this returns NaN rather than a displacement.
        LoFTR's coarse grid is 8 px, so a 32 px radius over well-matched tissue
        holds tens of matches; a handful means the matcher essentially failed
        there, outlier rejection has nothing to work against, and any number
        produced is a guess. The observed failure this guards is a
        three-neighbour fit returning 40 px on a clip whose true motion is 4 --
        one such point is enough to move a strain curve by 20 percentage
        points, because arc length is a sum of local distances.
    min_support:
        Below this many neighbours the affine fit is not attempted; it falls
        back to ``"translation"``, which needs two parameters instead of six.
    reject_sigma, robust_passes:
        Outlier rejection. Each pass drops matches whose residual exceeds
        ``reject_sigma`` times ``1.4826 * MAD`` of the residuals and refits.
        ``robust_passes=0`` disables it.
    deviation_tolerance:
        How far the affine fit may depart from the neighbourhood's own median
        displacement before it is treated as extrapolation, in units of the
        neighbourhood's robust displacement spread (plus a 1 px floor, so a
        neighbourhood that agrees perfectly does not get a zero-width gate).
        Beyond it the weighted mean is returned instead. This is the guard on
        the failure that matters: where support thins along a contour, an
        affine fit will happily predict a displacement an order of magnitude
        larger than anything it was fitted on, and arc length -- hence strain
        -- is very sensitive to one such point. ``0`` disables it.

    Returns
    -------
    dx, dy, support:
        Displacements with the shape of *x*, NaN where no fit was possible, and
        the integer count of matches each fit used.
    """
    if model not in ("affine", "translation"):
        raise ValueError(f"model must be 'affine' or 'translation', got {model!r}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: x {x.shape} vs y {y.shape}")
    if sigma is None:
        sigma = radius / 2.0

    shape = x.shape
    query = np.column_stack([x.ravel(), y.ravel()])
    if one_based:
        query = query - 1.0

    dx = np.full(query.shape[0], np.nan)
    dy = np.full(query.shape[0], np.nan)
    support = np.zeros(query.shape[0], dtype=int)

    if matches.n_matches == 0:
        return dx.reshape(shape), dy.reshape(shape), support.reshape(shape)

    source = matches.xy0
    target = matches.xy1
    confidence = matches.confidence

    # (n_query, n_matches) -- 200 points x 3500 matches is 5 MB, fine.
    delta = query[:, None, :] - source[None, :, :]
    distance = np.hypot(delta[..., 0], delta[..., 1])
    within = distance <= radius

    for index in range(query.shape[0]):
        if not np.isfinite(query[index]).all():
            continue
        neighbours = np.flatnonzero(within[index])
        if neighbours.size < max(min_neighbours, 1):
            continue

        local_source = source[neighbours]
        local_target = target[neighbours]
        weights = confidence[neighbours] * np.exp(
            -0.5 * (distance[index, neighbours] / sigma) ** 2
        )
        this_model = model if neighbours.size >= min_support else "translation"

        passes = max(robust_passes, 0)
        for pass_index in range(passes + 1):
            step = _weighted_affine(
                local_source, local_target, weights, query[index], this_model
            )
            if step is None and this_model == "affine":
                this_model = "translation"
                step = _weighted_affine(
                    local_source, local_target, weights, query[index], this_model
                )
            # The fit must be the last thing that happens, so never trim on the
            # final pass -- a trimmed set that is not refitted is not the set
            # the returned displacement came from.
            if step is None or pass_index == passes:
                break
            # The floor is the fitted model's own parameter count, not
            # ``min_support``: a two-parameter translation can be checked for
            # outliers on four matches, and keying this to the affine
            # threshold meant rejection never ran on exactly the thin
            # neighbourhoods that most needed it.
            floor = 6 if this_model == "affine" else 3
            if weights.size <= floor:
                break
            residual = _residuals(
                local_source, local_target, weights, this_model
            )
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
            if not np.isfinite(scale) or scale <= 0:
                break
            keep = np.abs(residual - np.median(residual)) <= reject_sigma * scale
            if keep.all() or keep.sum() <= floor:
                break
            local_source = local_source[keep]
            local_target = local_target[keep]
            weights = weights[keep]

        if step is None:
            # Not even a weighted mean was available (no positive weight).
            continue
        if this_model == "affine" and deviation_tolerance > 0:
            step = _gate_extrapolation(
                step, local_source, local_target, weights, deviation_tolerance
            )
            if step is None:
                continue
        dx[index] = step[0]
        dy[index] = step[1]
        support[index] = int(weights.size)

    return dx.reshape(shape), dy.reshape(shape), support.reshape(shape)


def _gate_extrapolation(
    step: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    tolerance: float,
) -> np.ndarray | None:
    """Replace an extrapolating affine prediction with the weighted mean.

    The gate is the neighbourhood's own displacements: their median, and a
    robust spread from the MAD, with a 1 px floor so a neighbourhood in perfect
    agreement still admits a subpixel correction. A prediction further than
    ``tolerance`` spreads from that median is not interpolation, it is the fit
    running off the end of its support.
    """
    displacement = target - source
    median = np.median(displacement, axis=0)
    spread = 1.4826 * np.median(np.abs(displacement - median), axis=0)
    limit = tolerance * np.hypot(*spread) + 1.0
    if np.hypot(*(step - median)) <= limit:
        return step
    return _weighted_mean(source, target, weights)


def _residuals(source, target, weights, model) -> np.ndarray:
    """Magnitude of each match's departure from the fitted model."""
    if model == "translation":
        mean = ((target - source) * weights[:, None]).sum(0) / max(weights.sum(), 1e-12)
        return np.hypot(*(target - source - mean).T)
    design = np.column_stack([source, np.ones(len(source))])
    root_w = np.sqrt(weights)[:, None]
    solution, *_ = np.linalg.lstsq(design * root_w, target * root_w, rcond=None)
    return np.hypot(*(target - design @ solution).T)


def track_points(
    frames,
    start_x,
    start_y,
    matcher=None,
    matches=None,
    pairing: str = "consecutive",
    reference_frame: int = 0,
    progress: bool = False,
    **interp,
):
    """Track ``(start_x, start_y)`` across a clip. See the module docstring.

    Parameters
    ----------
    frames:
        ``(n_frames, rows, cols)``, as returned by ``image_prep.frames_first``.
        Ignored when *matches* is supplied.
    start_x, start_y:
        ``(n_points,)`` 1-based pixel coordinates on ``frames[reference_frame]``
        -- the points you want tracked.
    matcher:
        A :class:`~motion_estimator.methods.loftr.matcher.LoFTRMatcher`. Optional if
        *matches* is given, which is how a precomputed set of correspondences
        is reused across interpolation settings without re-running the GPU.
    matches:
        Precomputed output of ``matcher.match_sequence`` for the same
        *pairing*.
    pairing:
        ``"consecutive"`` or ``"reference"``.
    **interp:
        Passed to :func:`displacement_at` (``radius``, ``model``, ...).

    Notes
    -----
    ``reference_frame`` other than 0 is allowed for ``pairing="reference"``
    (MITT records a ``ref_frame``, which is not always frame 1), and the
    returned arrays still run over all frames in order.
    """
    if pairing not in ("consecutive", "reference"):
        raise ValueError(
            f"pairing must be 'consecutive' or 'reference', got {pairing!r}"
        )
    start_x = np.asarray(start_x, dtype=float).ravel()
    start_y = np.asarray(start_y, dtype=float).ravel()
    if start_x.shape != start_y.shape:
        raise ValueError(f"shape mismatch: {start_x.shape} vs {start_y.shape}")

    if matches is None:
        if matcher is None:
            raise ValueError("supply either a matcher or precomputed matches")
        matches = matcher.match_sequence(
            frames,
            reference=None if pairing == "consecutive" else reference_frame,
            progress=progress,
        )

    if pairing == "consecutive":
        n_frames = len(matches) + 1
        if reference_frame != 0:
            raise ValueError(
                "pairing='consecutive' chains from frame 0; use "
                "pairing='reference' to start elsewhere"
            )
    else:
        n_frames = len(matches)
        if not 0 <= reference_frame < n_frames:
            raise ValueError(
                f"reference_frame {reference_frame} outside 0..{n_frames - 1}"
            )

    n_points = start_x.size
    x = np.full((n_frames, n_points), np.nan)
    y = np.full((n_frames, n_points), np.nan)
    support = np.zeros((n_frames, n_points), dtype=int)
    n_matches = np.array([m.n_matches for m in matches], dtype=int)

    if pairing == "consecutive":
        x[0] = start_x
        y[0] = start_y
        for step, pair in enumerate(matches):
            dx, dy, used = displacement_at(pair, x[step], y[step], **interp)
            x[step + 1] = x[step] + dx
            y[step + 1] = y[step] + dy
            support[step + 1] = used
    else:
        x[reference_frame] = start_x
        y[reference_frame] = start_y
        support[reference_frame] = -1  # not estimated: the given annotation
        for frame, pair in enumerate(matches):
            if frame == reference_frame:
                continue
            dx, dy, used = displacement_at(pair, start_x, start_y, **interp)
            x[frame] = start_x + dx
            y[frame] = start_y + dy
            support[frame] = used

    return TrackResult(
        x=x,
        y=y,
        support=support,
        n_matches=n_matches,
        pairing=pairing,
        meta={"reference_frame": reference_frame, **interp},
    )


def track_bidirectional(
    frames,
    start_x,
    start_y,
    matcher=None,
    matches=None,
    end_x=None,
    end_y=None,
    blend: str = "linear",
    progress: bool = False,
    **interp,
):
    """Chain forwards and backwards, and blend the two.

    Chained tracking accumulates error in one direction, so the estimate is
    best at the frame it started from and worst at the far end. Running the
    same chain backwards from the last frame puts the accurate end where the
    forward pass is worst, and a weighted blend uses whichever is closer to its
    own anchor.

    The catch, and it is not a detail: the backward pass needs positions on the
    **last** frame, which we do not have. For a cardiac cycle the last frame is
    the same phase as the first, so *start* is the natural estimate for it --
    that is exactly the assumption ``drift_compensation`` already makes when it
    forces the curve back to zero, made explicit here and applied to the
    positions rather than to the strain curve afterwards. Pass *end_x* /
    *end_y* when the real contour on the last frame is known; the assumption
    then drops out entirely.

    ``blend="linear"`` weights the forward pass by ``1 - k/(n-1)`` and the
    backward pass by ``k/(n-1)``, so each is trusted in proportion to its
    distance from its own anchor. ``blend="forward"`` and ``"backward"`` return
    one pass unblended, which is what makes the comparison measurable rather
    than assumed.

    Only whichever passes are finite contribute: where one has lost a point the
    other carries it alone, and the result is NaN only where both failed.
    """
    if blend not in ("linear", "forward", "backward"):
        raise ValueError(
            f"blend must be linear, forward or backward, got {blend!r}"
        )

    forward = track_points(
        frames,
        start_x,
        start_y,
        matcher=matcher,
        matches=matches,
        pairing="consecutive",
        progress=progress,
        **interp,
    )
    if blend == "forward":
        return forward

    n_frames = forward.n_frames
    reversed_frames = None if frames is None else np.asarray(frames)[::-1]
    reversed_matches = _reverse_matches(matches) if matches is not None else None

    backward = track_points(
        reversed_frames,
        forward.x[0] if end_x is None else end_x,
        forward.y[0] if end_y is None else end_y,
        matcher=matcher,
        matches=reversed_matches,
        pairing="consecutive",
        progress=progress,
        **interp,
    )
    # Undo the reversal so frame k means frame k in both.
    backward_x = backward.x[::-1]
    backward_y = backward.y[::-1]
    backward_support = backward.support[::-1]
    if blend == "backward":
        return TrackResult(
            x=backward_x,
            y=backward_y,
            support=backward_support,
            n_matches=backward.n_matches[::-1],
            pairing="backward",
            meta={"blend": blend, **interp},
        )

    weight = np.linspace(0.0, 1.0, n_frames)[:, None] if n_frames > 1 else np.zeros((1, 1))
    x = _blend(forward.x, backward_x, weight)
    y = _blend(forward.y, backward_y, weight)

    return TrackResult(
        x=x,
        y=y,
        support=np.minimum(forward.support, backward_support),
        n_matches=forward.n_matches,
        pairing="bidirectional",
        meta={"blend": blend, "assumed_end_from_start": end_x is None, **interp},
    )


def _blend(forward, backward, weight):
    """Weighted mean that tolerates either side being NaN."""
    forward_ok = np.isfinite(forward)
    backward_ok = np.isfinite(backward)
    weight_f = np.where(forward_ok, 1.0 - weight, 0.0)
    weight_b = np.where(backward_ok, weight, 0.0)
    total = weight_f + weight_b

    # Where only one pass survives it carries full weight; where a surviving
    # pass has zero blend weight (the anchor frames) it still has to win, or
    # the anchor itself would come back NaN.
    only_forward = forward_ok & ~backward_ok
    only_backward = backward_ok & ~forward_ok
    weight_f = np.where(only_forward, 1.0, weight_f)
    weight_b = np.where(only_backward, 1.0, weight_b)
    degenerate = (total == 0) & (forward_ok | backward_ok)
    weight_f = np.where(degenerate & forward_ok, 1.0, weight_f)
    weight_b = np.where(degenerate & backward_ok, 1.0, weight_b)
    total = weight_f + weight_b

    blended = (
        np.nan_to_num(forward) * weight_f + np.nan_to_num(backward) * weight_b
    )
    return np.where(total > 0, blended / np.where(total > 0, total, 1.0), np.nan)


def _reverse_matches(matches):
    """A consecutive match list for the reversed clip.

    Reversing the frames turns the pair ``(k, k+1)`` into ``(n-1-k, n-2-k)``,
    so the list is reversed *and* each match has its two ends swapped.
    """
    return [
        Matches(pair.xy1, pair.xy0, pair.confidence) for pair in reversed(list(matches))
    ]


def track_strided(
    frames,
    start_x,
    start_y,
    matcher,
    stride: int = 6,
    min_gap: int | None = None,
    reference_frame: int = 0,
    max_sources: int = 3,
    progress: bool = False,
    **interp,
):
    """Track using wide frame gaps only, never consecutive pairs.

    This exists because of a measured property of the matcher, not a hunch.
    LoFTR's fine level regresses its subpixel offset as the spatial expectation
    of a softmax heatmap, and that estimator is biased toward the centre of its
    window: it shrinks small offsets toward zero. Measured against exact ground
    truth on the simulated cohort, the fraction of true motion it reproduces
    depends strongly on how much motion there is:

    ==========  ===========  ==================
    frame gap   true motion  gain (raw matches)
    ==========  ===========  ==================
    1           ~1 px        0.798
    2           ~2 px        0.859
    6           ~6 px        0.979
    12          ~12 px       0.987
    ==========  ===========  ==================

    Our clips move about 1 px per frame, so consecutive tracking runs at a gain
    near 0.8 -- every step is 20% short, and strain reads low by the same
    fraction before drift is even considered. The fix is not a better
    interpolant or a fine-tuned model; it is to stop asking the matcher for
    displacements near its resolution floor.

    So: anchor frames every *stride* frames are chained to each other, and each
    remaining frame is estimated in **one hop** from an anchor at least
    *min_gap* away. Nothing is ever measured across a gap of 1, and no frame is
    more than one hop from a chained anchor, which cuts accumulation too.

    **Per-point fallback.** A point that the matcher loses at one anchor is not
    lost for the rest of the clip: each target frame is filled from its
    preferred source first, then from further anchors for whichever points are
    still missing, up to *max_sources* attempts. Without this the anchor chain
    propagates every gap forward -- on IV02/4ch the last frame came back with
    4 of 72 points instead of 47, and cycle closure was undefined. Each extra
    source costs one more match, hence the cap.

    *matcher* is required -- unlike :func:`track_points` this cannot reuse a
    precomputed consecutive match list, because the pairs it needs are not
    consecutive.
    """
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    if matcher is None:
        raise ValueError(
            "track_strided needs a matcher: the frame pairs it matches are not "
            "consecutive, so a precomputed consecutive match list cannot serve"
        )
    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise ValueError(f"expected (n_frames, rows, cols), got {frames.shape}")

    start_x = np.asarray(start_x, dtype=float).ravel()
    start_y = np.asarray(start_y, dtype=float).ravel()
    n_frames = int(frames.shape[0])
    if not 0 <= reference_frame < n_frames:
        raise ValueError(f"reference_frame {reference_frame} outside the clip")
    gap_floor = stride if min_gap is None else min_gap

    anchors = sorted(
        {reference_frame, n_frames - 1}
        | set(range(reference_frame, n_frames, stride))
        | set(range(reference_frame, -1, -stride))
    )

    x = np.full((n_frames, start_x.size), np.nan)
    y = np.full((n_frames, start_x.size), np.nan)
    support = np.zeros((n_frames, start_x.size), dtype=int)
    n_matches = np.zeros(n_frames, dtype=int)

    x[reference_frame] = start_x
    y[reference_frame] = start_y
    support[reference_frame] = -1  # the given annotation, not an estimate

    from .matcher import value_range

    vrange = value_range(frames)
    done = {reference_frame}

    def fill(target, candidates):
        """Fill *target* from *candidates* in order, until nothing is missing."""
        missing = ~np.isfinite(x[target])
        used_sources = 0
        for source in candidates:
            if used_sources >= max_sources or not missing.any():
                break
            available = missing & np.isfinite(x[source])
            if not available.any():
                continue
            pair = matcher.match(frames[source], frames[target], vrange)
            used_sources += 1
            n_matches[target] = max(n_matches[target], pair.n_matches)
            dx, dy, count = displacement_at(pair, x[source], y[source], **interp)
            filled = available & np.isfinite(dx) & np.isfinite(dy)
            if not filled.any():
                continue
            x[target] = np.where(filled, x[source] + dx, x[target])
            y[target] = np.where(filled, y[source] + dy, y[target])
            support[target] = np.where(filled, count, support[target])
            missing &= ~filled
        done.add(target)

    # 1. chain the anchors outward from the reference, in both directions.
    forward = [a for a in anchors if a >= reference_frame]
    backward = [a for a in anchors if a <= reference_frame][::-1]
    for chain in (forward, backward):
        for source, target in zip(chain, chain[1:]):
            others = sorted(
                (a for a in done if a != source),
                key=lambda a: abs(a - target),
            )
            fill(target, [source] + others)

    # 2. every other frame, one hop from a far enough anchor.
    remaining = [k for k in range(n_frames) if k not in set(anchors)]
    iterator = remaining
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(remaining, desc="strided", unit="frame")
        except ImportError:
            pass

    for frame in iterator:
        fill(frame, _anchor_order(anchors, frame, gap_floor))

    return TrackResult(
        x=x,
        y=y,
        support=support,
        n_matches=n_matches,
        pairing="strided",
        meta={
            "stride": stride,
            "min_gap": gap_floor,
            "reference_frame": reference_frame,
            "n_anchors": len(anchors),
            "max_sources": max_sources,
            **interp,
        },
    )


def _anchor_order(anchors, frame, gap_floor):
    """Anchors to try for *frame*, best first.

    Anchors at least *gap_floor* away come first, nearest of those leading,
    because the whole point is to avoid small displacements. Anchors that are
    too close are kept as a last resort rather than dropped -- a short clip may
    offer nothing better, and a shrunken displacement still beats a NaN.
    """
    far = sorted(
        (a for a in anchors if abs(a - frame) >= gap_floor),
        key=lambda a: abs(a - frame),
    )
    near = sorted(
        (a for a in anchors if abs(a - frame) < gap_floor),
        key=lambda a: -abs(a - frame),
    )
    return far + near


def to_dense_flow(matches: Matches, shape, **interp) -> np.ndarray:
    """Rasterise a correspondence field to ``(rows, cols, 2)``.

    Only needed to feed ``strain.propagate.propagate_points``, i.e. to compare
    against the dense-flow route the MATLAB pipeline used. Interpolating at the
    query points directly is cheaper and does not round-trip through a grid.
    """
    rows, cols = shape
    grid_x, grid_y = np.meshgrid(
        np.arange(1, cols + 1, dtype=float), np.arange(1, rows + 1, dtype=float)
    )
    dx, dy, _ = displacement_at(matches, grid_x, grid_y, **interp)
    return np.stack([dx, dy], axis=-1)
