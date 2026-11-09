"""Where is the motion lost? Raw matches vs interpolation.

``diagnose_drift`` measured a gain of 0.855 -- the tracker reproduces 85% of
the true step, so strain reads about 15% low no matter how well drift is
handled. That is a scale error, and this locates it.

Two measurements on the same frame pairs, against the same exact ground truth:

* **raw** -- LoFTR's own matches, compared with the true displacement of the
  scatterers nearest each match. No interpolation, no guards.
* **interpolated** -- ``displacement_at`` evaluated at the scatterer positions
  themselves, which is what tracking actually consumes.

If raw is ~1.0 and interpolated is ~0.85, the loss is in the interpolation, and
the two knobs that would cause it are both here: ``radius`` sets how far the
local affine fit averages, and the robust rejection pulls the fit toward the
median of its neighbourhood. Both shrink a displacement field that varies
across the neighbourhood -- which a deforming myocardium always does.

Gain is measured **along the true direction of travel**, so perpendicular noise
does not inflate or deflate it, and only points with real motion are counted --
a stationary point carries no information about scale.

    python -m motion_estimator.scripts.diagnose_gain
    python -m motion_estimator.scripts.diagnose_gain --radii 8 16 24 32 48
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "artifacts" / "simulated-cache"


def _gain(predicted, true, floor=0.5):
    """Median projection of *predicted* onto *true*, over the moving points."""
    magnitude = np.linalg.norm(true, axis=-1)
    moving = np.isfinite(predicted).all(axis=-1) & (magnitude > floor)
    if not moving.any():
        return float("nan"), 0
    direction = true[moving] / magnitude[moving][:, None]
    along = (predicted[moving] * direction).sum(-1)
    return float(np.median(along / magnitude[moving])), int(moving.sum())


def raw_gain(matcher, frames, scat_x, scat_y, pair, neighbours=4, radius=6.0):
    """Gain of LoFTR's own matches, with no interpolation involved."""
    source, target = pair
    matches = matcher.match(frames[source], frames[target])
    if matches.n_matches == 0:
        return float("nan"), 0

    points = np.c_[scat_x[source] - 1.0, scat_y[source] - 1.0]
    tree = cKDTree(points)
    distance, index = tree.query(
        matches.xy0, k=neighbours, distance_upper_bound=radius
    )
    weight = np.where(np.isfinite(distance), 1.0 / np.maximum(distance, 1e-3), 0.0)
    index = np.clip(index, 0, len(points) - 1)
    total = weight.sum(axis=1)
    usable = total > 0
    if not usable.any():
        return float("nan"), 0

    true_dx = (scat_x[target] - scat_x[source])[index]
    true_dy = (scat_y[target] - scat_y[source])[index]
    true = np.stack(
        [
            (weight * true_dx).sum(axis=1)[usable] / total[usable],
            (weight * true_dy).sum(axis=1)[usable] / total[usable],
        ],
        axis=-1,
    )
    return _gain(matches.displacement[usable], true)


def interpolated_gain(matcher, frames, scat_x, scat_y, pair, columns, **interp):
    """Gain of the field `displacement_at` produces, at the tracked points."""
    from motion_estimator.methods.loftr import displacement_at

    source, target = pair
    matches = matcher.match(frames[source], frames[target])
    if matches.n_matches == 0:
        return float("nan"), 0

    x = scat_x[source][columns]
    y = scat_y[source][columns]
    dx, dy, _ = displacement_at(matches, x, y, **interp)
    true = np.stack(
        [
            scat_x[target][columns] - x,
            scat_y[target][columns] - y,
        ],
        axis=-1,
    )
    return _gain(np.stack([dx, dy], axis=-1), true)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--clips", type=int, default=5)
    parser.add_argument("--spacing", type=int, default=6,
                        help="frame gap; larger gaps give more motion to measure")
    parser.add_argument("--n-points", type=int, default=300)
    parser.add_argument("--radii", type=float, nargs="+",
                        default=[8, 16, 24, 32, 48])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    from motion_estimator.methods.loftr import LoFTRMatcher
    from motion_estimator.loaders.train_loader import list_cached, split_by_patient

    clips = list_cached(args.cache_dir)
    _, held_out = split_by_patient(clips, args.val_fraction, args.seed)
    held_out = held_out[: args.clips]
    if not held_out:
        print("no held-out clips", file=sys.stderr)
        return 1

    matcher = LoFTRMatcher(checkpoint=args.checkpoint)
    loaded = []
    for clip in held_out:
        with np.load(clip.path, allow_pickle=False) as data:
            loaded.append(
                (
                    clip.key,
                    data["frames"].astype(np.float32),
                    data["scat_x"],
                    data["scat_y"],
                )
            )

    # 1. the matcher alone
    raws = []
    for key, frames, scat_x, scat_y in loaded:
        gap = min(args.spacing, frames.shape[0] - 1)
        value, count = raw_gain(matcher, frames, scat_x, scat_y, (0, gap))
        if np.isfinite(value):
            raws.append(value)
    print(f"raw LoFTR matches          gain {np.median(raws):.3f}   "
          f"({len(raws)} clips)")

    # 2. the interpolation, swept over radius
    print(f"\n{'radius':>8s} {'affine':>10s} {'translation':>13s} "
          f"{'affine, no reject':>19s}")
    rows = []
    for radius in args.radii:
        variants = {}
        for label, settings in (
            ("affine", dict(model="affine")),
            ("translation", dict(model="translation")),
            ("affine_no_reject", dict(model="affine", robust_passes=0,
                                      deviation_tolerance=0.0)),
        ):
            values = []
            for key, frames, scat_x, scat_y in loaded:
                gap = min(args.spacing, frames.shape[0] - 1)
                columns = np.linspace(
                    0, scat_x.shape[1] - 1, args.n_points
                ).astype(int)
                value, _ = interpolated_gain(
                    matcher, frames, scat_x, scat_y, (0, gap), columns,
                    radius=radius, **settings,
                )
                if np.isfinite(value):
                    values.append(value)
            variants[label] = float(np.median(values)) if values else float("nan")
        rows.append({"radius": radius, **variants})
        print(f"{radius:8.0f} {variants['affine']:10.3f} "
              f"{variants['translation']:13.3f} "
              f"{variants['affine_no_reject']:19.3f}")

    print("\n1.000 is perfect. Below 1 means motion is under-predicted, "
          "and strain reads low by the same fraction.")
    best = max(rows, key=lambda r: -abs(r["affine"] - 1.0))
    print(f"closest to 1.0: affine at radius {best['radius']:.0f} "
          f"-> gain {best['affine']:.3f}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"raw": float(np.median(raws)), "sweep": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
