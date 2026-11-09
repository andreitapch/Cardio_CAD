"""Does the fine-tune fail because it was trained at one resolution only?

Fine-tuning ran on 320-pixel crops; evaluation tracks whole frames, 908 px
simulated and 500 px in vivo. LoFTR's coarse transformer attends over a token
grid that is the image divided by 8 -- 40x40 in training, 113x113 at
evaluation -- and its positional encoding is a function of that grid. Five
million parameters tuned at a single size for 2000 steps can specialise to it,
while the pretrained weights saw many sizes and generalise across them.

This measures matching directly, with no tracking or interpolation in the way,
at several sizes for both checkpoints::

    python -m motion_estimator.scripts.diagnose_resolution --checkpoint <run>/best.pt

The metric is endpoint error of LoFTR's own matches against the exact scatterer
field: for each match, the true displacement is taken from the scatterers
nearest its source point, and compared with the displacement the match implies.
No `displacement_at`, no guards, no chaining -- if the fine-tune damaged the
matcher itself, it shows up here.

How to read it:

* fine-tuned better at 320 and worse at full size -> resolution specialisation.
  Retrain at the evaluation size, or sample crop sizes during training.
* fine-tuned worse at every size -> the training is harmful regardless of
  resolution, and the learning rate is the next thing to halve.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "artifacts" / "simulated-cache"


def match_error(matcher, frames, scat_x, scat_y, pair, size, rng, radius=6.0):
    """Endpoint error of the raw matches, on a centred crop of *size*."""
    source, target = pair
    rows, cols = frames.shape[1:]
    x0, y0 = scat_x[source] - 1.0, scat_y[source] - 1.0

    if size is None:
        top = left = 0
        crop_rows, crop_cols = rows, cols
    else:
        crop_rows = crop_cols = min(size, rows, cols)
        left = int(np.clip(round(x0.mean() - crop_cols / 2), 0, cols - crop_cols))
        top = int(np.clip(round(y0.mean() - crop_rows / 2), 0, rows - crop_rows))

    image0 = frames[source][top: top + crop_rows, left: left + crop_cols]
    image1 = frames[target][top: top + crop_rows, left: left + crop_cols]
    matches = matcher.match(image0, image1)
    if matches.n_matches == 0:
        return np.nan, 0

    # True displacement at each match, from the scatterers nearest its source.
    points = np.c_[x0 - left, y0 - top]
    tree = cKDTree(points)
    distance, index = tree.query(matches.xy0, k=4, distance_upper_bound=radius)
    weight = np.where(np.isfinite(distance), 1.0 / np.maximum(distance, 1e-3), 0.0)
    index = np.clip(index, 0, len(points) - 1)

    true_dx = (scat_x[target] - scat_x[source])[index]
    true_dy = (scat_y[target] - scat_y[source])[index]
    total = weight.sum(axis=1)
    usable = total > 0
    if not usable.any():
        return np.nan, 0
    dx = (weight * true_dx).sum(axis=1)[usable] / total[usable]
    dy = (weight * true_dy).sum(axis=1)[usable] / total[usable]

    predicted = matches.displacement[usable]
    error = np.hypot(predicted[:, 0] - dx, predicted[:, 1] - dy)
    return float(np.median(error)), int(usable.sum())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="a fine-tuned best.pt")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--sizes", type=int, nargs="+", default=[320, 448, 640, 0],
                        help="crop sizes to test; 0 means the whole frame")
    parser.add_argument("--clips", type=int, default=4)
    parser.add_argument("--spacing", type=int, default=12,
                        help="frame gap, in the range training sampled from")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from motion_estimator.methods.loftr import LoFTRMatcher
    from motion_estimator.loaders.train_loader import list_cached, split_by_patient

    clips = list_cached(args.cache_dir)
    _, held_out = split_by_patient(clips, args.val_fraction, args.seed)
    held_out = held_out[: args.clips]
    if not held_out:
        print("no held-out clips", file=sys.stderr)
        return 1

    models = {
        "pretrained": LoFTRMatcher(),
        "fine-tuned": LoFTRMatcher(checkpoint=args.checkpoint),
    }
    rng = np.random.default_rng(args.seed)

    print(f"median endpoint error of the raw matches, {len(held_out)} held-out clips, "
          f"frame gap {args.spacing}\n")
    header = "size".ljust(8) + "".join(f"{name:>22s}" for name in models)
    print(header)
    print("-" * len(header))

    for size in args.sizes:
        label = "full" if size == 0 else str(size)
        cells = []
        for name, matcher in models.items():
            errors, counts = [], []
            for clip in held_out:
                with np.load(clip.path, allow_pickle=False) as data:
                    frames = data["frames"].astype(np.float32)
                    scat_x, scat_y = data["scat_x"], data["scat_y"]
                gap = min(args.spacing, frames.shape[0] - 1)
                error, count = match_error(
                    matcher, frames, scat_x, scat_y, (0, gap),
                    None if size == 0 else size, rng,
                )
                if np.isfinite(error):
                    errors.append(error)
                    counts.append(count)
            if errors:
                cells.append(f"{np.median(errors):8.2f} px ({int(np.mean(counts)):5d})")
            else:
                cells.append(f"{'no matches':>20s}")
        print(label.ljust(8) + "".join(f"{c:>22s}" for c in cells))

    print("\n(px = median endpoint error of the matches, lower is better; "
          "bracket = mean usable matches)")
    print("fine-tuned better at 320 but worse at full -> resolution specialisation.")
    print("fine-tuned worse everywhere -> the training itself, try a lower --lr.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
