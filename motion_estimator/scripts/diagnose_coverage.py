"""Can preprocessing get LoFTR to match near the contour at all?

Point loss is the binding constraint in vivo: roughly 576 of 720 tracked points
are touched by NaN, and on IV02/4ch **100% of those cells had ``support == 0``**
-- no matches within the interpolation radius. Nothing is being rejected by the
guards; the matcher simply does not fire there. So the lever is the image, not
the interpolation or the model.

Four things worth trying, measured against each other on real clips:

``none``
    The frames as the pipeline uses them now.
``clahe``
    Contrast-limited adaptive histogram equalisation. Echo has a large dynamic
    range and the endocardial border is a low-contrast edge between two dim
    regions; LoFTR was trained on photographs, where edges are not.
``gamma``
    A global mid-tone lift. Cheaper than CLAHE and does not amplify noise
    locally, which is the risk CLAHE carries in speckle.
``upsample``
    Match at 2x and scale the coordinates back. LoFTR's coarse grid is a fixed
    1/8 of the input, so upsampling halves the anatomical spacing between
    candidate matches -- twice as many chances to land near the contour. It
    also doubles every displacement in pixels, which moves the matcher off the
    part of its range where it shrinks small offsets (see ``diagnose_gain``).

Reported per variant: **coverage** (the fraction of tracked positions that are
finite, which is the number under attack), agreement with MITT, and cycle
closure. A variant that improves coverage while worsening agreement has bought
points that are wrong, so both are shown.

    python -m motion_estimator.scripts.diagnose_coverage
    python -m motion_estimator.scripts.diagnose_coverage --variants none upsample
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

VARIANTS = ("none", "clahe", "gamma", "upsample")


def preprocess(frames, variant, factor=2):
    """Return (frames, scale) -- *scale* maps original pixels to the output."""
    if variant == "none":
        return frames, 1.0

    if variant == "upsample":
        import cv2

        out = np.stack(
            [
                cv2.resize(
                    f, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC
                )
                for f in frames
            ]
        )
        return out, float(factor)

    # Both remaining variants are intensity-only, so they need a common 0-255
    # footing first; the clip's own range, not each frame's, or the mapping
    # would change between frames.
    low, high = float(np.nanmin(frames)), float(np.nanmax(frames))
    scaled = np.clip((frames - low) / max(high - low, 1e-9), 0, 1)

    if variant == "gamma":
        return (scaled ** 0.5).astype(np.float32), 1.0

    if variant == "clahe":
        import cv2

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = np.stack(
            [clahe.apply((f * 255).astype(np.uint8)).astype(np.float32) / 255.0
             for f in scaled]
        )
        return out, 1.0

    raise ValueError(f"unknown variant {variant!r}")


def run_clip(ref, matcher, variant, args):
    from motion_estimator.loaders.images.image_prep import frames_first, prepare_clip
    from motion_estimator.loaders.annotations.mitt import mitt_path, read_mitt
    from motion_estimator.loaders.annotations.scar_struct import read_view_analysis
    from motion_estimator.loaders.images.srd_reader import read_clip
    from motion_estimator.methods.loftr import track_points

    path = mitt_path(ref.key.patient, ref.key.view, ref.key.dataset, args.mitt_variant)
    if not path.is_file():
        return None
    mitt = read_mitt(path)
    analysis = read_view_analysis(ref.scar_struct_path, ref.key.view)
    prepared = prepare_clip(
        read_clip(ref.clip_path), cycle_frame_limits=analysis.cycle_frame_limits
    )
    frames = frames_first(prepared.images).astype(np.float32)
    spacing_rc = np.asarray(prepared.pixel_spacing_mm).ravel()[:2]
    mitt_x = mitt.points_y / spacing_rc[1] + 1.0
    mitt_y = mitt.points_x / spacing_rc[0] + 1.0
    start = mitt.ref_frame - 1

    processed, scale = preprocess(frames, variant, args.upsample_factor)
    result = track_points(
        processed,
        (mitt_x[start] - 1.0) * scale + 1.0,
        (mitt_y[start] - 1.0) * scale + 1.0,
        matcher,
        pairing="reference",
        reference_frame=start,
        radius=args.radius * scale,
    )
    # Back to original pixels before comparing with anything.
    x = (result.x - 1.0) / scale + 1.0
    y = (result.y - 1.0) / scale + 1.0

    finite = np.isfinite(x) & np.isfinite(y)
    agreement = np.hypot(x - mitt_x, y - mitt_y)
    closure = np.hypot(x[-1] - x[0], y[-1] - y[0])
    return {
        "key": str(ref.key),
        "variant": variant,
        "coverage": float(finite.mean()),
        "vs_mitt_px_median": float(np.nanmedian(agreement)),
        "closure_px_mean": float(np.nanmean(closure)),
        "matches_median": int(np.median(result.n_matches)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                        choices=list(VARIANTS))
    parser.add_argument("--dataset", default="Retest_dataset")
    parser.add_argument("--view", default="4ch")
    parser.add_argument("--mitt-variant", default="uni")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--radius", type=float, default=16.0)
    parser.add_argument("--upsample-factor", type=int, default=2)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    from motion_estimator.loaders.dataset_index import index_dataset
    from motion_estimator.methods.loftr import LoFTRMatcher

    refs = [r for r in index_dataset(datasets=(args.dataset,)) if r.is_complete]
    if args.view:
        refs = [r for r in refs if r.key.view == args.view]
    refs = refs[: args.limit]
    if not refs:
        print("no clips", file=sys.stderr)
        return 1

    matcher = LoFTRMatcher(checkpoint=args.checkpoint)
    rows = []
    for variant in args.variants:
        for ref in refs:
            row = run_clip(ref, matcher, variant, args)
            if row is None:
                continue
            rows.append(row)
            print(f"  {variant:9s} {row['key'][15:]:14s} "
                  f"coverage {100 * row['coverage']:5.1f}%  "
                  f"vs MITT {row['vs_mitt_px_median']:5.2f} px  "
                  f"closure {row['closure_px_mean']:6.2f} px  "
                  f"matches {row['matches_median']}", file=sys.stderr)

    print(f"\n{'variant':10s} {'coverage':>9s} {'vs MITT':>9s} {'closure':>9s} "
          f"{'matches':>9s}")
    for variant in args.variants:
        subset = [r for r in rows if r["variant"] == variant]
        if not subset:
            continue
        print(f"{variant:10s} "
              f"{100 * np.median([r['coverage'] for r in subset]):8.1f}% "
              f"{np.median([r['vs_mitt_px_median'] for r in subset]):9.2f} "
              f"{np.nanmedian([r['closure_px_mean'] for r in subset]):9.2f} "
              f"{int(np.median([r['matches_median'] for r in subset])):9d}")

    print("\ncoverage is the fraction of tracked positions that are finite; "
          "it is the constraint. A variant that lifts coverage but worsens "
          "agreement has bought points that are wrong.")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
