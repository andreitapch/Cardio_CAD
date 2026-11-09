"""Is the drift a systematic bias or a random walk?

Cycle-closure error on the in-vivo cohort is 7.24 px median and 14.30 px on
IV07 -- larger than the disagreement with MITT itself. But "drift" covers two
different failures with different fixes, and the remedy depends on which one
this is:

* **Bias.** Every step is slightly short (or long) in the same direction, so
  the error grows linearly with frame count. Bidirectional tracking cancels it,
  because the forward and backward passes are wrong in opposite directions.
* **Random walk.** Each step is unbiased but noisy, and the errors accumulate
  as the square root of the step count. Bidirectional helps only by a factor of
  about sqrt(2); what actually helps is rejecting bad steps and smoothing.

The simulated cohort separates them, because the true per-step displacement is
known exactly::

    python -m motion_estimator.scripts.diagnose_drift
    python -m motion_estimator.scripts.diagnose_drift --checkpoint <run>/best.pt

The test is the ratio of observed closure to the two predictions. A linear
accumulation of the measured per-step bias predicts the bias case; the measured
per-step noise times sqrt(n_steps) predicts the random-walk case. Whichever
prediction lands closer to the observed closure is the mechanism in play.

Also reported is the **gain**: the ratio of predicted to true step magnitude.
A gain below 1 means the tracker systematically under-predicts motion, which is
the specific failure that makes strain read low while cycle closure looks good.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "artifacts" / "simulated-cache"


def analyse_clip(matcher, frames, truth_x, truth_y, radius):
    """Per-step error decomposition for one clip, tracked consecutively."""
    from motion_estimator.methods.loftr import track_points

    result = track_points(
        frames, truth_x[0], truth_y[0], matcher, pairing="consecutive", radius=radius
    )

    # Per-step displacement the tracker produced, and the truth for that step.
    predicted = np.stack(
        [np.diff(result.x, axis=0), np.diff(result.y, axis=0)], axis=-1
    )
    true = np.stack(
        [np.diff(truth_x, axis=0), np.diff(truth_y, axis=0)], axis=-1
    )
    finite = np.isfinite(predicted).all(axis=-1)
    if not finite.any():
        return None

    magnitude = np.linalg.norm(true, axis=-1)
    moving = finite & (magnitude > 0.2)  # a still point says nothing about gain
    if not moving.any():
        return None

    # Gain: how much of the true motion the tracker reproduces, measured along
    # the true direction so that perpendicular noise does not inflate it.
    direction = true / np.maximum(magnitude[..., None], 1e-9)
    along = (predicted * direction).sum(-1)
    gain = float(np.median(along[moving] / magnitude[moving]))

    error = predicted - true
    # Signed error along the direction of travel is the part that accumulates
    # coherently; its median over all steps and points is the bias per step.
    bias_per_step = float(np.median((error * direction).sum(-1)[moving]))
    noise_per_step = float(
        np.median(np.abs((error * direction).sum(-1)[moving] - bias_per_step))
        * 1.4826
    )

    n_steps = int(predicted.shape[0])
    observed = float(
        np.nanmedian(
            np.hypot(result.x[-1] - result.x[0], result.y[-1] - result.y[0])
        )
    )
    return {
        "n_steps": n_steps,
        "gain": gain,
        "bias_per_step_px": bias_per_step,
        "noise_per_step_px": noise_per_step,
        "closure_observed_px": observed,
        "closure_if_bias_px": abs(bias_per_step) * n_steps,
        "closure_if_random_walk_px": noise_per_step * np.sqrt(n_steps),
        "lost": int((result.lost >= 0).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--clips", type=int, default=6)
    parser.add_argument("--n-points", type=int, default=150)
    parser.add_argument("--radius", type=float, default=32.0)
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
    rows = []
    print(f"{'clip':40s} {'gain':>6s} {'bias/step':>10s} {'noise/step':>11s} "
          f"{'closure':>8s} {'if bias':>8s} {'if walk':>8s}")
    for clip in held_out:
        with np.load(clip.path, allow_pickle=False) as data:
            frames = data["frames"].astype(np.float32)
            scat_x, scat_y = data["scat_x"], data["scat_y"]
        columns = np.linspace(0, scat_x.shape[1] - 1, args.n_points).astype(int)
        stats = analyse_clip(
            matcher, frames, scat_x[:, columns], scat_y[:, columns], args.radius
        )
        if stats is None:
            continue
        stats["key"] = clip.key
        rows.append(stats)
        print(f"{clip.key[:40]:40s} {stats['gain']:6.3f} "
              f"{stats['bias_per_step_px']:+10.3f} {stats['noise_per_step_px']:11.3f} "
              f"{stats['closure_observed_px']:8.2f} {stats['closure_if_bias_px']:8.2f} "
              f"{stats['closure_if_random_walk_px']:8.2f}")

    if not rows:
        print("nothing measurable", file=sys.stderr)
        return 1

    def median(field):
        return float(np.median([r[field] for r in rows]))

    observed = median("closure_observed_px")
    from_bias = median("closure_if_bias_px")
    from_walk = median("closure_if_random_walk_px")
    print("\n=== medians ===")
    print(f"  gain (predicted / true step)   {median('gain'):.3f}")
    print(f"  bias per step                  {median('bias_per_step_px'):+.3f} px")
    print(f"  noise per step                 {median('noise_per_step_px'):.3f} px")
    print(f"  closure observed               {observed:.2f} px")
    print(f"  closure predicted by bias      {from_bias:.2f} px")
    print(f"  closure predicted by walk      {from_walk:.2f} px")

    verdict = "bias" if abs(from_bias - observed) < abs(from_walk - observed) else "random walk"
    print(f"\n  -> dominated by {verdict.upper()}")
    if verdict == "bias":
        print("     Bidirectional tracking cancels this; it is the right first fix.")
    else:
        print("     Bidirectional buys only ~sqrt(2). Reject bad steps "
              "(forward-backward consistency) and smooth instead.")
    if median("gain") < 0.95:
        print(f"     Gain {median('gain'):.3f} < 1: motion is systematically "
              "under-predicted, which reads as low strain.")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
