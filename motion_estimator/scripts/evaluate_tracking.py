"""Measure any method, on either cohort, with end-systole reported first.

    python -m motion_estimator.scripts.evaluate_tracking --in-vivo
    python -m motion_estimator.scripts.evaluate_tracking --simulated --in-vivo \
        --methods loftr searaft --pairings consecutive reference strided
    python -m motion_estimator.scripts.evaluate_tracking --in-vivo --run-dir latest

Loading comes from :mod:`~motion_estimator.loaders.test_loader` and the
measurements from :mod:`~motion_estimator.evaluation.metrics`, so this script
only chooses what to run and where the answer goes. Both cohorts are reported,
and they answer different questions:

* **simulated** -- held-out clips with exact scatterer correspondence. The only
  place a tracking error can honestly be called an error, and the only place
  ``gain`` can be measured at all.
* **in-vivo** -- the GE cohort against MITT's contour. MITT is a speckle
  tracker with its own error, so a disagreement here is not necessarily ours.

Read ``ES err`` and ``ES cov`` before anything else. A median over all frames is
dominated by frames near end-diastole where motion is small and matching easy:
on IV02/4ch it read 3.65 px while end-systole -- the frame strain is read at --
was 7.48 px with 8 of 72 points surviving. Coverage sits beside every error for
the same reason, because an error over only the surviving points flatters a
method that loses the hard ones.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from motion_estimator.evaluation.metrics import evaluate, format_summary, summarise
from motion_estimator.runs import latest_run, write_config

RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "results"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "artifacts" / "simulated-cache"
from motion_estimator.methods.registry import METHODS
PAIRINGS = ("consecutive", "reference", "strided", "bidirectional")


def build_method(name, args, clip):
    """An estimator for one method on one clip, from the registry.

    Per clip rather than once, because the sector mask is per clip and the
    dense methods need it: they produce a value at every pixel including
    outside the sector, and the mask is what lets them answer "no answer".
    """
    from motion_estimator.methods.registry import build

    options = {}
    if name == "loftr":
        options.update(
            weights=args.weights, radius=args.radius,
            min_confidence=args.min_confidence, scale=args.scale,
        )
    else:
        options.update(consistency_px=args.consistency_px)
    return build(
        name,
        sector=clip.sector if args.use_mask else None,
        checkpoint=args.checkpoint if args.checkpoint_method == name else None,
        **options,
    )


def run_cohort(clips, label, args):
    rows = []
    from motion_estimator.tracking import chain

    for clip in clips:
        for method in args.methods:
            estimator = build_method(method, args, clip)
            for pairing in args.pairings:
                try:
                    result = chain(
                        clip.frames, *clip.start(), estimator, pairing=pairing,
                        reference_frame=clip.start_frame, stride=args.stride,
                    )
                except Exception as error:  # one bad clip must not end the sweep
                    print(f"  {method}:{pairing} {clip.key}: "
                          f"{type(error).__name__}: {error}", file=sys.stderr)
                    continue
                row = evaluate(result, clip, method=method, pairing=pairing)
                row["group"] = f"{method}:{pairing}"
                rows.append(row)
                print(f"  {label:9s} {clip.key[-18:]:18s} {row['group']:22s} "
                      f"ES {row['error_px_es']:6.2f} px "
                      f"({100 * row['coverage_es']:3.0f}% cov)  "
                      f"med {row['error_px_median']:5.2f}  "
                      f"closure {row['closure_px_mean']:6.2f}  "
                      f"gain {row['gain']:.3f}", file=sys.stderr)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--simulated", action="store_true")
    parser.add_argument("--in-vivo", action="store_true")
    parser.add_argument("--methods", nargs="+", default=["loftr"], choices=list(METHODS))
    parser.add_argument("--pairings", nargs="+",
                        default=["consecutive", "reference"], choices=PAIRINGS)
    parser.add_argument("--checkpoint", default=None,
                        help="a fine-tuned LoFTR .pt; omit for pretrained")
    parser.add_argument("--weights", default="outdoor", choices=("outdoor", "indoor"))
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--min-confidence", type=float, default=0.2)
    parser.add_argument("--radius", type=float, default=16.0,
                        help="LoFTR interpolation radius; 16 measured better than 32")
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--consistency-px", type=float, default=2.0,
                        help="dense-flow forward-backward tolerance in px; 0 disables")
    parser.add_argument("--checkpoint-method", default="loftr", choices=METHODS,
                        help="which method --checkpoint belongs to")
    parser.add_argument("--use-mask", action="store_true", default=True)
    parser.add_argument("--no-mask", dest="use_mask", action="store_false")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--dataset", default="Retest_dataset",
                        help="Test_dataset is held out; see the split protocol")
    parser.add_argument("--view", default="4ch")
    parser.add_argument("--mitt-variant", default="uni")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", default=None,
                        help="a run directory; results land inside it and "
                             "--checkpoint defaults to its best.pt. "
                             "'latest' picks the newest run.")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    if not args.simulated and not args.in_vivo:
        parser.error("pass --simulated, --in-vivo, or both")
    if args.consistency_px == 0:
        args.consistency_px = None

    run_dir = None
    if args.run_dir:
        run_dir = latest_run() if args.run_dir == "latest" else Path(args.run_dir)
        if run_dir is None or not run_dir.is_dir():
            parser.error(f"no run directory at {args.run_dir!r}")
        if args.checkpoint is None and (run_dir / "best.pt").is_file():
            args.checkpoint = str(run_dir / "best.pt")
            print(f"using checkpoint {args.checkpoint}")

    from motion_estimator.loaders.test_loader import load_in_vivo, load_simulated

    report = {"args": {key: str(value) for key, value in vars(args).items()}}
    cohorts = (
        (args.simulated, "simulated",
         lambda: load_simulated(args.cache_dir, args.val_fraction, args.seed,
                                args.limit, args.n_points)),
        (args.in_vivo, "in-vivo",
         lambda: load_in_vivo(args.dataset, args.view, args.mitt_variant, args.limit)),
    )

    for enabled, label, loader in cohorts:
        if not enabled:
            continue
        try:
            clips = loader()
        except FileNotFoundError as error:
            print(f"{label}: {error}", file=sys.stderr)
            continue
        if not clips:
            print(f"{label}: no clips found", file=sys.stderr)
            continue
        print(f"\n{label}: {len(clips)} clips, {len(args.methods)} method(s) "
              f"x {len(args.pairings)} pairing(s)")
        rows = run_cohort(clips, label, args)
        if not rows:
            continue
        report[label] = {"rows": rows, "summary": summarise(rows, by="group")}
        print()
        print(format_summary(report[label]["summary"], f"=== {label} ==="))
        # After each cohort rather than only at the end: a pass over both is
        # tens of minutes, and losing the first because the second failed is
        # avoidable.
        _write(report, args, run_dir)

    _write(report, args, run_dir)
    print(f"\nwrote {args.json}")
    if run_dir is not None:
        write_config(run_dir, {"stage": "evaluate", **vars(args)})
    return 0


def _write(report, args, run_dir):
    if args.json is None:
        tag = args.tag or ("finetuned" if args.checkpoint else "pretrained")
        base = run_dir if run_dir is not None else RESULTS_DIR
        base.mkdir(parents=True, exist_ok=True)
        args.json = str(base / f"eval_{'-'.join(args.methods)}_{tag}.json")
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
