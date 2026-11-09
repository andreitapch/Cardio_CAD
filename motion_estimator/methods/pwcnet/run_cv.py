"""The whole cross-validation, one command, resumable.

    python -m motion_estimator.methods.pwcnet.run_cv                       # new run, all folds
    python -m motion_estimator.methods.pwcnet.run_cv --run-dir <cv>        # resume
    python -m motion_estimator.methods.pwcnet.run_cv --folds 0 --quick     # smoke test (minutes)

Stages per fold, each skipped when its ``last.pt`` already exists:

1. ``sim``   -- pretrain on the fold's simulated training patients
2. ``unsup`` -- fine-tune on in-vivo training patients, photometric loss
3. ``mitt``  -- fine-tune on in-vivo training patients, MITT point loss
4. ``eval``  -- track the fold's held-out patients with all three, plot

then one summary across folds. Every stage is also its own script
(``data``, ``train``, ``evaluate``) if you would rather run them by hand.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ...runs import create_run
from . import evaluate, train
from .data import INVIVO_CACHE, build_invivo_cache, list_invivo_cached, make_folds

STAGES = ("sim", "unsup", "mitt", "eval")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", default=None, help="resume this CV directory")
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--sim-steps", type=int, default=None)
    parser.add_argument("--ft-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pairings", nargs="+", default=["consecutive"])
    parser.add_argument("--quick", action="store_true",
                        help="tiny step counts to check the pipeline end to end")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)

    if args.quick:
        args.sim_steps = args.sim_steps or 200
        args.ft_steps = args.ft_steps or 100

    run_dir = Path(args.run_dir) if args.run_dir else create_run(
        "pwcnet-cv-quick" if args.quick else "pwcnet-cv")
    print(f"cross-validation directory: {run_dir}")

    if not list_invivo_cached(INVIVO_CACHE):
        print("building the in-vivo cache (once)...")
        build_invivo_cache()
    if not (run_dir / "folds.json").is_file():
        make_folds(run_dir, k=args.k)

    folds = args.folds if args.folds is not None else range(args.k)
    common = ["--run-dir", str(run_dir), "--batch-size", str(args.batch_size),
              "--workers", str(args.workers)]
    if args.quick:
        common += ["--val-every", "50", "--val-batches", "5"]

    for fold in folds:
        fold_args = common + ["--fold", str(fold)]
        jobs = {
            "sim": (["--cohort", "simulated"], args.sim_steps, "simulated", "points"),
            "unsup": (["--cohort", "in_vivo", "--mode", "unsup"], args.ft_steps,
                      "in_vivo", "unsup"),
            "mitt": (["--cohort", "in_vivo", "--mode", "mitt"], args.ft_steps,
                     "in_vivo", "mitt"),
        }
        for stage, (extra, steps, cohort, mode) in jobs.items():
            if stage not in args.stages:
                continue
            if (train.output_dir(run_dir, cohort, fold, mode) / "last.pt").is_file():
                print(f"fold {fold} {stage}: done already, skipping")
                continue
            steps_arg = ["--steps", str(steps)] if steps else []
            print(f"\n===== fold {fold}: {stage} =====")
            train.main(fold_args + extra + steps_arg)
        if "eval" in args.stages:
            print(f"\n===== fold {fold}: eval =====")
            evaluate.main(["--run-dir", str(run_dir), "--folds", str(fold),
                           "--pairings", *args.pairings, "--no-summary"]
                          + (["--sim-limit", "3"] if args.quick else []))

    evaluate.main(["--run-dir", str(run_dir), "--summarise-only"])
    print(f"\ndone. open notebooks/pwcnet_results.ipynb with RUN_DIR = {run_dir!s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
