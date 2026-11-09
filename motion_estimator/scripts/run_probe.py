"""Baseline probes on the cached embeddings.

    python -m motion_estimator.scripts.run_probe
    python -m motion_estimator.scripts.run_probe --grouping patient
    python -m motion_estimator.scripts.run_probe --dev-only --alpha 1 10 100

Development runs use ``Retest_dataset`` only; the Test folder is holdout and is
touched solely with ``--final``, which exists to make using it a deliberate act.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from motion_estimator.loaders.dataset_index import index_dataset
from motion_estimator.echoclip.config import DEFAULT_CACHE_DIR
from motion_estimator.echoclip.embed_cache import GROUPINGS, load_embeddings
from motion_estimator.echoclip.probe import run_cv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--phase", default="end_systolic")
    parser.add_argument("--grouping", choices=GROUPINGS, default="dataset_patient")
    parser.add_argument("--alpha", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--dev-only",
        action="store_true",
        default=True,
        help="restrict to Retest_dataset (the default)",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="ALSO use the held-out Test_dataset. Deliberate act; not for iteration.",
    )
    args = parser.parse_args(argv)

    try:
        refs = index_dataset(args.root)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    data = load_embeddings(refs, args.cache_dir, args.phase, args.grouping)
    if not args.final:
        data = data.subset(data.datasets == "Retest_dataset")

    print(f"clips {len(data)}  groups {data.n_groups}  patients {data.n_patients}")
    print(f"grouping '{args.grouping}'  phase '{data.phase}'")
    print(f"encoder  {data.config}")
    print("target support (clips with a real value):")
    from motion_estimator.echoclip.metrics import TARGET_LABELS

    print(
        "  "
        + "  ".join(
            f"{label}={count}" for label, count in zip(TARGET_LABELS, data.support())
        )
    )
    if args.final:
        print("\n*** using the held-out Test_dataset ***")

    print("\n" + "=" * 78)
    baseline = run_cv(data, "mean", n_splits=args.n_splits)
    print(baseline.report())

    for alpha in args.alpha:
        print()
        result = run_cv(data, "ridge", alpha=alpha, n_splits=args.n_splits)
        print(result.report())
        gls_gain = baseline.metrics[-1].mae - result.metrics[-1].mae
        print(f"  -> GLS MAE vs mean-only baseline: {gls_gain:+.3f} (positive is better)")

    print("\n" + "=" * 78)
    print(
        "Every number is out-of-fold under grouped CV. Alpha is being *read off*\n"
        "these same folds, so treat the best one as optimistic until it is chosen\n"
        "in an inner loop."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
