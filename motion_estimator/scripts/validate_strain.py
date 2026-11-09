"""Check the strain port against every MITT result on disk.

MITT stores the tracked contour points and the strain curves it derived from
them side by side. A correct port must turn one into the other, so this is an
exact target on real data rather than a plausibility check::

    python -m motion_estimator.scripts.validate_strain
    python -m motion_estimator.scripts.validate_strain --variant bi_corrected

Any clip that does not reproduce to tolerance is a bug in the port, not noise.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from motion_estimator.loaders.dataset_index import DATASET_FOLDERS, index_dataset
from motion_estimator.loaders.annotations.mitt import (
    DEFAULT_MITT_ROOT,
    MITT_VARIANTS,
    mitt_path,
    read_mitt,
)
from motion_estimator.strain.curve import compute_strain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None, help="intervendor dataset root")
    parser.add_argument("--mitt-root", default=str(DEFAULT_MITT_ROOT))
    parser.add_argument("--variant", choices=sorted(MITT_VARIANTS), default="uni")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_FOLDERS))
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--layer", default="endo")
    args = parser.parse_args(argv)

    try:
        refs = index_dataset(args.root, tuple(args.datasets))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    checked, matched, missing, failures = 0, 0, 0, []
    worst = 0.0
    es_gap = []

    for ref in refs:
        path = mitt_path(
            ref.key.patient, ref.key.view, ref.key.dataset, args.variant, args.mitt_root
        )
        if not path.is_file():
            missing += 1
            continue
        try:
            mitt = read_mitt(path)
        except Exception as error:
            failures.append((ref.key, f"read: {type(error).__name__}: {error}"))
            continue
        if mitt.strain_curve is None:
            missing += 1
            continue

        result = compute_strain(
            mitt.points_x,
            mitt.points_y,
            mitt.es_frame,
            drift_comp=mitt.drift_compensated,
            open_contour=True,
            layer=args.layer,
        )
        got = result.curve.numpy()
        reference = mitt.strain_curve
        checked += 1

        if got.shape != reference.shape:
            failures.append((ref.key, f"shape {got.shape} vs {reference.shape}"))
            continue

        difference = float(np.nanmax(np.abs(got - reference)))
        worst = max(worst, difference)
        if difference <= args.tol:
            matched += 1
        else:
            failures.append((ref.key, f"max|diff| {difference:.3e}"))

        # MITT's own end-systolic value, for the record.
        if mitt.peak_value is not None:
            es_gap.append(result.value_es - mitt.peak_value)

    print(f"variant '{args.variant}', layer '{args.layer}', tol {args.tol:g}")
    print(f"  clips indexed        {len(refs)}")
    print(f"  MITT results found   {checked}")
    print(f"  no MITT result       {missing}")
    print(f"  reproduced exactly   {matched}/{checked}")
    print(f"  worst max|diff|      {worst:.3e}")

    if es_gap:
        gap = np.abs(np.asarray(es_gap))
        print(f"  |ours - stored timings[0]| at ES: max {gap.max():.3e}")

    if failures:
        print(f"\n{len(failures)} clip(s) did not reproduce:")
        for key, reason in failures[:25]:
            print(f"  {key}: {reason}")
        return 1

    print("\nport reproduces every stored MITT strain curve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
