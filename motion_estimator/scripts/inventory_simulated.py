"""What ``realistic_dataset_v1`` actually contains.

Reads metadata only -- no pixels, no scatterers -- so it can sweep 1.5 TB in
minutes instead of hours::

    python -m motion_estimator.scripts.inventory_simulated
    python -m motion_estimator.scripts.inventory_simulated --probe        # opens every file
    python -m motion_estimator.scripts.inventory_simulated --csv sim.csv

Worth running before the cache builder: it reports which ``sim_type`` each
patient actually has (the readme lists four, the tree does not always carry
them), the frame counts, and the EF distribution -- the CAMUS selection was
biased to reduced ejection fraction, which bounds what a model trained on this
can be expected to generalise to.
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys

import numpy as np

from motion_estimator.loaders.images.simulated import (
    index_simulated,
    read_simulated_meta,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None, help="dataset root")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="open every file for shapes, timing and EF (slower, still no pixels)",
    )
    parser.add_argument("--limit", type=int, default=None, help="probe only N files")
    parser.add_argument("--csv", default=None, help="write the per-clip table here")
    args = parser.parse_args(argv)

    try:
        refs = index_simulated(args.root)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    if not refs:
        print("no recordings found", file=sys.stderr)
        return 1

    patients = sorted({r.patient for r in refs})
    by_type = collections.Counter(r.sim_type for r in refs)
    by_view = collections.Counter(r.view for r in refs)
    types_per_patient = collections.defaultdict(set)
    for ref in refs:
        types_per_patient[ref.patient].add(ref.sim_type)

    print(f"{len(refs)} recordings, {len(patients)} patients\n")
    print("by simulation type")
    for name, count in by_type.most_common():
        holders = sum(1 for types in types_per_patient.values() if name in types)
        print(f"  {name:24s} {count:5d} recordings  {holders:3d}/{len(patients)} patients")
    print("\nby view")
    for name, count in sorted(by_view.items()):
        print(f"  {name:6s} {count:5d}")

    coverage = collections.Counter(
        tuple(sorted(types)) for types in types_per_patient.values()
    )
    print("\npatients by which types they hold")
    for combination, count in coverage.most_common():
        print(f"  {count:3d} patients: {', '.join(combination)}")

    if not args.probe:
        print("\n(metadata not read -- pass --probe for frame counts, EF and timing)")
        return 0

    rows, failures = [], []
    targets = refs if args.limit is None else refs[: args.limit]
    for index, ref in enumerate(targets, 1):
        try:
            meta = read_simulated_meta(ref.path)
        except Exception as error:  # a corrupt file should not stop the sweep
            failures.append((ref.key, f"{type(error).__name__}: {error}"))
            continue
        meta.update(
            key=ref.key,
            patient=ref.patient,
            sim_type=ref.sim_type,
            view=ref.view,
            sim_no=ref.sim_no,
        )
        rows.append(meta)
        if index % 25 == 0:
            print(f"  probed {index}/{len(targets)}", file=sys.stderr)

    if not rows:
        print("\nno file could be read", file=sys.stderr)
        return 1

    def column(name):
        return np.array([r[name] for r in rows if r.get(name) is not None], dtype=float)

    frames = column("n_frames")
    scatterers = column("n_myocardial_scatterers")
    ef = column("ejection_fraction")
    size = column("file_gb")
    print(f"\nprobed {len(rows)} files, {len(failures)} unreadable")
    print(f"  frames per clip     {frames.min():.0f}-{frames.max():.0f} "
          f"(median {np.median(frames):.0f})")
    print(f"  myocardial scatts   {scatterers.min():.0f}-{scatterers.max():.0f}")
    print(f"  frame size          {rows[0]['rows']} x {rows[0]['cols']} "
          f"@ {rows[0]['dy_mm']:.4f} x {rows[0]['dx_mm']:.4f} mm")
    print(f"  ejection fraction   {ef.min():.1f}-{ef.max():.1f} "
          f"(median {np.median(ef):.1f})")
    print(f"  file size           {size.mean():.2f} GB mean, "
          f"{size.sum():.0f} GB over the probed set")
    print(f"  consecutive pairs   {int(frames.sum() - len(frames))} in the probed set")

    for key, reason in failures[:10]:
        print(f"  FAILED {key}: {reason}", file=sys.stderr)

    if args.csv:
        fields = sorted({k for row in rows for k in row})
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fields})
        print(f"\nwrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
