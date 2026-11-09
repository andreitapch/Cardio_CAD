"""Print what the dataset tree actually contains. Run this first.

Everything downstream is sized by the answer -- how many patients per folder,
which views exist, which clips are missing ground truth. Run it as soon as the
dataset drive is available::

    python -m motion_estimator.scripts.report_cohort
    python -m motion_estimator.scripts.report_cohort --root E:\\path\\to\\dataset
    python -m motion_estimator.scripts.report_cohort --csv cohort.csv

``--probe`` additionally opens every complete clip and reads its cycle window,
which is slow (each ``data.mat`` is ~15 MB) but is the real check that the
readers cope with the whole cohort rather than only IV02.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from motion_estimator.loaders.dataset_index import (
    CLIP_SOURCES,
    DATA_MAT,
    DATASET_FOLDERS,
    DEFAULT_VIEWS,
    index_dataset,
    manifest,
    summarize,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        default=None,
        help="dataset root; defaults to $MOTION_ESTIMATOR_DATA_ROOT then the F: path",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_FOLDERS),
        help="dataset folders to scan",
    )
    parser.add_argument("--vendor", default="GE", help="ground-truth vendor")
    parser.add_argument(
        "--views",
        nargs="+",
        default=list(DEFAULT_VIEWS),
        help=(
            "cardiac views to index; 'all' indexes every view on disk, which "
            "includes Test_dataset's 3ch clips that scar_struct.mat does not "
            "define a cycle window for"
        ),
    )
    parser.add_argument(
        "--clip-source",
        choices=CLIP_SOURCES,
        default=DATA_MAT,
        help=(
            "which export to read; data.mat is the only one with a uniform "
            "geometry across the cohort, so mixing is a hazard"
        ),
    )
    parser.add_argument("--csv", default=None, help="also write the manifest here")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="open every complete clip and report its cycle window (slow)",
    )
    args = parser.parse_args(argv)

    views = None if args.views == ["all"] else tuple(args.views)

    try:
        refs = index_dataset(
            args.root,
            tuple(args.datasets),
            args.vendor,
            views=views,
            # The probe is about to open these files anyway, so the extra
            # scar_struct read costs nothing and turns a crash into a flag.
            verify_views=args.probe,
            clip_source=args.clip_source,
        )
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    print(summarize(refs))

    if args.csv:
        frame = manifest(refs)
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)
        print(f"\nmanifest written to {args.csv} ({len(frame)} rows)")

    if args.probe:
        print()
        _probe(refs)

    return 0


def _probe(refs) -> None:
    """Open each complete clip, so reader failures surface as a list, not a crash."""
    from motion_estimator.loaders.images.image_prep import prepare_clip
    from motion_estimator.loaders.annotations.scar_struct import read_view_analysis
    from motion_estimator.loaders.images.srd_reader import read_clip

    failures = []
    for ref in (ref for ref in refs if ref.is_complete):
        try:
            analysis = read_view_analysis(ref.scar_struct_path, ref.key.view)
            prepared = prepare_clip(
                read_clip(ref.clip_path),
                cycle_frame_limits=analysis.cycle_frame_limits,
            )
        except Exception as error:  # keep going; the summary is the point
            failures.append((ref.key, f"{type(error).__name__}: {error}"))
            print(f"  FAIL {ref.key}: {type(error).__name__}: {error}")
            continue

        first, last = analysis.cycle_frame_limits
        spacing = prepared.pixel_spacing_mm
        print(
            f"  ok   {ref.key}: {prepared.images.shape} "
            f"cycle {first}-{last} ({prepared.n_frames} frames) "
            f"spacing {spacing[0]:.3f}x{spacing[1]:.3f} mm "
            f"[{ref.clip_source}]"
        )

    print(f"\n{len(failures)} clip(s) failed to load")
    for key, message in failures:
        print(f"  {key}: {message}")


if __name__ == "__main__":
    raise SystemExit(main())
