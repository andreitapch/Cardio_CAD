"""One pass over ``realistic_dataset_v1``, writing a trainable cache.

Why a cache at all: the cohort is 1.5 TB, but almost all of that is
``RF_image``, which motion estimation does not use. Dropping it and storing
B-mode as ``uint8`` at a chosen pixel size, with a subset of the myocardial
scatterer tracks, turns 1.5 TB of HDF5 on an external drive into a few tens of
GB that a training loop can read at speed::

    python -m motion_estimator.scripts.build_simulated_cache
    python -m motion_estimator.scripts.build_simulated_cache --pixel-mm 0.26 --max-scatterers 20000
    python -m motion_estimator.scripts.build_simulated_cache --sim-types regional_simulations

One ``.npz`` per recording, so an interrupted pass resumes and a failed file
costs only itself. A ``meta.json`` records the settings every file was written
under; a cache built with different ones is refused rather than silently mixed,
the same contract as ``echoclip/embed_cache.py``.

Two choices that are settings rather than conventions, because both should be
measured rather than assumed:

* **``--pixel-mm``** resamples to a common pixel size. The simulations are
  ~0.195 mm and the GE in-vivo clips 0.26 x 0.32 mm, so a model trained at the
  simulated scale sees speckle at the wrong size. The default matches the
  in-vivo axial spacing.
* **``--max-scatterers``** subsamples the 79 k-127 k myocardial scatterers,
  evenly rather than randomly so repeated builds agree. 20 k over the wall is
  already several times LoFTR's coarse cell count.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from motion_estimator.loaders.images.simulated import (
    SimClipRef,
    index_simulated,
    read_simulated,
)

DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "artifacts" / "simulated-cache"
META_FILENAME = "meta.json"


def cache_path(cache_dir: Path, ref: SimClipRef) -> Path:
    return cache_dir / f"Pat_{ref.patient}" / (
        f"{ref.sim_type}_{ref.view}_sim{ref.sim_no}.npz"
    )


def fingerprint(args) -> dict:
    return {
        "pixel_mm": args.pixel_mm,
        "max_scatterers": args.max_scatterers,
        "orientation": args.orientation,
        "layout": "frames uint8 (n, rows, cols); scat_x/scat_y float32 1-based pixels",
    }


def resample(frames: np.ndarray, source_mm, target_mm: float):
    """Resample to an isotropic *target_mm*, keeping NaN outside the sector.

    Order-1 interpolation on the frames; NaN is carried by resampling a
    validity mask alongside and re-applying it, because ``map_coordinates``
    would otherwise smear the NaN border inwards over several pixels.
    """
    from scipy import ndimage

    source_mm = np.asarray(source_mm, dtype=float)
    if target_mm is None:
        return frames, source_mm

    scale = source_mm / float(target_mm)  # (row, col) output-per-input
    rows = max(int(round(frames.shape[1] * scale[0])), 1)
    cols = max(int(round(frames.shape[2] * scale[1])), 1)

    row_grid = np.linspace(0, frames.shape[1] - 1, rows)
    col_grid = np.linspace(0, frames.shape[2] - 1, cols)
    mesh = np.meshgrid(row_grid, col_grid, indexing="ij")
    coords = np.stack(mesh)

    out = np.empty((frames.shape[0], rows, cols), dtype=np.float32)
    for index, frame in enumerate(frames):
        valid = np.isfinite(frame)
        filled = np.where(valid, frame, 0.0)
        sampled = ndimage.map_coordinates(filled, coords, order=1, mode="nearest")
        keep = ndimage.map_coordinates(
            valid.astype(np.float32), coords, order=1, mode="nearest"
        )
        out[index] = np.where(keep > 0.999, sampled, np.nan)

    return out, np.array([target_mm, target_mm])


def to_uint8(frames: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Quantise to uint8 over the clip's own finite range, 0 outside the sector.

    One range for the whole clip, not per frame: per-frame normalisation would
    rescale the image between frames, which is a change in appearance a matcher
    can see and would learn to chase.
    """
    finite = np.isfinite(frames)
    if not finite.any():
        raise ValueError("every pixel is NaN")
    low = float(np.nanmin(frames))
    high = float(np.nanmax(frames))
    if not np.isfinite(low) or high <= low:
        raise ValueError(f"degenerate intensity range ({low}, {high})")
    scaled = (frames - low) / (high - low)
    # 0 is reserved for "outside the sector", so the data starts at 1.
    quantised = np.where(finite, 1 + np.clip(scaled, 0, 1) * 254, 0)
    return quantised.astype(np.uint8), (low, high)


def build_one(ref: SimClipRef, args) -> dict:
    clip = read_simulated(
        ref.path,
        max_scatterers=args.max_scatterers,
        orientation=args.orientation,
        ref=ref,
    )
    frames, pixel_mm = resample(clip.frames, clip.pixel_size_mm, args.pixel_mm)

    scale = np.asarray(clip.pixel_size_mm, dtype=float) / np.asarray(pixel_mm, float)
    # 1-based pixel coordinates scale about the origin at 0.5, not 0.
    scat_x = (clip.scat_x - 0.5) * scale[1] + 0.5
    scat_y = (clip.scat_y - 0.5) * scale[0] + 0.5

    quantised, (low, high) = to_uint8(frames)
    return {
        "frames": quantised,
        "scat_x": scat_x.astype(np.float32),
        "scat_y": scat_y.astype(np.float32),
        "pixel_mm": np.asarray(pixel_mm, dtype=np.float32),
        "intensity_range": np.array([low, high], dtype=np.float32),
        "ed_frame": np.int32(clip.ed_frame),
        "es_frame": np.int32(clip.es_frame),
        "frame_interval_s": np.float32(clip.frame_interval_s),
        "ejection_fraction": np.float32(clip.ejection_fraction or np.nan),
        "heart_rate": np.float32(clip.heart_rate or np.nan),
        "long_strain": np.asarray(
            clip.long_strain if clip.long_strain is not None else [], dtype=np.float32
        ),
        "rad_strain": np.asarray(
            clip.rad_strain if clip.rad_strain is not None else [], dtype=np.float32
        ),
        "key": np.array(ref.key),
        "group_id": np.array(ref.group_id),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--pixel-mm", type=float, default=0.26,
                        help="resample to this isotropic size; 0 keeps the native grid")
    parser.add_argument("--max-scatterers", type=int, default=20000)
    parser.add_argument("--orientation", default="clinical",
                        choices=("clinical", "native"))
    parser.add_argument("--sim-types", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="rebuild files that already exist")
    args = parser.parse_args(argv)
    if args.pixel_mm is not None and args.pixel_mm <= 0:
        args.pixel_mm = None

    try:
        refs = index_simulated(args.root, args.sim_types)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    if args.limit:
        refs = refs[: args.limit]

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    settings = fingerprint(args)

    meta_path = cache_dir / META_FILENAME
    if meta_path.is_file():
        existing = json.loads(meta_path.read_text(encoding="utf-8")).get("settings")
        if existing and existing != settings and not args.overwrite:
            print(
                "cache at this path was built with different settings; mixing them "
                "would produce a silently inconsistent dataset.\n"
                f"  on disk:  {existing}\n  requested: {settings}\n"
                "Use a different --cache-dir, or --overwrite.",
                file=sys.stderr,
            )
            return 1

    # Written before the loop as well as after it. A pass over 798 clips takes
    # hours and is meant to be interrupted; without this, a cache killed
    # part-way holds perfectly good .npz files that nothing will load, because
    # list_cached refuses a directory with no meta.json.
    meta_path.write_text(
        json.dumps({"settings": settings, "complete": False}, indent=2),
        encoding="utf-8",
    )

    written, skipped, failures = 0, 0, []
    started = time.time()
    for index, ref in enumerate(refs, 1):
        destination = cache_path(cache_dir, ref)
        if destination.is_file() and not args.overwrite:
            skipped += 1
            continue
        try:
            payload = build_one(ref, args)
        except Exception as error:
            failures.append((ref.key, f"{type(error).__name__}: {error}"))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        # savez appends ".npz" to a path that lacks it, so the temporary is
        # written through an open handle instead -- otherwise the rename below
        # chases a filename numpy silently changed.
        temporary = destination.with_name(destination.name + ".partial")
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(destination)  # never leave a half-written .npz behind
        written += 1

        elapsed = time.time() - started
        rate = elapsed / max(written, 1)
        print(
            f"[{index}/{len(refs)}] {ref.key}  "
            f"{payload['frames'].shape}  {destination.stat().st_size / 1e6:.0f} MB  "
            f"{rate:.1f}s/clip  eta {(len(refs) - index) * rate / 3600:.1f}h",
            file=sys.stderr,
        )

    meta_path.write_text(
        json.dumps(
            {
                "settings": settings,
                "complete": True,
                "n_written": written,
                "n_skipped": skipped,
                "n_failed": len(failures),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nwrote {written}, skipped {skipped}, failed {len(failures)}")
    for key, reason in failures[:10]:
        print(f"  FAILED {key}: {reason}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
