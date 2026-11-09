"""Folds, the in-vivo cache, and in-vivo frame pairs for fine-tuning.

    python -m motion_estimator.methods.pwcnet.data --build-invivo-cache
    python -m motion_estimator.methods.pwcnet.data --make-folds --run-dir <dir>

**Folds are by patient on both cohorts.** Simulated: ``Pat_<id>`` (both views
and every simulation type come from one geometry). In vivo:
``<dataset>/<patient>`` -- IV numbers are reused across Retest and Test for
different people. Only ``Retest_dataset`` is folded; ``Test_dataset`` stays
locked for the final number.

Inside each fold's training patients a few more are held out as a
**validation** set, used only to pick ``best.pt``. The test patients are never
seen during training or model selection.

The in-vivo cache exists because ``load_in_vivo`` reads and prepares every
``data.mat`` from the F: drive, which is slow; fine-tuning and evaluation
reread the same clips many times. Frames are stored as uint8 after scaling by
each clip's own min/max.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ...loaders.test_loader import EvalClip, sector_mask
from ...loaders.train_loader import list_cached

__all__ = [
    "SIM_CACHE", "INVIVO_CACHE", "make_folds", "load_folds", "build_invivo_cache",
    "list_invivo_cached", "load_invivo_clip", "load_simulated_clip",
    "InVivoPairDataset",
]

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
SIM_CACHE = ARTIFACTS / "simulated-cache"
INVIVO_CACHE = ARTIFACTS / "invivo-cache"
DEV_DATASET = "Retest_dataset"


# -- folds ------------------------------------------------------------------

def _split_groups(groups, k, seed, val_fraction):
    """k test folds over *groups*, plus a validation subset of each train part."""
    groups = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    folds = []
    for fold in range(k):
        test = sorted(groups[fold::k])
        rest = [g for g in groups if g not in set(test)]
        n_val = max(1, int(round(len(rest) * val_fraction)))
        val = sorted(rest[:n_val])
        train = sorted(rest[n_val:])
        folds.append({"fold": fold, "train": train, "val": val, "test": test})
    return folds


def make_folds(run_dir, k=5, seed=0, val_fraction=0.1,
               sim_cache=SIM_CACHE, invivo_cache=INVIVO_CACHE):
    """Write ``folds.json`` into *run_dir* and return it.

    Simulated fold *k* and in-vivo fold *k* are paired: the in-vivo fold is
    fine-tuned from the model pretrained on simulated fold *k*.
    """
    sim_groups = [clip.group_id for clip in list_cached(sim_cache)]
    invivo_groups = [item["group_id"] for item in list_invivo_cached(invivo_cache)]
    if not invivo_groups:
        raise FileNotFoundError(
            f"no in-vivo cache at {invivo_cache}; run --build-invivo-cache first"
        )
    folds = {
        "k": k,
        "seed": seed,
        "val_fraction": val_fraction,
        "simulated": _split_groups(sim_groups, k, seed, val_fraction),
        "in_vivo": _split_groups(invivo_groups, k, seed, val_fraction),
    }
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    return folds


def load_folds(run_dir):
    return json.loads((Path(run_dir) / "folds.json").read_text(encoding="utf-8"))


# -- in-vivo cache ----------------------------------------------------------

def _cache_path(cache_dir, key: str) -> Path:
    dataset, patient, view = key.split("/")
    return Path(cache_dir) / dataset / f"{patient}_{view}.npz"


def build_invivo_cache(dataset=DEV_DATASET, cache_dir=INVIVO_CACHE,
                       mitt_variant="uni", overwrite=False):
    """Every clip with a MITT contour: frames, contour, sector, GE strain."""
    from ...loaders.annotations.gt_strain import read_gt_strain
    from ...loaders.dataset_index import index_dataset
    from ...loaders.test_loader import load_in_vivo

    gt_paths = {str(ref.key): ref.gt_strain_path
                for ref in index_dataset(None, (dataset,))}
    written = 0
    for view in ("2ch", "4ch"):
        for clip in load_in_vivo(dataset, view, mitt_variant, limit=None):
            path = _cache_path(cache_dir, clip.key)
            if path.is_file() and not overwrite:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            frames = clip.frames
            low, high = float(np.nanmin(frames)), float(np.nanmax(frames))
            scaled = np.nan_to_num((frames - low) / max(high - low, 1e-9), nan=0.0)
            gls_ge = np.nan
            gt = gt_paths.get(clip.key)
            if gt is not None:
                try:
                    gls_ge = float(read_gt_strain(gt).gls_es.get(view, np.nan))
                except Exception as error:  # a broken GT file must not stop the cache
                    print(f"  {clip.key}: GE strain unreadable ({error})")
            np.savez_compressed(
                path.with_suffix(".partial.npz"),
                frames=np.round(scaled * 255).astype(np.uint8),
                reference_x=clip.reference_x.astype(np.float32),
                reference_y=clip.reference_y.astype(np.float32),
                spacing_xy=clip.spacing_xy,
                start_frame=clip.start_frame,
                es_frame=clip.es_frame,
                sector=clip.sector,
                key=clip.key,
                group_id=clip.group_id,
                gls_ge=gls_ge,
            )
            path.with_suffix(".partial.npz").replace(path)
            written += 1
            print(f"  cached {clip.key}")
    print(f"wrote {written} clips to {cache_dir}")


def list_invivo_cached(cache_dir=INVIVO_CACHE, dataset=DEV_DATASET):
    """``[{"path", "key", "group_id"}]`` for every cached in-vivo clip."""
    folder = Path(cache_dir) / dataset
    items = []
    for path in sorted(folder.glob("*.npz")):
        if path.name.endswith(".partial.npz"):
            continue
        with np.load(path, allow_pickle=False) as data:
            items.append({"path": path, "key": str(data["key"]),
                          "group_id": str(data["group_id"])})
    return items


def load_invivo_clip(path) -> EvalClip:
    """A cached in-vivo clip as an :class:`EvalClip`, with ``gls_ge`` attached."""
    with np.load(path, allow_pickle=False) as data:
        clip = EvalClip(
            key=str(data["key"]),
            frames=data["frames"].astype(np.float32),
            reference_x=data["reference_x"].astype(float),
            reference_y=data["reference_y"].astype(float),
            spacing_xy=np.asarray(data["spacing_xy"], dtype=float),
            start_frame=int(data["start_frame"]),
            es_frame=int(data["es_frame"]),
            truth=False,
            sector=np.asarray(data["sector"], dtype=bool),
            group_id=str(data["group_id"]),
        )
        clip.gls_ge = float(data["gls_ge"])
    return clip


def load_simulated_clip(path, n_points=200) -> EvalClip:
    """A cached simulated clip, scored against evenly subsampled scatterers."""
    with np.load(path, allow_pickle=False) as data:
        frames = data["frames"].astype(np.float32)
        scat_x, scat_y = data["scat_x"], data["scat_y"]
        columns = np.linspace(0, scat_x.shape[1] - 1, n_points).astype(int)
        clip = EvalClip(
            key=str(data["key"]),
            frames=frames,
            reference_x=scat_x[:, columns].astype(float),
            reference_y=scat_y[:, columns].astype(float),
            spacing_xy=np.asarray(data["pixel_mm"], dtype=float).ravel()[:2][::-1].copy(),
            start_frame=0,
            es_frame=int(data["es_frame"]),
            truth=True,
            sector=sector_mask(frames),
            group_id=str(data["group_id"]),
        )
    return clip


# -- in-vivo pairs ----------------------------------------------------------

class InVivoPairDataset(Dataset):
    """Frame pairs from in-vivo clips, with MITT's contour as sparse targets.

    Same item layout as :class:`~motion_estimator.loaders.train_loader.SimPairDataset`
    plus ``sector`` ``(1, size, size)``, so one training loop serves both. The
    unsupervised mode ignores the points; the MITT mode ignores the sector.

    MITT points are 1-based in the clip; they come out 0-based in crop
    coordinates, as the simulated ones do.
    """

    def __init__(self, paths, size=320, min_spacing=1, max_spacing=3,
                 n_points=128, jitter=0.25, epoch_length=None, seed=0, augment=True):
        self.clips = [load_invivo_clip(p) for p in paths]
        self.size = size
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.n_points = n_points
        self.jitter = jitter
        self.augment = augment
        self.epoch_length = epoch_length or len(self.clips) * 16
        self.seed = seed

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, index):
        rng = np.random.default_rng((self.seed, index))
        clip = self.clips[int(rng.integers(len(self.clips)))]
        n_frames = min(clip.n_frames, clip.reference_x.shape[0])
        spacing = int(rng.integers(self.min_spacing, self.max_spacing + 1))
        spacing = max(1, min(spacing, n_frames - 1))
        first = int(rng.integers(0, n_frames - spacing))
        second = first + spacing
        if rng.random() < 0.5:
            first, second = second, first

        x0, y0 = clip.reference_x[first] - 1.0, clip.reference_y[first] - 1.0
        x1, y1 = clip.reference_x[second] - 1.0, clip.reference_y[second] - 1.0
        rows, cols = clip.frames.shape[1:]
        span = self.size * self.jitter
        left = np.nanmean(x0) - self.size / 2 + rng.uniform(-span, span)
        top = np.nanmean(y0) - self.size / 2 + rng.uniform(-span, span)
        left = int(np.clip(round(left), 0, max(cols - self.size, 0)))
        top = int(np.clip(round(top), 0, max(rows - self.size, 0)))

        image0 = self._crop(clip.frames[first], top, left) / 255.0
        image1 = self._crop(clip.frames[second], top, left) / 255.0
        sector = self._crop(clip.sector.astype(np.float32), top, left)

        points0 = np.stack([x0 - left, y0 - top], -1)
        points1 = np.stack([x1 - left, y1 - top], -1)
        valid = (np.isfinite(points0).all(-1) & np.isfinite(points1).all(-1)
                 & (points0 >= 0).all(-1) & (points0 < self.size).all(-1)
                 & (points1 >= 0).all(-1) & (points1 < self.size).all(-1))
        keep = np.flatnonzero(valid)
        if keep.size:
            chosen = rng.choice(keep, self.n_points, replace=keep.size < self.n_points)
            points0, points1 = points0[chosen], points1[chosen]
            valid = np.ones(self.n_points, bool)
        else:
            points0 = points1 = np.zeros((self.n_points, 2))
            valid = np.zeros(self.n_points, bool)

        if self.augment:
            gain, gamma = rng.uniform(0.8, 1.25, size=2)
            image0 = np.clip(image0 * gain, 0, 1) ** gamma
            image1 = np.clip(image1 * gain, 0, 1) ** gamma

        return {
            "image0": torch.from_numpy(image0.astype(np.float32))[None],
            "image1": torch.from_numpy(image1.astype(np.float32))[None],
            "sector": torch.from_numpy(sector)[None],
            "points0": torch.from_numpy(points0.astype(np.float32)),
            "points1": torch.from_numpy(points1.astype(np.float32)),
            "valid": torch.from_numpy(valid),
            "key": clip.key,
            "spacing": spacing,
        }

    def _crop(self, frame, top, left):
        patch = np.zeros((self.size, self.size), dtype=np.float32)
        piece = frame[top: top + self.size, left: left + self.size]
        patch[: piece.shape[0], : piece.shape[1]] = piece
        return patch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-invivo-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mitt-variant", default="uni")
    parser.add_argument("--make-folds", action="store_true")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.build_invivo_cache:
        build_invivo_cache(mitt_variant=args.mitt_variant, overwrite=args.overwrite)
    if args.make_folds:
        if not args.run_dir:
            parser.error("--make-folds needs --run-dir")
        folds = make_folds(args.run_dir, args.k, args.seed)
        for cohort in ("simulated", "in_vivo"):
            sizes = [(len(f["train"]), len(f["val"]), len(f["test"])) for f in folds[cohort]]
            print(f"{cohort}: train/val/test patients per fold {sizes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
