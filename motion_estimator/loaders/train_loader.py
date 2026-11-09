"""Frame pairs and tracked points from the simulated cache, for training.

Reads what ``scripts/build_simulated_cache.py`` wrote. Every item is one frame
pair plus the scatterers visible in both crops -- the exact correspondence the
supervision needs.

Choices worth stating, because each is a guess otherwise:

* **Crops are centred on the myocardium**, jittered, not taken uniformly over
  the frame. A random crop of a 749x908 sector is mostly far field, where there
  is no ground truth and nothing to learn.
* **Pair spacing has a floor, not just a range.** Motion runs at ~1 px per
  frame step and LoFTR's coarse grid is 8 px, so consecutive frames give the
  coarse level nothing to learn from. ``min_spacing`` keeps the bulk of the
  training pairs above one cell; see :meth:`SimPairDataset._spacing` for what
  happened when it did not.
* **Splits are by patient.** One CAMUS geometry feeds both views and every
  ``sim_type`` derived from it, so anything finer puts the same ventricle on
  both sides.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["SimPairDataset", "CachedClip", "list_cached", "split_by_patient"]


@dataclass
class CachedClip:
    path: Path
    key: str
    group_id: str
    n_frames: int


def list_cached(cache_dir: str | Path) -> list[CachedClip]:
    """Every clip in a cache directory, with its grouping key."""
    cache_dir = Path(cache_dir)
    meta = cache_dir / "meta.json"
    if not meta.is_file():
        raise FileNotFoundError(
            f"{cache_dir} has no meta.json -- run build_simulated_cache first"
        )
    clips = []
    for path in sorted(cache_dir.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            clips.append(
                CachedClip(
                    path=path,
                    key=str(data["key"]),
                    group_id=str(data["group_id"]),
                    n_frames=int(data["frames"].shape[0]),
                )
            )
    if not clips:
        raise FileNotFoundError(f"no cached clips under {cache_dir}")
    return clips


def split_by_patient(clips, fraction: float = 0.2, seed: int = 0):
    """Hold out whole patients, never individual recordings."""
    groups = sorted({clip.group_id for clip in clips})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_held = max(1, int(round(len(groups) * fraction)))
    held = set(groups[:n_held])
    train = [c for c in clips if c.group_id not in held]
    validate = [c for c in clips if c.group_id in held]
    return train, validate


class SimPairDataset(Dataset):
    """Frame pairs with exact point correspondence.

    Each item is a dict of tensors: ``image0`` / ``image1`` ``(1, size, size)``
    in ``[0, 1]``, ``points0`` / ``points1`` ``(n_points, 2)`` 0-based ``(x, y)``
    in crop coordinates, and ``valid`` ``(n_points,)``.
    """

    def __init__(
        self,
        clips,
        size: int = 320,
        min_spacing: int = 8,
        max_spacing: int = 20,
        n_points: int = 2048,
        jitter: float = 0.25,
        epoch_length: int | None = None,
        seed: int = 0,
        augment: bool = True,
        clip_run: int = 1,
    ) -> None:
        if size % 8:
            raise ValueError(f"size must be a multiple of 8 for LoFTR, got {size}")
        self.clips = list(clips)
        self.size = size
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.n_points = n_points
        self.jitter = jitter
        self.augment = augment
        self.epoch_length = epoch_length or len(self.clips) * 8
        self.seed = seed
        # clip_run > 1 draws that many consecutive items from the same clip, so
        # a batch costs one clip load instead of one per item. 1 keeps the
        # original sampling exactly.
        self.clip_run = max(1, int(clip_run))
        self._cache: dict[Path, dict] = {}

    def __len__(self) -> int:
        return self.epoch_length

    def _load(self, clip: CachedClip) -> dict:
        # One clip is ~14 MB; keeping a handful resident avoids re-reading the
        # same file for every pair sampled from it.
        if clip.path not in self._cache:
            if len(self._cache) > 4:
                self._cache.pop(next(iter(self._cache)))
            with np.load(clip.path, allow_pickle=False) as data:
                self._cache[clip.path] = {
                    "frames": data["frames"],
                    "scat_x": data["scat_x"],
                    "scat_y": data["scat_y"],
                }
        return self._cache[clip.path]

    def _spacing(self, rng, n_frames: int) -> int:
        """Frame gaps large enough for the coarse level to see the motion.

        This is uniform over ``[min_spacing, max_spacing]``, and the floor is
        the whole point. Measured over 30 cached clips, median motion runs at
        ~1 px per frame step:

        ==========  =============
        spacing     median motion
        ==========  =============
        1           1.02 px
        2           2.03 px
        4           3.99 px
        8           7.65 px
        12          10.71 px
        ==========  =============

        LoFTR's coarse grid is 1/8 resolution, so anything under 8 px lands in
        the same cell it started in and the correct coarse target is "no
        motion". An earlier version of this sampler weighted spacing by ``1/n``,
        which put 32% of pairs at spacing 1 and **77% below one coarse cell** --
        it trained the model to predict zero displacement. Val loss fell
        steadily (that target is easy), cycle closure improved (zero motion
        closes perfectly), and accuracy got worse on held-out simulated *and*
        in-vivo data than the pretrained weights.

        Sub-cell motion is not useless -- it is what the fine level regresses --
        but it cannot be the bulk of the coarse level's diet.
        """
        limit = min(self.max_spacing, n_frames - 1)
        low = min(self.min_spacing, limit)
        if limit < 1:
            return 1
        return int(rng.integers(max(low, 1), limit + 1))

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng((self.seed, index))
        clip_rng = rng if self.clip_run == 1 else np.random.default_rng(
            (self.seed, index // self.clip_run, 1)
        )
        clip = self.clips[int(clip_rng.integers(len(self.clips)))]
        payload = self._load(clip)
        frames = payload["frames"]
        n_frames = frames.shape[0]

        spacing = self._spacing(rng, n_frames)
        first = int(rng.integers(0, n_frames - spacing))
        second = first + spacing
        if rng.random() < 0.5:
            first, second = second, first  # both directions, so it is symmetric

        x0 = payload["scat_x"][first] - 1.0  # cache is 1-based; crops are 0-based
        y0 = payload["scat_y"][first] - 1.0
        x1 = payload["scat_x"][second] - 1.0
        y1 = payload["scat_y"][second] - 1.0

        top, left = self._crop_origin(rng, frames.shape[1:], x0, y0)
        image0 = self._crop(frames[first], top, left)
        image1 = self._crop(frames[second], top, left)

        points0 = np.stack([x0 - left, y0 - top], axis=-1)
        points1 = np.stack([x1 - left, y1 - top], axis=-1)
        valid = (
            (points0[:, 0] >= 0) & (points0[:, 0] < self.size)
            & (points0[:, 1] >= 0) & (points0[:, 1] < self.size)
            & (points1[:, 0] >= 0) & (points1[:, 0] < self.size)
            & (points1[:, 1] >= 0) & (points1[:, 1] < self.size)
        )
        points0, points1, valid = self._sample_points(rng, points0, points1, valid)

        image0, image1 = image0.astype(np.float32) / 255.0, image1.astype(np.float32) / 255.0
        if self.augment:
            image0, image1 = self._augment(rng, image0, image1)

        return {
            "image0": torch.from_numpy(image0)[None],
            "image1": torch.from_numpy(image1)[None],
            "points0": torch.from_numpy(points0.astype(np.float32)),
            "points1": torch.from_numpy(points1.astype(np.float32)),
            "valid": torch.from_numpy(valid),
            "key": clip.key,
            "spacing": spacing,
        }

    def _crop_origin(self, rng, shape, x, y):
        rows, cols = shape
        centre_x = float(np.mean(x))
        centre_y = float(np.mean(y))
        span = self.size * self.jitter
        left = centre_x - self.size / 2 + rng.uniform(-span, span)
        top = centre_y - self.size / 2 + rng.uniform(-span, span)
        left = int(np.clip(round(left), 0, max(cols - self.size, 0)))
        top = int(np.clip(round(top), 0, max(rows - self.size, 0)))
        return top, left

    def _crop(self, frame, top, left):
        patch = frame[top: top + self.size, left: left + self.size]
        if patch.shape != (self.size, self.size):
            padded = np.zeros((self.size, self.size), dtype=frame.dtype)
            padded[: patch.shape[0], : patch.shape[1]] = patch
            return padded
        return patch

    def _sample_points(self, rng, points0, points1, valid):
        """A fixed-length point set, so items collate into a batch."""
        keep = np.flatnonzero(valid)
        if keep.size == 0:
            zeros = np.zeros((self.n_points, 2), dtype=np.float32)
            return zeros, zeros.copy(), np.zeros(self.n_points, dtype=bool)
        chosen = rng.choice(keep, size=self.n_points, replace=keep.size < self.n_points)
        return points0[chosen], points1[chosen], np.ones(self.n_points, dtype=bool)

    def _augment(self, rng, image0, image1):
        """Gain and gamma only, applied identically to both frames.

        Applying different gain to the two images of a pair would teach
        invariance to something the data never does, and applying geometry
        would invalidate the point correspondence.
        """
        gain = float(rng.uniform(0.8, 1.25))
        gamma = float(rng.uniform(0.8, 1.25))
        image0 = np.clip(image0 * gain, 0, 1) ** gamma
        image1 = np.clip(image1 * gain, 0, 1) ** gamma
        return image0.astype(np.float32), image1.astype(np.float32)
