"""Frame pairs with dense targets for the paper's tests, read from the split files.

Every item is a dict of tensors:

* ``image0`` / ``image1`` -- ``(1, S, S)`` in ``[0, 1]``;
* ``flow`` -- ``(2, S, S)`` target displacement ``(dx, dy)`` in pixels;
* ``mask`` -- ``(1, S, S)``, 1 where the target is defined;
* ``source`` -- 0 synthetic, 1 in vivo, so validation can report each.

**Synthetic targets come from the scatterers.** Their displacement is splatted
bilinearly onto the pixel grid, both the weighted displacement and the weight
are blurred with a Gaussian (sigma 1 px), and the ratio is the flow. The mask
is where the blurred weight exceeds 0.1, which is the myocardium: on a test
clip it covered 98% of the pixels within 2 px of a scatterer and none farther
out, and agreed with Delaunay-linear interpolation to 0.004 px (median).

**In vivo targets are the paper's MITT dense fields** (``paper_cache --iv``),
defined wherever MITT tracked, i.e. non-zero; the mask is that support.

**No scipy here.** Its OpenMP runtime and torch's abort the process when both
load, so the splat is done with torch ops.

Crops are ``S x S`` (a multiple of 64 for PWC-Net). Synthetic crops are
centred on the myocardium, in vivo crops on the 500 x 500 frame, both jittered
in training; validation crops are centred, never jittered or flipped, so the
validation number is comparable across steps.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .paper_cache import PAPER_CACHE, iv_clip_dir, ms_clip_dir

MS, IV = 0, 1
SPLAT_SIGMA = 1.0
SPLAT_MIN_WEIGHT = 0.1
JITTER = 48  # px each way around the myocardium centre


# -- items ------------------------------------------------------------------

def load_split(splits_dir, test: str) -> tuple[dict, dict]:
    splits_dir = Path(splits_dir)
    split = json.loads((splits_dir / f"{test}.json").read_text(encoding="utf-8"))
    pools = json.loads((splits_dir / "pools.json").read_text(encoding="utf-8"))
    return split, pools


def _fold(split: dict, fold: int | None) -> dict:
    folds = split["folds"]
    if fold is None:
        if len(folds) != 1:
            raise ValueError(f"{split['test']['name']} is cross-validated: pass a fold")
        return folds[0]
    return next(f for f in folds if f["fold"] == fold)


def train_items(split: dict, pools: dict, fold: int | None) -> list[tuple[int, str, int]]:
    """``(source, clip, t)`` for every training pair; ``t`` is the pair's first frame.

    Synthetic ``t`` is 0-based; in vivo ``t`` is 1-based (``frame<t>.mat``).
    """
    items = [(MS, clip, t) for clip, ts in split["ms"]["train"].items() for t in ts]
    iv_pairs = {c["clip"]: c["pairs"] for c in pools["in_vivo"]["clips"]}
    for clip in _fold(split, fold)["iv_train"]:
        items += [(IV, clip, t) for t in range(1, iv_pairs[clip] + 1)]
    return items


def val_items(split: dict, pools: dict, fold: int | None, per_source: int,
              seed: int = 0) -> list[tuple[int, str, int]]:
    """A fixed sample of validation pairs, at most *per_source* from each side."""
    rng = np.random.default_rng(seed)
    held = set(split["ms"]["val_patients"])
    ms = [(MS, c["clip"], t) for c in pools["synthetic"]["clips"]
          if c["patient"] in held for t in range(c["pairs"])]
    iv_pairs = {c["clip"]: c["pairs"] for c in pools["in_vivo"]["clips"]}
    iv = [(IV, clip, t) for clip in _fold(split, fold)["iv_val"]
          for t in range(1, iv_pairs[clip] + 1)]
    chosen = []
    for group in (ms, iv):
        if len(group) > per_source:
            group = [group[i] for i in sorted(rng.choice(len(group), per_source, replace=False))]
        chosen += group
    return chosen


# -- dense targets ----------------------------------------------------------

def _gaussian(sigma: float) -> torch.Tensor:
    radius = int(np.ceil(3 * sigma))
    kernel = torch.exp(-torch.arange(-radius, radius + 1.0) ** 2 / (2 * sigma ** 2))
    return kernel / kernel.sum()


_KERNEL = _gaussian(SPLAT_SIGMA)


def splat_flow(points0: np.ndarray, points1: np.ndarray, size: int):
    """Dense ``(2, S, S)`` flow and ``(1, S, S)`` mask from scatterers in crop pixels.

    *points0* / *points1* are ``(2, N)`` 0-based ``(x, y)``.
    """
    p0 = torch.from_numpy(np.ascontiguousarray(points0, dtype=np.float32))
    disp = torch.from_numpy(np.ascontiguousarray(points1 - points0, dtype=np.float32))
    values = torch.cat([disp, torch.ones(1, p0.shape[1])])
    x, y = p0[0], p0[1]
    x0, y0 = x.floor(), y.floor()
    acc = torch.zeros(3, size * size)
    for dx in (0.0, 1.0):
        for dy in (0.0, 1.0):
            xi, yi = x0 + dx, y0 + dy
            weight = (1 - (x - xi).abs()) * (1 - (y - yi).abs())
            inside = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
            index = (yi[inside] * size + xi[inside]).long()
            acc.index_add_(1, index, values[:, inside] * weight[inside])
    acc = acc.view(1, 3, size, size)
    radius = (_KERNEL.numel() - 1) // 2
    acc = F.conv2d(acc, _KERNEL.view(1, 1, 1, -1).repeat(3, 1, 1, 1), padding=(0, radius), groups=3)
    acc = F.conv2d(acc, _KERNEL.view(1, 1, -1, 1).repeat(3, 1, 1, 1), padding=(radius, 0), groups=3)
    weight = acc[0, 2:]
    mask = weight > SPLAT_MIN_WEIGHT
    flow = torch.where(mask, acc[0, :2] / weight.clamp_min(1e-6), torch.zeros(()))
    return flow, mask.float()


# -- dataset ----------------------------------------------------------------

def _crop(image: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    """``size x size`` window, zero-padded where it leaves the image."""
    out = np.zeros((size, size), dtype=np.float32)
    h, w = image.shape
    t0, l0 = max(top, 0), max(left, 0)
    t1, l1 = min(top + size, h), min(left + size, w)
    if t1 > t0 and l1 > l0:
        out[t0 - top:t1 - top, l0 - left:l1 - left] = image[t0:t1, l0:l1]
    return out


class PaperPairDataset(Dataset):
    def __init__(self, items, size: int = 384, train: bool = True,
                 cache=PAPER_CACHE, max_open: int = 64, max_spacing: int = 1):
        if size % 64:
            raise ValueError(f"size must be a multiple of 64, got {size}")
        self.items = list(items)
        self.size = size
        self.train = train
        self.cache = Path(cache)
        self.max_open = max_open
        # >1 pairs frame t with t + s, s uniform in [1, max_spacing] (synthetic only; the
        # in vivo targets exist for consecutive frames only). 1 = the paper's pairs.
        self.max_spacing = max_spacing
        self._open: OrderedDict = OrderedDict()

    def __len__(self) -> int:
        return len(self.items)

    def _arrays(self, source: int, clip: str) -> dict:
        key = (source, clip)
        if key in self._open:
            self._open.move_to_end(key)
            return self._open[key]
        if source == MS:
            folder = ms_clip_dir(self.cache, clip)
            arrays = {"frames": np.load(folder / "frames.npy", mmap_mode="r"),
                      "scat": np.load(folder / "scat.npy", mmap_mode="r")}
        else:
            folder = iv_clip_dir(self.cache, clip)
            arrays = {"frames": np.load(folder / "frames.npy", mmap_mode="r"),
                      "flow": np.load(folder / "flow.npy", mmap_mode="r")}
        self._open[key] = arrays
        if len(self._open) > self.max_open:
            self._open.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> dict:
        source, clip, t = self.items[index]
        rng = np.random.default_rng(int(torch.randint(2 ** 31, (1,))) if self.train else index)
        arrays = self._arrays(source, clip)
        size = self.size

        frames = arrays["frames"]
        h, w = frames.shape[1:]
        step = 1
        if source == MS:
            first = t                     # 0-based
            if self.max_spacing > 1:
                step = int(rng.integers(1, min(self.max_spacing, len(frames) - 1 - t) + 1))
            p0 = np.asarray(arrays["scat"][t], dtype=np.float32)
            p1 = np.asarray(arrays["scat"][t + step], dtype=np.float32)
            centre = (p0.min(axis=1) + p0.max(axis=1)) / 2
        else:
            first = t - 1                 # frame<t>.mat is index t-1; its pair's flow is flow[t-1]
            centre = np.array([w, h], dtype=np.float64) / 2
        if self.train:
            centre = centre + rng.uniform(-JITTER, JITTER, size=2)
        left, top = (np.round(centre - size / 2)).astype(int)
        left = int(np.clip(left, min(0, w - size), max(0, w - size)))
        top = int(np.clip(top, min(0, h - size), max(0, h - size)))
        image0 = _crop(frames[first], top, left, size)
        image1 = _crop(frames[first + step], top, left, size)
        if source == MS:
            shift = np.array([[left], [top]], dtype=np.float32)
            flow, mask = splat_flow(p0 - shift, p1 - shift, size)
        else:
            field = np.asarray(arrays["flow"][first], dtype=np.float32)
            flow = torch.from_numpy(np.stack([_crop(field[0], top, left, size),
                                              _crop(field[1], top, left, size)]))
            mask = (flow != 0).any(dim=0, keepdim=True).float()

        image0 = torch.from_numpy(image0)[None] / 255.0
        image1 = torch.from_numpy(image1)[None] / 255.0
        if self.train and rng.random() < 0.5:
            image0, image1, mask = image0.flip(-1), image1.flip(-1), mask.flip(-1)
            flow = flow.flip(-1) * torch.tensor([-1.0, 1.0]).view(2, 1, 1)
        return {"image0": image0, "image1": image1, "flow": flow, "mask": mask,
                "source": torch.tensor(source)}
