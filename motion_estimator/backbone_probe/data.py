"""Frame pairs with reference motion for the probe: simulated (exact) and in vivo (MITT).

Each pair is a dict: ``image0`` / ``image1`` ``(1, S, S)`` float in ``[0, 1]``,
``points0`` / ``points1`` ``(N, 2)`` 0-based ``(x, y)`` in crop pixels, ``key``,
``spacing``. Simulated crops are centred on the myocardium; in vivo frames are
500 x 500 and zero-padded to the crop size, so coordinates are unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
PAPER_CACHE = ARTIFACTS / "paper-cache"
INVIVO_CACHE = ARTIFACTS / "invivo-cache"
SPLITS = ARTIFACTS / "splits" / "paper_replication"


def _crop(image, top, left, size):
    out = np.zeros((size, size), np.float32)
    h, w = image.shape
    t1, l1 = min(top + size, h), min(left + size, w)
    out[: t1 - top, : l1 - left] = image[top:t1, left:l1]
    return out / 255.0


def _starts(n_frames, spacing, n_starts, rng):
    last = n_frames - 1 - spacing
    if last < 0:
        return []
    return sorted(set(int(v) for v in np.linspace(0, last, n_starts).round()))


def simulated_clips(n_clips, seed=0):
    """Validation-patient clips of the paper split, never trained on."""
    split = json.loads((SPLITS / "MS47_IV0.json").read_text(encoding="utf-8"))
    pools = json.loads((SPLITS / "pools.json").read_text(encoding="utf-8"))
    held = set(split["ms"]["val_patients"])
    clips = [c["clip"].removesuffix(".npz") for c in pools["synthetic"]["clips"] if c["patient"] in held]
    rng = np.random.default_rng(seed)
    return [clips[i] for i in sorted(rng.choice(len(clips), min(n_clips, len(clips)), replace=False))]


def simulated_pairs(clips, spacings, n_starts, n_queries, size, seed=0):
    rng = np.random.default_rng(seed)
    for clip in clips:
        folder = PAPER_CACHE / "ms" / clip
        frames = np.load(folder / "frames.npy", mmap_mode="r")
        scat = np.load(folder / "scat.npy", mmap_mode="r")
        h, w = frames.shape[1:]
        for spacing in spacings:
            for t in _starts(len(frames), spacing, n_starts, rng):
                p0 = np.asarray(scat[t], np.float32)
                p1 = np.asarray(scat[t + spacing], np.float32)
                centre = (p0.min(1) + p0.max(1)) / 2
                left = int(np.clip(round(centre[0] - size / 2), 0, max(w - size, 0)))
                top = int(np.clip(round(centre[1] - size / 2), 0, max(h - size, 0)))
                q0 = p0.T - [left, top]
                q1 = p1.T - [left, top]
                ok = ((q0 >= 0) & (q0 < size) & (q1 >= 0) & (q1 < size)).all(1)
                idx = np.flatnonzero(ok)
                if idx.size < 16:
                    continue
                idx = idx[np.linspace(0, idx.size - 1, min(n_queries, idx.size)).astype(int)]
                yield {"image0": _crop(frames[t], top, left, size)[None],
                       "image1": _crop(frames[t + spacing], top, left, size)[None],
                       "points0": q0[idx], "points1": q1[idx],
                       "key": clip, "spacing": spacing}


def invivo_clips(n_clips, dataset="Retest_dataset", seed=0):
    paths = sorted((INVIVO_CACHE / dataset).glob("*.npz"))
    rng = np.random.default_rng(seed)
    return [paths[i] for i in sorted(rng.choice(len(paths), min(n_clips, len(paths)), replace=False))]


def invivo_pairs(paths, spacings, n_starts, size, seed=0):
    rng = np.random.default_rng(seed)
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            frames, rx, ry, key = data["frames"], data["reference_x"], data["reference_y"], str(data["key"])
        n = min(len(frames), len(rx))
        for spacing in spacings:
            for t in _starts(n, spacing, n_starts, rng):
                q0 = np.stack([rx[t], ry[t]], 1) - 1.0           # MITT is 1-based
                q1 = np.stack([rx[t + spacing], ry[t + spacing]], 1) - 1.0
                ok = np.isfinite(q0).all(1) & np.isfinite(q1).all(1) & (q1 >= 0).all(1) & (q1 < size).all(1)
                if ok.sum() < 8:
                    continue
                yield {"image0": _crop(frames[t], 0, 0, size)[None],
                       "image1": _crop(frames[t + spacing], 0, 0, size)[None],
                       "points0": q0[ok].astype(np.float32), "points1": q1[ok].astype(np.float32),
                       "key": key, "spacing": spacing}
