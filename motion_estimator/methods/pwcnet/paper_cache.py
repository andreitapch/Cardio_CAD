"""Uncompressed, memory-mappable copies of the training pools.

    python -m motion_estimator.methods.pwcnet.paper_cache --ms
    python -m motion_estimator.methods.pwcnet.paper_cache --iv

The simulated cache is one compressed ``.npz`` per clip, so reading one frame
pair decompresses the whole clip (~30 MB). The paper's tests sample pairs
uniformly from thousands of clips, so nearly every pair would pay that. Here
each clip becomes plain ``.npy`` files opened with ``mmap_mode="r"``, and a
pair costs two frames.

    paper-cache/ms/<Pat_id>/<clip stem>/frames.npy   uint8 (T, H, W), 0 outside the sector
                                       /scat.npy     float32 (T, 2, N), 0-based (x, y) pixels

    paper-cache/iv/<dataset>/<IV##>/<view>/frames.npy  uint8 (T, 500, 500)
                                         /flow.npy    float16 (T-1, 2, 500, 500), pair k -> k+1 at [k-1]

The clip key in the split files (``Pat_108/regional_simulations_2ch_sim1.npz``,
``Retest_dataset/IV02/2ch``) maps to the folder directly (synthetic: drop
``.npz``). Each clip is written through a ``.partial`` folder and renamed, so
an interrupted build resumes cleanly.

**In vivo source: the paper's own training tree** -- ``mat/frame<k>.mat``
(``usFrame``, three identical channels) and the MITT dense fields
``disp/frame<k>.flo`` (Middlebury, ``(dx, dy)`` in pixels, defined on the first
frame's grid). A clip of T frames has T + 1 forward files, so which one belongs
to pair k -> k+1 had to be measured: warping frame k+1 back with each candidate
and comparing with frame k, **``frame<k+1>.flo``** won 60 of 72 pairs over
three clips from both cohorts (offsets -1, 0, +2 won 12 between them).
``Bframe<k+1>.flo`` is its negative.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
SIM_CACHE = ARTIFACTS / "simulated-cache"
PAPER_CACHE = ARTIFACTS / "paper-cache"
IV_ROOT = Path("F:/PHD/Experiments/Journal_EXP_2022/Datasets/"
               "2D_GE_intervendor_in_vivo_pxDisp/inVivo/train")


def ms_clip_dir(cache, clip_key: str) -> Path:
    return Path(cache) / "ms" / clip_key.removesuffix(".npz")


def iv_clip_dir(cache, clip_key: str) -> Path:
    return Path(cache) / "iv" / clip_key


def read_flo(path) -> np.ndarray:
    """Middlebury ``.flo`` as ``(2, H, W)`` float32."""
    with open(path, "rb") as handle:
        if np.frombuffer(handle.read(4), np.float32)[0] != 202021.25:
            raise ValueError(f"{path} is not a .flo file")
        width, height = np.frombuffer(handle.read(8), np.int32)
        return np.frombuffer(handle.read(), np.float32).reshape(height, width, 2).transpose(2, 0, 1)


def _write_clip(target: Path, arrays: dict) -> None:
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    for name, array in arrays.items():
        np.save(partial / f"{name}.npy", array)
    partial.rename(target)


def build_iv(root=IV_ROOT, cache=PAPER_CACHE) -> None:
    from scipy.io import loadmat  # here only: scipy must never load beside torch

    root, out = Path(root), Path(cache) / "iv"
    clips = sorted(view for view in root.glob("*_dataset/IV*/*ch") if (view / "mat").is_dir())
    written = 0
    for i, folder in enumerate(clips, 1):
        key = folder.relative_to(root).as_posix()
        target = iv_clip_dir(cache, key)
        if target.is_dir():
            continue
        n_frames = sum(1 for _ in (folder / "mat").glob("frame*.mat"))
        frames = np.stack([loadmat(folder / "mat" / f"frame{k}.mat")["usFrame"][..., 0]
                           for k in range(1, n_frames + 1)]).astype(np.uint8)
        flow = np.stack([read_flo(folder / "disp" / f"frame{k + 1}.flo")
                         for k in range(1, n_frames)]).astype(np.float16)
        if flow.shape[2:] != frames.shape[1:]:
            raise ValueError(f"{key}: flow {flow.shape} does not match frames {frames.shape}")
        _write_clip(target, {"frames": frames, "flow": flow})
        written += 1
        if written % 10 == 0 or i == len(clips):
            print(f"  iv {i}/{len(clips)}", flush=True)
    (out / "meta.json").write_text(json.dumps({
        "source": str(root), "clips": len(clips),
        "layout": "frames uint8 (T,H,W); flow float16 (T-1,2,H,W), pair k->k+1 (1-based) at [k-1], "
                  "from disp/frame<k+1>.flo",
    }, indent=2), encoding="utf-8")
    print(f"iv cache: {written} written, {len(clips)} total -> {out}")


def build_ms(source=SIM_CACHE, cache=PAPER_CACHE) -> None:
    source, out = Path(source), Path(cache) / "ms"
    paths = sorted(source.glob("Pat_*/*.npz"))
    written = 0
    for i, path in enumerate(paths, 1):
        key = path.relative_to(source).as_posix()
        target = ms_clip_dir(cache, key)
        if target.is_dir():
            continue
        with np.load(path, allow_pickle=False) as data:
            scat = np.stack([data["scat_x"], data["scat_y"]], axis=1).astype(np.float32) - 1.0
            _write_clip(target, {"frames": data["frames"].astype(np.uint8), "scat": scat})
        written += 1
        if written % 25 == 0 or i == len(paths):
            print(f"  ms {i}/{len(paths)}", flush=True)
    (out / "meta.json").write_text(json.dumps({
        "source": str(source), "clips": len(paths),
        "layout": "frames uint8 (T,H,W); scat float32 (T,2,N) 0-based (x,y)",
    }, indent=2), encoding="utf-8")
    print(f"ms cache: {written} written, {len(paths)} total -> {out}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ms", action="store_true", help="the simulated pool")
    parser.add_argument("--iv", action="store_true", help="the paper's in vivo pool")
    parser.add_argument("--cache", type=Path, default=PAPER_CACHE)
    args = parser.parse_args(argv)
    if not (args.ms or args.iv):
        parser.error("nothing to build: pass --ms and/or --iv")
    if args.ms:
        build_ms(cache=args.cache)
    if args.iv:
        build_iv(cache=args.cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
