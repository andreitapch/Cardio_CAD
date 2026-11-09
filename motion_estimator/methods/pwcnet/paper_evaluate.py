"""The paper's in vivo EPE for a trained run, on its held-out patients, both views.

    python -m motion_estimator.methods.pwcnet.paper_evaluate <run dir>
    python -m motion_estimator.methods.pwcnet.paper_evaluate <run dir> --checkpoint last.pt --views 2ch

For every test clip of the run's fold (all 236 clips for the synthetic-only
tests): predict the flow of each consecutive pair on the full 500 x 500
frames, propagate MITT's frame-1 contour through it, and score the contour
against MITT with :mod:`.paper_metrics` -- corrected spacing, and the paper's
swapped spacing for comparison with Table 2. Writes ``<run>/eval_invivo.json``
(per sequence and summaries per view and overall) and copies it to the mirror.

Frames come from ``artifacts/invivo-cache`` (``data.mat``, one cardiac cycle,
scaled per clip to uint8): frames 2..n are pixel-identical to the training
frames, and frame 1 is the one the contour starts from.

**6 of 236 clips are skipped and listed**, because MITT tracked a different
number of frames than the cycle holds, so the frames cannot be paired without
guessing: Retest IV43 2ch/4ch, Retest IV53 4ch, Test IV02 2ch/4ch, Test IV43
2ch. The paper's evaluation also left out IV02, IV27 and IV43.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ...strain.propagate import propagate_points
from .net import PWCNet
from .net_pretrained import PretrainedPWCNet
from .paper_data import load_split
from .paper_metrics import contour_epe, summarise
from .train_paper import MIRROR_ROOT, OUT_ROOT, SPLITS, mirror

INVIVO_CACHE = Path(__file__).resolve().parents[2] / "artifacts" / "invivo-cache"


def load_model(run: Path, checkpoint: str, device) -> torch.nn.Module:
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    model = PretrainedPWCNet(None) if config.get("net") == "pretrained" else PWCNet()
    state = torch.load(run / checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def clip_file(key: str) -> Path:
    dataset, patient, view = key.split("/")
    return INVIVO_CACHE / dataset / f"{patient}_{view}.npz"


@torch.no_grad()
def predict_flows(model, frames: np.ndarray, device, amp: bool = True, batch: int = 4) -> np.ndarray:
    """``(n-1, H, W, 2)`` flow of each consecutive pair, full frame."""
    images = torch.from_numpy(frames.astype(np.float32) / 255.0)[:, None]
    height, width = images.shape[-2:]
    pad = (0, (-width) % 64, 0, (-height) % 64)
    flows = []
    for start in range(0, len(images) - 1, batch):
        stop = min(start + batch, len(images) - 1)
        image0 = F.pad(images[start:stop], pad).to(device)
        image1 = F.pad(images[start + 1:stop + 1], pad).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            flow = model(image0, image1)["flow"].float()
        flows.append(flow[:, :, :height, :width].permute(0, 2, 3, 1).cpu())
    return torch.cat(flows).numpy()


def evaluate_clips(model, keys, device, amp: bool = True, batch: int = 4) -> list[dict]:
    """One row per clip: corrected and paper-exact contour EPE (mm)."""
    rows = []
    for key in keys:
        with np.load(clip_file(key), allow_pickle=False) as data:
            frames = data["frames"]
            ref_x, ref_y = data["reference_x"].astype(float), data["reference_y"].astype(float)
            spacing = data["spacing_xy"].astype(float)
        if len(frames) != len(ref_x):
            rows.append({"key": key, "view": key.rsplit("/", 1)[1], "n_frames": int(len(frames)),
                         "epe_mm": float("nan"), "epe_mm_paper_exact": float("nan"), "lost_points": 0,
                         "skipped": f"cycle has {len(frames)} frames, MITT {len(ref_x)}"})
            continue
        flows = predict_flows(model, frames, device, amp, batch)
        track_x, track_y = propagate_points(flows, ref_x[0], ref_y[0])   # MATLAB refFrame = 1
        track_x, track_y = track_x.numpy(), track_y.numpy()
        fixed = contour_epe(track_x, track_y, ref_x, ref_y, spacing, key)
        exact = contour_epe(track_x, track_y, ref_x, ref_y, spacing, key, paper_exact=True)
        rows.append({"key": key, "view": key.rsplit("/", 1)[1], "n_frames": int(len(frames)),
                     "epe_mm": fixed.epe_mm, "epe_mm_paper_exact": exact.epe_mm,
                     "lost_points": fixed.lost_points,
                     "per_frame_mm": [None if np.isnan(v) else round(float(v), 4) for v in fixed.per_frame_mm]})
    return rows


def summaries(rows: list[dict]) -> dict:
    """Mean +- SD across sequences, per view and overall, both spacing conventions."""
    out = {}
    for name, subset in [("all", rows)] + [(v, [r for r in rows if r["view"] == v])
                                          for v in sorted({r["view"] for r in rows})]:
        for field, label in (("epe_mm", "corrected"), ("epe_mm_paper_exact", "paper_exact")):
            values = np.array([r[field] for r in subset], float)
            out[f"{name}_{label}"] = {
                "n": int(np.isfinite(values).sum()),
                "mean": float(np.nanmean(values)) if values.size else float("nan"),
                "sd": float(np.nanstd(values, ddof=1)) if values.size > 1 else float("nan"),
            }
        out[f"{name}_lost_points"] = int(sum(r["lost_points"] for r in subset))
        out[f"{name}_skipped"] = [r["key"] for r in subset if r.get("skipped")]
    return out


def test_keys(run: Path, splits: Path, views) -> list[str]:
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    split, _ = load_split(splits, config["test"])
    folds = split["folds"]
    fold = folds[0] if config["fold"] is None else next(f for f in folds if f["fold"] == config["fold"])
    return [k for k in fold["iv_test"] if k.rsplit("/", 1)[1] in views]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--views", nargs="+", default=["2ch", "4ch"])
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    keys = test_keys(args.run, args.splits, args.views)
    if not keys:
        print(f"{args.run}: no in vivo test clips (external test set) -- nothing to score")
        return 0
    device = torch.device(args.device)
    model = load_model(args.run, args.checkpoint, device)
    started = time.time()
    rows = evaluate_clips(model, keys, device, batch=args.batch)
    result = {"run": str(args.run), "checkpoint": args.checkpoint, "views": args.views,
              "metric": "contour EPE vs MITT, mm, mean over frames 2..n-1, mean +- SD across sequences",
              "summary": summaries(rows), "sequences": rows,
              "minutes": round((time.time() - started) / 60, 1)}
    name = "eval_invivo.json" if args.checkpoint == "best.pt" else f"eval_invivo_{Path(args.checkpoint).stem}.json"
    (args.run / name).write_text(json.dumps(result, indent=1), encoding="utf-8")
    for label, value in result["summary"].items():
        if isinstance(value, dict):
            print(f"  {label:24s} {value['mean']:.3f} +- {value['sd']:.3f} mm (n={value['n']})")
    try:
        mirror(args.run, MIRROR_ROOT / args.run.resolve().relative_to(OUT_ROOT))
    except ValueError:
        pass  # a run outside paper-runs has no mirror location
    print(f"wrote {args.run / name} ({result['minutes']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
