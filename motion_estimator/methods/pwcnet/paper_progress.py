"""One-page progress figure for a training run: curves plus predicted vs true flow.

    python -m motion_estimator.methods.pwcnet.paper_progress <run dir>
    python -m motion_estimator.methods.pwcnet.paper_progress <run dir> --log <train.log> --out progress.png

Top row: training loss and EPE (every 100 steps, from the log) and validation
EPE (from ``history.json``) against the **zero-flow baseline** -- the EPE of
predicting no motion, i.e. the mean true displacement. A model below that line
has learnt motion; one sitting on it has learnt nothing.

Bottom rows: validation pairs with frame 1, the true flow, the predicted flow
(colour = direction, brightness = magnitude, same scale in both) and the
end-point error, all inside the myocardium mask.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb

from .net import PWCNet
from .net_pretrained import PretrainedPWCNet
from .paper_data import PaperPairDataset, load_split, val_items
from .train_paper import SPLITS

_LINE = re.compile(r"\[(\d+)/\d+\] loss ([\d.]+) epe ([\d.]+)px")


def flow_rgb(flow, mask, max_mag):
    """``(2, H, W)`` flow to RGB: hue = direction, value = magnitude / max_mag."""
    u, v = flow
    hue = (np.arctan2(v, u) + np.pi) / (2 * np.pi)
    value = np.clip(np.hypot(u, v) / max_mag, 0, 1)
    rgb = hsv_to_rgb(np.stack([hue, np.ones_like(hue), value], axis=-1))
    rgb[mask < 0.5] = 1.0
    return rgb


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--log", type=Path, default=None, help="default: <run>/train.log")
    parser.add_argument("--out", type=Path, default=None, help="default: <run>/progress.png")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    args = parser.parse_args(argv)

    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    history = json.loads((args.run / "history.json").read_text(encoding="utf-8"))["history"]
    log = args.log or args.run / "train.log"
    train = np.array([[float(g) for g in m.groups()] for m in _LINE.finditer(log.read_text(encoding="utf-8"))])

    test, fold = config["test"], config["fold"]
    split, pools = load_split(args.splits, test)
    items = val_items(split, pools, fold, config["val_pairs"], seed=config["seed"])
    dataset = PaperPairDataset(items, config["size"], train=False)
    picks = np.linspace(0, len(items) - 1, args.pairs).astype(int)
    samples = [dataset[int(i)] for i in picks]

    # Zero-flow baseline over the whole validation sample, not just the plotted pairs
    zero = np.mean([float((torch.linalg.norm(s["flow"], dim=0) * s["mask"][0]).sum()
                          / s["mask"].sum().clamp_min(1)) for s in (dataset[i] for i in range(0, len(items), 4))])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Runs before the pretrained switch have no "net" key and were trained from scratch
    model = (PretrainedPWCNet(None) if config.get("net") == "pretrained" else PWCNet()).to(device).eval()
    state = torch.load(args.run / args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    with torch.no_grad():
        image0 = torch.stack([s["image0"] for s in samples]).to(device)
        image1 = torch.stack([s["image1"] for s in samples]).to(device)
        predicted = model(image0, image1)["flow"].float().cpu().numpy()

    fig = plt.figure(figsize=(16, 4 + 3.6 * len(samples)))
    grid = fig.add_gridspec(1 + len(samples), 4, height_ratios=[1.1] + [1] * len(samples))
    ax = fig.add_subplot(grid[0, :2])
    if len(train):
        ax.plot(train[:, 0], train[:, 1], color="#4c72b0", label="train loss")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.grid(alpha=0.25); ax.legend(frameon=False)
    ax = fig.add_subplot(grid[0, 2:])
    if len(train):
        ax.plot(train[:, 0], train[:, 2], color="#9aa5b1", label="train EPE (augmented crops)")
    if history:
        ax.plot([h["step"] for h in history], [h["val_epe"] for h in history], "o-",
                color="#c44e52", label="validation EPE")
    ax.axhline(zero, color="black", linestyle="--", linewidth=1, label=f"zero-flow baseline {zero:.2f} px")
    ax.set_xlabel("step"); ax.set_ylabel("EPE (px)"); ax.grid(alpha=0.25); ax.legend(frameon=False)
    best = min((h["val_epe"] for h in history), default=float("nan"))
    ax.set_title(f"{test} {'' if fold is None else f'fold {fold}'}: best val EPE {best:.3f} px "
                 f"({state.get('step', '?')} steps, {args.checkpoint})", fontsize=10)

    for row, (sample, pred) in enumerate(zip(samples, predicted), 1):
        true, mask = sample["flow"].numpy(), sample["mask"][0].numpy()
        top = max(np.percentile(np.hypot(*true)[mask > 0.5], 99), 1e-3) if mask.any() else 1.0
        error = np.hypot(*(pred - true))
        epe = float((error * mask).sum() / max(mask.sum(), 1))
        panels = [(sample["image0"][0].numpy(), "frame t", "gray"),
                  (flow_rgb(true, mask, top), f"true flow (max {top:.2f} px)", None),
                  (flow_rgb(pred, mask, top), "predicted flow", None),
                  (np.where(mask > 0.5, error, np.nan), f"EPE, mean {epe:.3f} px", "magma")]
        for col, (image, title, cmap) in enumerate(panels):
            ax = fig.add_subplot(grid[row, col])
            shown = ax.imshow(image, cmap=cmap, vmin=0 if cmap == "magma" else None,
                              vmax=max(top, 1e-3) if cmap == "magma" else None)
            if cmap == "magma":
                fig.colorbar(shown, ax=ax, fraction=0.046)
            ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = args.out or args.run / "progress.png"
    fig.savefig(out, dpi=80)
    print(f"wrote {out}  (best val EPE {best:.3f} px, zero-flow {zero:.3f} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
