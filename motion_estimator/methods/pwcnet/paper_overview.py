"""One figure with every run so far: validation curves and the in vivo test scores.

    python -m motion_estimator.methods.pwcnet.paper_overview
    python -m motion_estimator.methods.pwcnet.paper_overview --out overview.png

1. Synthetic validation EPE (px) per run, against predicting no motion.
2. In vivo validation EPE against MITT's dense field (px), for runs with in
   vivo data.
3. In vivo test score, the paper's metric (contour EPE vs MITT, mm, mean +-
   SD across sequences, corrected spacing), for runs evaluated with
   ``paper_evaluate``, beside the paper's Table 2 values.

Colours follow the run, in a fixed order (validated palette, first three
slots pass all-pairs CVD checks); every line is labelled at its end, so
identity never depends on colour alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Not imported from train_paper: that pulls in torch, whose OpenMP runtime aborts beside matplotlib's
OUT_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "paper-runs"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PAPER = {"MS47_IV0": 4.71, "MS47_IV85": 1.10, "MS0_IV85": 1.39, "MS100_IV0": 4.70}


def _runs(root: Path):
    runs = []
    for history in sorted(root.glob("*/*/history.json"), key=lambda p: p.parent.stat().st_ctime):
        h = json.loads(history.read_text(encoding="utf-8"))["history"]
        if h:
            name = history.parent.parent.name + ("" if history.parent.name == "single" else f" {history.parent.name}")
            runs.append((name, history.parent, h))
    return runs


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _line(ax, x, y, colour, label):
    ax.plot(x, y, color=colour, linewidth=2, label=label)
    ax._end_labels = getattr(ax, "_end_labels", []) + [(x[-1], y[-1], label)]


def _place_end_labels(ax, gap):
    """Label each line at its end, nudging labels that end close together apart by ``gap`` (data units)."""
    ends = sorted(getattr(ax, "_end_labels", []), key=lambda e: e[1])
    placed = []
    for x, y, label in ends:
        near = [py for px, py in placed if abs(px - x) < 40]
        if near and y < max(near) + gap:
            y = max(near) + gap
        placed.append((x, y))
        ax.annotate(label, (x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=OUT_ROOT)
    parser.add_argument("--out", type=Path, default=None, help="default: <root>/overview.png")
    args = parser.parse_args(argv)

    runs = _runs(args.root)
    colour = {name: SERIES[i] if i < len(SERIES) else INK_2 for i, (name, _, _) in enumerate(runs)}
    n_bars = sum(len(list(folder.glob("eval_invivo*.json"))) for _, folder, _ in runs)
    fig, axes = plt.subplots(3, 1, figsize=(9, 10.5 + 0.35 * max(n_bars, 3)), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1.15, 1, 0.25 * max(n_bars, 3)]})

    ax = axes[0]
    for name, _, h in runs:
        _line(ax, [r["step"] / 1000 for r in h], [r.get("val_epe_ms", r["val_epe"]) for r in h], colour[name], name)
    ax.axhline(1.0, color=INK_2, linestyle="--", linewidth=1)
    ax.text(0, 1.02, "predicting no motion (~1.0-1.1 px)", fontsize=8, color=INK_2, va="bottom")
    _style(ax, "Synthetic validation error (lower is better)", "EPE (px)")
    ax.set_xlabel("training step (thousands)", color=INK_2)
    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, max(r["step"] for _, _, h in runs for r in h) / 1000 * 1.22)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _place_end_labels(ax, 0.035)

    ax = axes[1]
    with_iv = [(n, d, h) for n, d, h in runs if any("val_epe_iv" in r for r in h)]
    for name, _, h in with_iv:
        rows = [r for r in h if "val_epe_iv" in r]
        _line(ax, [r["step"] / 1000 for r in rows], [r["val_epe_iv"] for r in rows], colour[name], name)
    ax.axhline(1.35, color=INK_2, linestyle="--", linewidth=1)
    ax.text(0, 1.37, "predicting no motion (1.35 px)", fontsize=8, color=INK_2, va="bottom")
    _style(ax, "In vivo validation error vs MITT's dense field", "EPE (px)")
    ax.set_xlabel("training step (thousands)", color=INK_2)
    ax.set_ylim(0, 1.5)
    ax.set_xlim(axes[0].get_xlim())
    _place_end_labels(ax, 0.06)
    if not with_iv:
        ax.text(0.5, 0.5, "no run with in vivo data yet", transform=ax.transAxes, ha="center", color=INK_2)

    ax = axes[2]
    scored = []
    for name, folder, h in runs:
        # best.pt (chosen on validation) first, then the fixed-step snapshots
        files = [("best", folder / "eval_invivo.json")] + [
            (f"@{int(f.stem.rsplit('step', 1)[1]) // 1000}k", f)
            for f in sorted(folder.glob("eval_invivo_step*.json"), key=lambda f: int(f.stem.rsplit("step", 1)[1]))]
        for which, path in files:
            if path.is_file():
                s = json.loads(path.read_text(encoding="utf-8"))["summary"]["all_corrected"]
                label = f"{name}  best of {h[-1]['step'] // 1000}k" if which == "best" else f"{name}  {which}"
                scored.append((name, label, s["mean"], s["sd"], s["n"]))
    for i, (name, _, mean, sd, n) in enumerate(scored):
        ax.barh(i, mean, xerr=sd, height=0.55, color=colour[name], error_kw={"ecolor": INK_2, "elinewidth": 1})
        ax.text(mean + sd + 0.08, i, f"{mean:.2f} +- {sd:.2f} mm  (n={n})", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(scored)), [s[1] for s in scored], color=INK)
    for test, value in PAPER.items():
        if any(name.startswith(test) for name, *_ in scored):
            ax.axvline(value, color=INK_2, linestyle="--", linewidth=1)
            ax.text(value + 0.06, len(scored) - 0.55, f"paper {test}\n{value:.2f} mm", fontsize=8, color=INK_2, ha="left")
    _style(ax, "In vivo test score - paper metric (contour EPE vs MITT, 2ch + 4ch)", "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("EPE (mm), mean +- SD across sequences, lower is better", color=INK_2)
    ax.set_xlim(0, 5.6)
    ax.set_ylim(-0.6, len(scored) + 0.4)
    if not scored:
        ax.text(0.5, 0.5, "no run evaluated yet", transform=ax.transAxes, ha="center", color=INK_2)

    fig.tight_layout(h_pad=2.5)
    out = args.out or args.root / "overview.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
