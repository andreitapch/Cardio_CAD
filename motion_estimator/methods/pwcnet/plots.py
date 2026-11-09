"""Figures shared by ``evaluate.py`` (saved PNGs) and the results notebook.

One colour per variant, fixed wherever it appears; MITT is always the white
dashed contour (it is the reference, not a competitor). Every figure carries a
legend, because the aqua sits below 3:1 contrast on a white background.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["COLORS", "LABELS", "VARIANTS", "plot_clip", "plot_contours",
           "plot_fold_box", "plot_gls_agreement"]

VARIANTS = ("zero_shot", "unsup", "mitt")
COLORS = {"zero_shot": "#2a78d6", "unsup": "#eb6834", "mitt": "#1baf7a"}
LABELS = {"zero_shot": "zero-shot (synthetic only)",
          "unsup": "fine-tuned, unsupervised",
          "mitt": "fine-tuned on MITT"}
REFERENCE = {"color": "white", "linestyle": "--", "linewidth": 1.5}


def _variant(name):
    """``unsup:consecutive`` -> ``unsup``, so pairing suffixes keep the colour."""
    return name.split(":")[0]


def plot_contours(ax, frame, reference_x, reference_y, tracks, index, title=""):
    """One frame with MITT and every variant's contour drawn over it.

    Coordinates are 1-based, as everywhere in the package, hence the -1.
    """
    ax.imshow(frame, cmap="gray", interpolation="nearest")
    ax.plot(reference_x[index] - 1, reference_y[index] - 1, label="MITT", **REFERENCE)
    for name, (x, y) in tracks.items():
        ax.plot(x[index] - 1, y[index] - 1, color=COLORS.get(_variant(name), "C7"),
                linewidth=1.5, label=LABELS.get(_variant(name), name))
    finite = np.isfinite(reference_x[index])
    if finite.any():
        cx, cy = reference_x[index][finite] - 1, reference_y[index][finite] - 1
        pad = 40
        ax.set_xlim(cx.min() - pad, cx.max() + pad)
        ax.set_ylim(cy.max() + pad, cy.min() - pad)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_clip(frames, reference_x, reference_y, tracks, start_frame, es_frame,
              strains=None, mitt_strain=None, gls_ge=None, title="", path=None):
    """Four panels: contours at the start and at ES, error per frame, strain.

    *tracks* is ``{variant: (x, y)}`` with ``(n_frames, n_points)`` arrays.
    *strains* is ``{variant: curve}``; *es_frame* is 1-based.
    """
    es = int(np.clip(es_frame - 1, 0, frames.shape[0] - 1))
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8),
                             gridspec_kw={"width_ratios": [1, 1, 1.2, 1.2]})
    plot_contours(axes[0], frames[start_frame], reference_x, reference_y, tracks,
                  start_frame, f"frame {start_frame + 1} (start)")
    plot_contours(axes[1], frames[es], reference_x, reference_y, tracks, es,
                  f"frame {es + 1} (end-systole)")

    ax = axes[2]
    for name, (x, y) in tracks.items():
        error = np.nanmedian(np.hypot(x - reference_x, y - reference_y), axis=1)
        ax.plot(np.arange(1, len(error) + 1), error, linewidth=2,
                color=COLORS.get(_variant(name), "C7"),
                label=LABELS.get(_variant(name), name))
    ax.axvline(es + 1, color="0.6", linewidth=1, linestyle=":")
    ax.text(es + 1, ax.get_ylim()[1], " ES", va="top", fontsize=8, color="0.4")
    ax.set_xlabel("frame")
    ax.set_ylabel("median distance to MITT (px)")
    ax.set_title("distance to MITT per frame", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[3]
    if mitt_strain is not None:
        ax.plot(np.arange(1, len(mitt_strain) + 1), mitt_strain, color="0.2",
                linestyle="--", linewidth=2, label="MITT")
    for name, curve in (strains or {}).items():
        ax.plot(np.arange(1, len(curve) + 1), curve, linewidth=2,
                color=COLORS.get(_variant(name), "C7"),
                label=LABELS.get(_variant(name), name))
    if gls_ge is not None and np.isfinite(gls_ge):
        ax.plot([es + 1], [gls_ge], marker="D", markersize=8, color="0.2",
                linestyle="none", label=f"GE GLS {gls_ge:.1f}%")
    ax.axvline(es + 1, color="0.6", linewidth=1, linestyle=":")
    ax.set_xlabel("frame")
    ax.set_ylabel("longitudinal strain (%)")
    ax.set_title("strain curve", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

    handles, labels = axes[0].get_legend_handles_labels()
    # Dark legend patch: the white MITT line would vanish on a white one.
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9,
               frameon=True, facecolor="0.3", labelcolor="white")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    if path is not None:
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig


def plot_fold_box(rows, metric="error_px_es", ylabel="ES distance to MITT (px)",
                  path=None):
    """Per-clip *metric*, one box per variant, one group per fold."""
    folds = sorted({r["fold"] for r in rows})
    variants = [v for v in VARIANTS if any(_variant(r["group"]) == v for r in rows)]
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(folds)), 4.5))
    width = 0.8 / max(len(variants), 1)
    for i, variant in enumerate(variants):
        data = [[r[metric] for r in rows
                 if r["fold"] == f and _variant(r["group"]) == variant
                 and np.isfinite(r[metric])] for f in folds]
        positions = np.arange(len(folds)) + (i - (len(variants) - 1) / 2) * width
        box = ax.boxplot(data, positions=positions, widths=width * 0.85,
                         patch_artist=True, showfliers=True)
        for patch in box["boxes"]:
            patch.set_facecolor(COLORS[variant])
            patch.set_alpha(0.85)
        for median in box["medians"]:
            median.set_color("black")
        ax.plot([], [], color=COLORS[variant], linewidth=8, label=LABELS[variant])
    ax.set_xticks(np.arange(len(folds)), [f"fold {f}" for f in folds])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig


def plot_gls_agreement(rows, reference="gls_mitt", path=None):
    """Predicted GLS against *reference* (``gls_mitt`` or ``gls_ge``)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    values = []
    for variant in VARIANTS:
        subset = [r for r in rows if _variant(r["group"]) == variant
                  and np.isfinite(r.get("gls_pred", np.nan))
                  and np.isfinite(r.get(reference, np.nan))]
        if not subset:
            continue
        x = np.array([r[reference] for r in subset])
        y = np.array([r["gls_pred"] for r in subset])
        values += [x, y]
        bias = float(np.mean(y - x))
        ax.scatter(x, y, s=36, color=COLORS[variant], edgecolor="white",
                   linewidth=0.8, label=f"{LABELS[variant]} (bias {bias:+.1f})")
    if values:
        low = float(np.min(np.concatenate(values))) - 2
        high = float(np.max(np.concatenate(values))) + 2
        ax.plot([low, high], [low, high], color="0.5", linewidth=1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
    name = "MITT" if reference == "gls_mitt" else "GE"
    ax.set_xlabel(f"GLS {name} (%)")
    ax.set_ylabel("GLS PWC-Net (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
