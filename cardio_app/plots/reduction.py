import math

import numpy as np

from cardio_app.plots import style

STYLES = {"original": (style.INK, "-"), "PCA": (style.SERIES[0], "--"), "Autoencoder": (style.SERIES[1], ":")}


def comparison_grid(frames, study, title):
    original = frames["original"]
    rows = original[original["study_nr"] == study].sort_values("position")
    cols = 3
    count = len(rows)
    fig, axes = style.axes(4.0 * cols, 2.3 * math.ceil(count / cols), nrows=math.ceil(count / cols), ncols=cols,
                           sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, row in zip(axes.flat, rows.itertuples()):
        ax.set_visible(True)
        for label, frame in frames.items():
            match = frame[(frame["study_nr"] == study) & (frame["position"] == row.position)]
            if len(match):
                colour, dash = STYLES.get(label, (style.SERIES[2], "-."))
                ax.plot(match["trace"].iloc[0], color=colour, linestyle=dash, linewidth=1.5, label=label)
        style.finish(ax, title=row.curve_segment)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=style.INK)
    return fig


def loss_curves(history, title):
    fig, ax = style.axes(7, 3.5)
    for index, (name, values) in enumerate(sorted(history.items())):
        values = np.asarray(values)
        ax.plot(values[:, 1], color=style.SERIES[index % len(style.SERIES)], linewidth=1.2, label=name)
    ax.set_yscale("log")
    if len(history) <= 8:
        ax.legend(frameon=False, fontsize=8, ncol=2)
    style.finish(ax, title=title, xlabel="epoch", ylabel="validation MSE")
    return fig
