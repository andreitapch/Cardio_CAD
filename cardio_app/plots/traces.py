import math

import numpy as np

from cardio_app.plots import style
from pattern_recognition.utils.plot_dics import Cardiac_view_dict, segment_color_dict, wall_color_dict

COLOURS = {"segment_wall": wall_color_dict, "curve_segment": segment_color_dict, "cardiac_view": Cardiac_view_dict}


def colour(mapping, label, index=0):
    return mapping.get(label) or mapping.get(label.lower()) or style.SERIES[index % len(style.SERIES)]


def envelope_grid(groups, by, events, title):
    mapping = COLOURS[by]
    cols = min(3, len(groups))
    rows = math.ceil(len(groups) / cols)
    fig, axes = style.axes(4.2 * cols, 2.6 * rows, nrows=rows, ncols=cols, sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, (label, array) in zip(axes.flat, groups.items()):
        ax.set_visible(True)
        x = np.arange(array.shape[1])
        ax.fill_between(x, np.nanmin(array, axis=0), np.nanmax(array, axis=0), color=colour(mapping, label), alpha=0.2, linewidth=0)
        ax.plot(x, np.nanmean(array, axis=0), color=colour(mapping, label), linewidth=2)
        for time, name in events:
            ax.axvline(time, color=style.INK_2, linestyle="--", linewidth=1)
            ax.text(time, 0.02, name, transform=ax.get_xaxis_transform(), fontsize=8, ha="center", va="bottom", color=style.INK_2)
        style.finish(ax, title=f"{label} (n={array.shape[0]})", xlabel="sample", ylabel="strain")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=style.INK)
    return fig


def segment_lines(rows, title):
    fig, ax = style.axes(9, 5)
    for index, (label, trace) in enumerate(zip(rows["label"], rows["trace"])):
        ax.plot(trace, color=colour(segment_color_dict, label, index), linewidth=1.5, label=label)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    style.finish(ax, title=title, xlabel="sample", ylabel="strain")
    return fig
