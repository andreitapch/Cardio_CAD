import math

from matplotlib.patches import Wedge

from cardio_app.plots import style

RINGS = ((0.68, 1.0), (0.36, 0.68), (0.04, 0.36))


def bullseye(labels, highlighted, title):
    fig, ax = style.axes(4.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")
    for position in range(1, 19):
        ring, wedge = divmod(position - 1, 6)
        inner, outer = RINGS[ring]
        start = 60 + 60 * wedge
        colour = style.SERIES[7] if position in highlighted else "#f0efec"
        ax.add_patch(Wedge((0, 0), outer, start, start + 60, width=outer - inner, facecolor=colour, edgecolor="white", linewidth=1.5))
        angle = math.radians(start + 30)
        radius = (inner + outer) / 2
        ax.text(radius * math.cos(angle), radius * math.sin(angle), labels.get(position, str(position)),
                ha="center", va="center", fontsize=7, color=style.INK)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title, loc="left", fontsize=11, color=style.INK)
    return fig
