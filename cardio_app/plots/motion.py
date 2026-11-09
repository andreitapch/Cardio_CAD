import numpy as np
from PIL import Image, ImageDraw

from cardio_app.plots import style

A_COLOUR, B_COLOUR, MITT_COLOUR = "#3987e5", "#eb6834", "#ffffff"


def composite(frame, overlay, contours):
    image = Image.fromarray(frame, "L").convert("RGBA")
    if overlay is not None:
        layer = Image.fromarray(overlay, "RGBA").resize(image.size, Image.NEAREST)
        image = Image.alpha_composite(image, layer)
    draw = ImageDraw.Draw(image)
    for x, y, colour, dashed in contours:
        points = list(zip((x - 1).tolist(), (y - 1).tolist()))
        if dashed:
            for i in range(0, len(points) - 1, 2):
                draw.line(points[i:i + 2], fill=colour, width=2)
        else:
            draw.line(points, fill=colour, width=2)
    return image.convert("RGB")


def _curve(ax, frame, title, unit):
    ax.axvline(frame + 1, color=style.MUTED, linestyle=":", linewidth=1)
    ax.legend(frameon=False, fontsize=8)
    style.finish(ax, title=title, xlabel="frame", ylabel=unit)


def epe_curve(frame, results):
    fig, ax = style.axes(6, 3)
    for label, colour, r in results:
        ax.plot(np.arange(1, len(r["epe_per_frame"]) + 1), r["epe_per_frame"], color=colour, linewidth=2, marker="o", markersize=3, label=label)
    _curve(ax, frame, "Contour EPE vs MITT per frame", "mm")
    return fig


def strain_curve(frame, mitt, results):
    fig, ax = style.axes(6, 3)
    if mitt is not None:
        ax.plot(np.arange(1, len(mitt["strain"]) + 1), mitt["strain"], color=style.INK, linewidth=2, linestyle="--", label="MITT")
        ax.axhline(float(mitt["gls_ge"]), color=style.INK_2, linestyle=":", linewidth=1)
        ax.text(1, float(mitt["gls_ge"]), f"GE GLS {float(mitt['gls_ge']):.1f}%", fontsize=8, color=style.INK_2, va="bottom")
        ax.axvline(int(mitt["es_frame"]), color=style.INK_2, linewidth=1)
        ax.text(int(mitt["es_frame"]), 1.0, "ES", transform=ax.get_xaxis_transform(), fontsize=8, ha="center", va="bottom", color=style.INK_2)
    for label, colour, r in results:
        ax.plot(np.arange(1, len(r["strain"]) + 1), r["strain"], color=colour, linewidth=2, label=label)
    _curve(ax, frame, "Longitudinal strain", "%")
    return fig


def epe_scatter(pairs):
    fig, ax = style.axes(4, 4)
    if len(pairs):
        a, b = pairs["epe_a"].to_numpy(), pairs["epe_b"].to_numpy()
        top = float(max(a.max(), b.max())) * 1.05
        ax.plot([0, top], [0, top], color=style.GRID, linewidth=1)
        ax.scatter(a, b, s=18, color=style.INK_2, alpha=0.6, linewidths=0)
    style.finish(ax, xlabel="A EPE (mm)", ylabel="B EPE (mm)")
    return fig
