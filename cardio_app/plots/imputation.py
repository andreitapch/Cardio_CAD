from cardio_app.plots import style


def imputed_vs_original(rows, title):
    fig, ax = style.axes(8, 4)
    for index, row in enumerate(rows.itertuples()):
        colour = style.SERIES[index % len(style.SERIES)]
        ax.plot(row.original, color=colour, linewidth=1.5, alpha=0.45, label=f"original {row.label}")
        ax.plot(row.imputed, color=colour, linewidth=1.5, linestyle="--", label=f"imputed {row.label}")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    style.finish(ax, title=title, xlabel="sample", ylabel="strain")
    return fig


def study_traces(rows, title):
    fig, ax = style.axes(8, 4.5)
    for row in rows.itertuples():
        if row.isImputed:
            ax.plot(row.trace, color=style.SERIES[7], linewidth=1.5, linestyle="--", label=f"imputed {row.curve_segment}")
        else:
            ax.plot(row.trace, color="#c3c2b7", linewidth=1)
    ax.plot([], [], color="#c3c2b7", linewidth=1, label="original segments")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    style.finish(ax, title=title, xlabel="sample", ylabel="strain")
    return fig


def k_curve(errors, best):
    fig, ax = style.axes(6, 3)
    ax.plot(errors["k"], errors["mean RRSSE"], color=style.SERIES[0], linewidth=2, marker="o", markersize=4)
    ax.axvline(best, color=style.INK_2, linestyle=":", linewidth=1)
    style.finish(ax, title="Mean RRSSE over folds per K", xlabel="K", ylabel="RRSSE")
    return fig
