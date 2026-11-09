import numpy as np

from cardio_app.plots import style


def counts(series, title):
    values = series.value_counts().sort_index()
    fig, ax = style.axes(5, 4)
    ax.bar(values.index.astype(str), values.to_numpy(), color=style.SERIES[0], width=0.7)
    style.finish(ax, title=title, xlabel=series.name, ylabel="count")
    return fig


def grouped_sums(frame, by, columns, title):
    totals = frame.groupby(by)[columns].sum()
    fig, ax = style.axes(6, 4)
    width = 0.8 / len(columns)
    x = np.arange(len(totals))
    for i, column in enumerate(columns):
        ax.bar(x + i * width, totals[column].to_numpy(), width, label=column, color=style.SERIES[i])
    ax.set_xticks(x + width * (len(columns) - 1) / 2, totals.index.astype(str))
    ax.legend(frameon=False)
    style.finish(ax, title=title, xlabel=by, ylabel="count")
    return fig
