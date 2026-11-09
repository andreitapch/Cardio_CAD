import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import precision_recall_curve, roc_curve

from cardio_app.plots import style
from cardio_app.services.classification import shap_explainer
from pattern_recognition.utils.plot_dics import segment_color_dict


def confusion_matrix(cm, title):
    cm = np.asarray(cm)
    fig, ax = style.axes(3.6, 3.2)
    ax.imshow(cm, cmap="Blues")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > threshold else style.INK)
    ax.set_xticks([0, 1], ["0", "1"])
    ax.set_yticks([0, 1], ["0", "1"])
    ax.grid(False)
    ax.set_title(title, loc="left", fontsize=11, color=style.INK)
    ax.set_xlabel("Predicted", color=style.INK_2)
    ax.set_ylabel("True", color=style.INK_2)
    return fig


def shap_beeswarm(shap_values, title):
    figure = plt.figure(figsize=(8, 5))
    shap.plots.beeswarm(shap_explainer.top_features(shap_values), show=False)
    plt.title(title, loc="left", fontsize=11)
    figure = plt.gcf()
    figure.set_tight_layout(True)
    return figure


def _curves(results, curve, title, xlabel, ylabel, baseline=None, chance=False):
    fig, ax = style.axes(5.5, 4.5)
    for index, (label, y_true, y_proba) in enumerate(results):
        x, y = curve(y_true, y_proba)
        ax.plot(x, y, color=style.SERIES[index % len(style.SERIES)], linewidth=1.8, label=label)
    if baseline is not None:
        ax.axhline(baseline, color=style.MUTED, linestyle="--", linewidth=1)
        ax.text(0.99, baseline, f"prevalence {baseline:.3f}", ha="right", va="bottom", fontsize=8, color=style.INK_2)
    if chance:
        ax.plot([0, 1], [0, 1], color=style.MUTED, linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    style.finish(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    return fig


def pr_curves(results, baseline):
    def curve(y_true, y_proba):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        return recall, precision
    return _curves(results, curve, "Precision-recall, pooled over held-out folds", "recall", "precision", baseline=baseline)


def roc_curves(results):
    def curve(y_true, y_proba):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        return fpr, tpr
    return _curves(results, curve, "ROC, pooled over held-out folds", "false positive rate", "true positive rate", chance=True)


def loading_segments(curves, component, es_frame=54):
    fig, ax = style.axes(9, 5)
    for label, curve, top in curves:
        ax.plot(curve, color=segment_color_dict.get(label, style.INK_2), linewidth=2.5 if top else 1, label=label, alpha=1 if top else 0.7)
    ax.axvline(es_frame, color=style.INK_2, linestyle=":", linewidth=1)
    ax.text(es_frame, 1.0, "ES", transform=ax.get_xaxis_transform(), fontsize=8, ha="center", va="bottom", color=style.INK_2)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    style.finish(ax, title=f"PCA component {component} loadings by segment", xlabel="sample", ylabel="loading")
    return fig


def feature_means_by_class(x, y, features):
    frame = x[features].copy()
    frame["class"] = np.asarray(y)
    means = frame.groupby("class").mean().T
    fig, ax = style.axes(max(6, 0.5 * len(features)), 4)
    width = 0.8 / len(means.columns)
    positions = np.arange(len(means))
    for index, label in enumerate(means.columns):
        ax.bar(positions + index * width, means[label].to_numpy(), width, label=f"remodeling {label}", color=style.SERIES[index])
    ax.set_xticks(positions + width * (len(means.columns) - 1) / 2, means.index, rotation=45, ha="right", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    style.finish(ax, title="Average standardised value by remodeling class", ylabel="z-score")
    return fig


def feature_importance_comparison(scores, columns):
    scores = scores.sort_values("SHAP", ascending=False)
    fig, ax = style.axes(max(6, 0.5 * len(scores)), 4)
    width = 0.8 / len(columns)
    positions = np.arange(len(scores))
    for index, column in enumerate(columns):
        ax.bar(positions + index * width, scores[column].to_numpy(), width, label=column, color=style.SERIES[index])
    ax.set_xticks(positions + width * (len(columns) - 1) / 2, scores.index, rotation=45, ha="right", fontsize=8)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    style.finish(ax, title="Feature importance across methods", ylabel="normalised score")
    return fig
