"""Evaluation helpers shared by the baseline and shared-track scripts:
threshold metrics with confusion-matrix counts, and the two report
figures (precision-recall curve, confusion matrix with counts).

Accuracy is not reported: with ~0.13% fraud a model that flags nothing
is 99.87% accurate.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

# Light-surface chart colors: categorical slots 1-3 in fixed order,
# with line styles as a second, color-independent cue.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_STYLES = [
    {"color": "#2a78d6", "linestyle": "-"},
    {"color": "#eb6834", "linestyle": "--"},
    {"color": "#1baf7a", "linestyle": ":"},
]


def threshold_metrics(y_true, scores, threshold):
    """AP, precision, recall, F1 and TN/FP/FN/TP at one threshold."""
    predictions = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "threshold": float(threshold),
        "average_precision": average_precision_score(y_true, scores),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_pr_curves(curves, path, title, operating_point=None):
    """Precision-recall curves on one chart.

    curves: list of (label, y_true, scores), at most three.
    operating_point: optional (label, recall, precision) marker for the
    chosen threshold.
    """
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    _style_axes(ax)

    for (label, y_true, scores), style in zip(curves, SERIES_STYLES):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(
            recall, precision, linewidth=2, **style,
            label=f"{label} (AP {ap:.4f})",
        )

    if operating_point is not None:
        label, recall, precision = operating_point
        ax.plot(
            recall, precision, marker="o", markersize=8,
            color=TEXT_PRIMARY, markeredgecolor=SURFACE,
            markeredgewidth=2, linestyle="none", label=label,
        )

    base_rate = float(np.mean(curves[0][1]))
    ax.axhline(base_rate, color=TEXT_SECONDARY, linewidth=1,
               linestyle="-.", label=f"No skill (fraud rate {base_rate:.4f})")

    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (share of fraud caught)", color=TEXT_PRIMARY)
    ax.set_ylabel("Precision (share of alerts that are fraud)",
                  color=TEXT_PRIMARY)
    ax.set_title(title, color=TEXT_PRIMARY, loc="left")
    ax.legend(loc="lower left", frameon=False, labelcolor=TEXT_PRIMARY)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion_counts(metrics, path, title):
    """2x2 confusion matrix with counts, from threshold_metrics() output."""
    counts = np.array([
        [metrics["true_negatives"], metrics["false_positives"]],
        [metrics["false_negatives"], metrics["true_positives"]],
    ])
    names = [["True negatives", "False positives"],
             ["Missed fraud (FN)", "Caught fraud (TP)"]]
    # Correct cells blue, error cells orange; the text carries the meaning.
    fills = [["#cde2fb", "#fbd9c9"], ["#fbd9c9", "#cde2fb"]]

    fig, ax = plt.subplots(figsize=(6, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for row in range(2):
        for col in range(2):
            ax.add_patch(plt.Rectangle(
                (col, 1 - row), 0.98, 0.98, facecolor=fills[row][col],
                edgecolor=SURFACE, linewidth=2,
            ))
            ax.text(col + 0.49, 1 - row + 0.56, f"{counts[row, col]:,}",
                    ha="center", va="center", fontsize=18,
                    color=TEXT_PRIMARY, fontweight="bold")
            ax.text(col + 0.49, 1 - row + 0.32, names[row][col],
                    ha="center", va="center", fontsize=10,
                    color=TEXT_SECONDARY)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.49, 1.49], ["Predicted legitimate", "Predicted fraud"])
    ax.set_yticks([1.49, 0.49], ["Actual legitimate", "Actual fraud"])
    ax.tick_params(length=0, colors=TEXT_PRIMARY)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"{title}\nthreshold {metrics['threshold']:g} | "
        f"precision {metrics['precision']:.4f} | "
        f"recall {metrics['recall']:.4f}",
        color=TEXT_PRIMARY, loc="left", fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
