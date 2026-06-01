"""Shared plotting style for the revised figures.

Addresses reviewer comments (Major 21, 22, 24):
- subplot panels labelled (a), (b), (c), ... instead of positional words
- larger fonts for axis ticks, titles, labels, and legends

Import and call apply_paper_style() once at the top of each figure
script, then call label_panels(axes) after creating the axes.
"""

import string
import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_paper_style():
    """Enlarge fonts globally for readable published figures."""
    mpl.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 17,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "axes.grid": False,
    })


def label_panels(axes, loc=(-0.02, 1.06), fontsize=16, fontweight="bold"):
    """Annotate each subplot with (a), (b), (c), ... in reading order.

    `axes` may be a 1D or 2D numpy array of Axes (as returned by
    plt.subplots), or a flat list.
    """
    try:
        flat = list(axes.flat)
    except AttributeError:
        flat = list(axes)
    letters = string.ascii_lowercase
    for i, ax in enumerate(flat):
        ax.text(loc[0], loc[1], f"({letters[i]})",
                transform=ax.transAxes,
                fontsize=fontsize, fontweight=fontweight,
                va="bottom", ha="right")
