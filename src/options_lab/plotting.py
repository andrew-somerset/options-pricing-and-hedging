"""Shared matplotlib styling so every figure in reports/ looks the same."""

import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def figure(ncols=1, nrows=1, figsize=(8, 5), **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE, **kwargs)
    for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
        style(ax)
    return fig, axes


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def title(fig, text, note=None):
    fig.suptitle(text, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    if note:
        fig.text(0.01, -0.02, note, color=MUTED, fontsize=7.5, wrap=True)
