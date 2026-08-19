"""Figure style for the paper.

Figures are for a printed LaTeX document, so this is a single committed light
palette rather than a theme-aware one.  The categorical hues are used in a fixed
order and never cycled: a series keeps its colour when other series are added or
removed, so colour identifies the thing, not its rank.  Grid and axes are
deliberately recessive; the data is the only thing with saturation.

Series are capped at four per axes.  Where more than that would be needed the
right answer is small multiples (`small_multiples` below), not a ninth hue.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["SERIES", "INK", "use_paper_style", "save", "small_multiples", "annotate"]

# Fixed categorical order.  Validated for adjacent-pair separation under
# colour-vision deficiency; do not reorder or cycle.
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

INK = {
    "surface": "#fcfcfb",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}

# Semantic slots, so a chart reads by role rather than by index.
MARKET = SERIES[0]
ESSCHER = SERIES[1]
MONTE_CARLO = SERIES[2]
BLACK_SCHOLES = INK["secondary"]


def use_paper_style() -> None:
    """Install the style globally.  Call once at the top of a script."""
    mpl.rcParams.update(
        {
            "figure.facecolor": INK["surface"],
            "axes.facecolor": INK["surface"],
            "savefig.facecolor": INK["surface"],
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.dpi": 110,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 8,
            "axes.labelsize": 9,
            "axes.labelcolor": INK["secondary"],
            "axes.edgecolor": INK["axis"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "grid.color": INK["grid"],
            "grid.linewidth": 0.7,
            "xtick.color": INK["muted"],
            "ytick.color": INK["muted"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.handlelength": 1.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
            "lines.solid_capstyle": "round",
        }
    )


def small_multiples(n: int, ncols: int = 3, panel=(3.1, 2.5), **kwargs):
    """A grid of panels -- the answer when more than four series are in play.

    Constrained layout by default: panels carry their own titles and axis
    labels, and without it a tall grid collides one row's x-label with the next
    row's title.
    """
    ncols = max(1, min(ncols, n))  # never open columns there is no panel for
    nrows = -(-n // ncols)
    kwargs.setdefault("layout", "constrained")
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * panel[0], nrows * panel[1]), **kwargs
    )
    # `plt.subplots` returns a bare Axes only for a 1x1 grid and an array
    # otherwise, so the test has to be on what came back rather than on `n`.
    axes = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def annotate(ax, text: str, xy, color: str | None = None, **kwargs) -> None:
    """Direct label in ink, never in the series colour.

    The coloured mark beside it carries identity; text stays readable.
    """
    ax.annotate(
        text,
        xy,
        color=color or INK["secondary"],
        fontsize=8,
        va="center",
        **kwargs,
    )


def save(fig, path: Path | str, also_pdf: bool = True) -> Path:
    """Write a figure as PNG (and PDF, for LaTeX inclusion)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path

