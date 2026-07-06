"""Shared plotting defaults for the imprints figures.

Centralizes the font, DPI, and color style so every figure looks the same, and
provides a single helper that writes each figure as PNG, SVG, and PDF.

These scripts run standalone (``python figures/scripts/ps_nyc_share.py``), so
the script directory is on ``sys.path`` and siblings can ``import style``.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cycler import cycler

#: Preferred font, with automatic fallback to matplotlib's stock sans-serif.
FONT_FAMILY = "Helvetica Now Micro"

#: Vector and raster formats every figure is written in.
OUTPUT_FORMATS = ("png", "svg", "pdf")

#: All figures share this resolution.
DPI = 1200

#: Okabe & Ito colorblind-safe qualitative palette, ordered for adjacent
#: contrast. Used as the default color cycle and as the source for the semantic
#: colors below. See https://jfly.uni-koeln.de/color/.
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

#: Distinct marker shapes cycled across line series so they stay separable in
#: grayscale print and for the colorblind, regardless of color.
MARKERS = ["o", "s", "^", "D", "v", ">", "<", "p", "*", "X", "h", "P"]

#: Linestyles cycled as a tertiary cue once there are more series than colors.
LINESTYLES = ["-", "--", "-.", ":"]

#: Fixed colors for meanings that recur across figures, so e.g. New York City is
#: the same blue everywhere it appears.
COLOR_NYC = OKABE_ITO[0]
COLOR_OTHER = OKABE_ITO[1]
COLOR_NOPLACE = OKABE_ITO[2]

#: Neutral gray for elements that should recede: the 50% rule lines, CI bands,
#: and faint background/context lines.
COLOR_REFERENCE = "0.5"


def series_style(i: int) -> dict:
    """Color/marker/linestyle for the ``i``-th line series in a multi-series chart.

    Markers cycle fastest, then colors, then linestyles, so adjacent series
    differ in shape first. The result is triple-encoded and stays legible in
    grayscale print and for the colorblind. Spread as ``**series_style(i)`` into
    a ``plot`` call.
    """
    return {
        "color": OKABE_ITO[i % len(OKABE_ITO)],
        "marker": MARKERS[i % len(MARKERS)],
        "linestyle": LINESTYLES[(i // len(OKABE_ITO)) % len(LINESTYLES)],
    }


def apply_style(base: str = "default") -> None:
    """Apply the shared figure defaults.

    Applies the ``base`` matplotlib style first, then overlays the shared font,
    DPI, and Okabe-Ito color cycle *after* it (a style reset would otherwise
    clobber them).
    """
    plt.style.use(base)

    # Prepend the preferred font to the existing sans-serif fallback list so a
    # missing font degrades gracefully instead of erroring.
    sans_fallback = [f for f in mpl.rcParams["font.sans-serif"] if f != FONT_FAMILY]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [FONT_FAMILY, *sans_fallback]
    mpl.rcParams["figure.dpi"] = DPI
    mpl.rcParams["savefig.dpi"] = DPI
    mpl.rcParams["axes.prop_cycle"] = cycler(color=OKABE_ITO)


def percent_yaxis(ax) -> None:
    """Format an axis's y ticks as whole-number percents (e.g. ``50%``).

    For axes whose values are already on a 0-100 scale (a share times 100).
    Appends the ``%`` sign to each integer tick so the axis label can describe
    the quantity without carrying a ``%`` of its own.
    """
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))


def save_figure(output_path) -> None:
    """Save the current figure as PNG, SVG, and PDF.

    ``output_path`` is treated as a base path; the figure is written next to it
    once per format in :data:`OUTPUT_FORMATS`, regardless of the given suffix.
    """
    base = Path(output_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in OUTPUT_FORMATS:
        path = base.with_suffix(f".{fmt}")
        plt.savefig(path)
        print(f"Saved figure to: {path}")
