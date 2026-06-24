"""Shared plotting defaults for the imprints figures.

Centralizes the font, DPI, and color style so every figure looks the same, and
provides a single helper that writes each figure as PNG, SVG, and PDF.

These scripts run standalone (``python figures/scripts/fig1.py``), so the
script directory is on ``sys.path`` and siblings can ``import style``.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Preferred font, with automatic fallback to matplotlib's stock sans-serif.
FONT_FAMILY = "Helvetica Now Micro"

#: Vector and raster formats every figure is written in.
OUTPUT_FORMATS = ("png", "svg", "pdf")

#: All figures share this resolution.
DPI = 600


def apply_style(base: str = "grayscale") -> None:
    """Apply the shared figure defaults.

    Applies the ``base`` matplotlib style first, then overlays the shared font
    and DPI settings *after* it (a style reset would otherwise clobber them).
    """
    plt.style.use(base)

    # Prepend the preferred font to the existing sans-serif fallback list so a
    # missing font degrades gracefully instead of erroring.
    sans_fallback = [f for f in mpl.rcParams["font.sans-serif"] if f != FONT_FAMILY]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [FONT_FAMILY, *sans_fallback]
    mpl.rcParams["figure.dpi"] = DPI
    mpl.rcParams["savefig.dpi"] = DPI


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
