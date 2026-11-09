"""
Single source of truth for matplotlib / seaborn styling.

Import :func:`apply_style` once at the top of any viz or analysis module
to get consistent, publication-quality figures. All rc parameters live
here — do not set them ad-hoc elsewhere.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# A stable factor → colour mapping so that the same factor always looks the
# same across every chart in the library. Keys match the factor display names.
FACTOR_PALETTE = {
    "Momentum":            "#C0392B",  # red
    "Value":               "#2874A6",  # blue
    "Quality":             "#239B56",  # green
    "Quality (Composite)": "#145A32",  # dark green
    "Investment":          "#7D3C98",  # purple
    "Size":                "#B9770E",  # ochre
    "Yield":               "#A04000",  # brown
    "Liquidity":           "#1A5276",  # navy
    "Low Beta / BAB":      "#AF601A",  # copper
    "Volatility / Low Risk": "#5B2C6F",  # plum
    "Market":              "#707B7C",  # grey
    "Profitability (FF)":  "#0E6655",  # teal
}

# Diverging colormap for correlation / factor-value heatmaps.
DIVERGING_CMAP = "RdBu_r"

# Sequential colormap for coverage / density plots.
SEQUENTIAL_CMAP = "rocket_r"


def apply_style() -> None:
    """Set the library-wide matplotlib/seaborn style. Idempotent."""
    sns.set_theme(
        context="paper",
        style="white",
        font="DejaVu Sans",
        rc={
            "figure.figsize":     (10, 6),
            "figure.dpi":         110,
            "savefig.dpi":        300,
            "savefig.bbox":       "tight",
            "axes.titleweight":   "semibold",
            "axes.titlesize":     13,
            "axes.labelsize":     11,
            "axes.spines.top":    False,
            "axes.spines.right":  False,
            "xtick.labelsize":    10,
            "ytick.labelsize":    10,
            "legend.frameon":     False,
            "legend.fontsize":    10,
            "lines.linewidth":    1.6,
            "grid.alpha":         0.25,
            "grid.linestyle":     "--",
        },
    )


def colour_for(factor_name: str) -> str:
    """Return the canonical colour for a factor (fallback: seaborn default)."""
    # Normalise the on-disk filename convention ("Quality_(Composite)") back
    # to the display form we use as the palette key.
    display = factor_name.replace("___", " / ").replace("_", " ")
    return FACTOR_PALETTE.get(display, sns.color_palette()[0])
