"""Plotting layer — publication-quality charts built on seaborn."""
from factor_library.viz.style import (
    FACTOR_PALETTE,
    DIVERGING_CMAP,
    SEQUENTIAL_CMAP,
    apply_style,
    colour_for,
)

__all__ = [
    "apply_style", "colour_for",
    "FACTOR_PALETTE", "DIVERGING_CMAP", "SEQUENTIAL_CMAP",
]
