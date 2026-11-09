"""
Library-wide overview charts — published-quality upgrades of the legacy
``factor_correlation_heatmap.png`` / ``factor_distributions.png`` /
``factor_coverage.png`` trio.

Each function takes the already-loaded set of factor composites (a dict
``{factor_display_name: DataFrame[fsym_primary_listing_id, date, factor_value]}``)
and writes a single PNG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from factor_library.viz.style import (
    DIVERGING_CMAP,
    FACTOR_PALETTE,
    apply_style,
    colour_for,
)


# ---------------------------------------------------------------------------
# Cross-factor correlation — clustermap with dendrogram
# ---------------------------------------------------------------------------
def plot_correlation_clustermap(
    composites: dict[str, pd.DataFrame],
    out_path: Path | str,
    method: str = "spearman",
) -> Path:
    """
    Cluster-ordered Spearman correlation of factor_value across all factors,
    computed on the **inner-join** of (date, fsym_primary_listing_id) common
    to every factor. A dendrogram groups factors with similar cross-sectional
    rankings together.
    """
    apply_style()
    # Build a wide (id-date × factor) frame
    long = []
    for name, df in composites.items():
        t = df[["fsym_primary_listing_id", "date", "factor_value"]].rename(
            columns={"factor_value": name}
        )
        long.append(t.set_index(["fsym_primary_listing_id", "date"]))
    wide = pd.concat(long, axis=1, join="inner")

    corr = wide.corr(method=method)

    g = sns.clustermap(
        corr, cmap=DIVERGING_CMAP, center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 9},
        linewidths=0.4, figsize=(10, 9),
        cbar_kws={"label": f"{method.capitalize()} ρ", "shrink": 0.5},
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.fig.suptitle(
        f"Factor correlation ({method}, inner-join on {len(wide):,} points)",
        y=1.02, fontsize=13, fontweight="semibold",
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path)
    plt.close(g.fig)
    return out_path


# ---------------------------------------------------------------------------
# Coverage timeline — horizontal bars for each factor's valid date range
# ---------------------------------------------------------------------------
def plot_coverage_timeline(
    composites: dict[str, pd.DataFrame],
    out_path: Path | str,
) -> Path:
    """
    One horizontal bar per factor spanning its min-to-max valid date. Shows
    at-a-glance which factors reach back to 1980 (price-only) vs which start
    in the 1990s / 2000s (fundamentals-dependent).
    """
    apply_style()
    rows = []
    for name, df in composites.items():
        rows.append({
            "factor": name,
            "start": df["date"].min(),
            "end":   df["date"].max(),
            "n_ids": df["fsym_primary_listing_id"].nunique(),
            "n_pts": len(df),
        })
    coverage = (pd.DataFrame(rows)
                .sort_values("start")
                .reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(11, 0.45 * len(coverage) + 1.2))
    for i, r in coverage.iterrows():
        ax.barh(i, (r["end"] - r["start"]).days, left=r["start"],
                color=colour_for(r["factor"]), height=0.6, alpha=0.85)
        ax.text(r["end"], i,
                f"  {r['n_ids']} ids · {r['n_pts']/1e3:.0f}k pts",
                va="center", fontsize=9, color="#333")
    ax.set_yticks(range(len(coverage)))
    ax.set_yticklabels(coverage["factor"])
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_title("Factor coverage timeline")
    ax.grid(True, axis="x")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Factor-value distributions — ridgeline by year
# ---------------------------------------------------------------------------
def plot_ridgeline_by_year(
    factor_df: pd.DataFrame,
    factor_name: str,
    out_path: Path | str,
    years: list[int] | None = None,
) -> Path:
    """
    Ridgeline (aka joyplot) of factor_value distributions stacked per year.

    Shows whether the percentile rank stays symmetric over time or whether
    the universe shifts. Year labels annotated on each row via
    transform=ax.transAxes so they remain visible when rows overlap.
    """
    apply_style()
    df = factor_df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    if years is None:
        # Keep at most ~6 evenly-spaced years for readability
        all_years = sorted(df["year"].unique())
        step = max(1, len(all_years) // 6)
        years = all_years[::step]
        # Always include the latest year
        if all_years and all_years[-1] not in years:
            years.append(all_years[-1])
    df = df[df["year"].isin(years)].copy()

    years = sorted(years)
    colour = colour_for(factor_name)
    palette = sns.light_palette(colour, n_colors=len(years) + 3)[3:]

    fig, axes = plt.subplots(
        nrows=len(years), ncols=1,
        figsize=(8.5, 0.55 * len(years) + 1.2),
        sharex=True,
    )
    if len(years) == 1:
        axes = [axes]

    for ax, yr, col in zip(axes, years, palette):
        sub = df[df["year"] == yr]["factor_value"].dropna()
        if len(sub) > 2:
            sns.kdeplot(sub, ax=ax, fill=True, alpha=0.75, lw=1.2,
                        color=col, clip=(0, 1), clip_on=False)
        ax.axhline(0, lw=0.5, color="grey", clip_on=False)
        ax.set_xlim(0, 1)
        ax.set_yticks([]); ax.set_ylabel("")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.text(-0.02, 0.15, str(yr), transform=ax.transAxes,
                ha="right", va="center", fontsize=10, fontweight="bold",
                color=col)

    axes[-1].set_xlabel("factor_value")
    fig.suptitle(
        f"{factor_name} — factor-value distribution by year",
        y=0.995, fontsize=13, fontweight="semibold",
    )
    fig.subplots_adjust(hspace=-0.35, left=0.10, right=0.97, top=0.92, bottom=0.08)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
