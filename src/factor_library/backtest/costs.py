"""
Transaction-cost and rebalance-frequency sensitivity for long/short factor
portfolios. Addresses the most common buy-side question:
"what's the Sharpe *after* cost?"

The long/short book is the top-quantile vs bottom-quantile equal-weight
portfolio produced by alphalens. `turnover` is the L1 change in normalised
weights (sum absolute diff / 2); `cost` is applied as
    net_return_t = gross_return_t - turnover_t * cost_per_side_bps / 1e4
with daily, weekly (Fri→Fri) or monthly (last-business-day) rebalance.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import empyrical as emp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning, module="alphalens")
import alphalens as al  # noqa: E402

from factor_library.analysis.ic import prepare_alphalens_input
from factor_library.viz.style import apply_style, DIVERGING_CMAP, colour_for


REBAL_FREQ_MAP = {
    "daily":   None,   # every day
    "weekly":  "W-FRI",
    "monthly": "BME",  # business month-end
}


# ---------------------------------------------------------------------------
# Weights & turnover
# ---------------------------------------------------------------------------
def long_short_weights(
    merged: pd.DataFrame,
    quantiles: int = 5,
) -> pd.DataFrame:
    """
    Build the daily (date × asset) equal-weight long/short weight matrix:
    +1/n_long in the top quantile, -1/n_short in the bottom quantile,
    0 everywhere else. Weights sum to 0 each day, L1 norm = 2.
    """
    q = merged["factor_quantile"].astype(int)
    top = (q == quantiles).astype(float)
    bot = (q == 1).astype(float)
    # Normalise per date
    dates = merged.index.get_level_values(0)
    top_norm = top.groupby(dates).transform(lambda s: s / s.sum() if s.sum() else 0.0)
    bot_norm = bot.groupby(dates).transform(lambda s: s / s.sum() if s.sum() else 0.0)
    w = top_norm - bot_norm
    w.name = "weight"
    wide = w.unstack(level=1).fillna(0.0)
    return wide


def apply_rebalance(
    weights: pd.DataFrame,
    freq: str = "daily",
) -> pd.DataFrame:
    """
    Resample the weight matrix to a coarser rebalance cadence. Between
    rebalance dates the weights are held constant (forward-filled).
    """
    if freq == "daily":
        return weights
    rule = REBAL_FREQ_MAP.get(freq)
    if rule is None:
        raise ValueError(f"Unknown rebalance freq: {freq!r}")
    rebal_dates = weights.resample(rule).last().index
    return weights.reindex(rebal_dates).reindex(weights.index, method="ffill")


def turnover_series(weights: pd.DataFrame) -> pd.Series:
    """
    Daily portfolio turnover = 0.5 × Σ|w_t - w_{t-1}|.
    First day counts as full cost of establishing the book.
    """
    diff = weights.diff().abs().sum(axis=1) * 0.5
    # Day zero: cost of opening the book
    diff.iloc[0] = weights.iloc[0].abs().sum() * 0.5
    diff.name = "turnover"
    return diff


# ---------------------------------------------------------------------------
# Return composition
# ---------------------------------------------------------------------------
def gross_long_short_return(
    merged: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    """
    The gross (pre-cost) daily long/short return, computed by taking the
    dot product of the held weights and the realised 1D forward return.
    """
    r1 = merged["1D"].unstack(level=1).reindex_like(weights).fillna(0.0)
    # Weights at start of day t applied to return realised at day t
    gross = (weights.shift(1).fillna(0.0) * r1).sum(axis=1)
    gross.name = "gross_return"
    return gross


def apply_cost(
    gross: pd.Series,
    turnover: pd.Series,
    cost_per_side_bps: float,
) -> pd.Series:
    """
    Net return after linear per-side cost.
    cost_per_side_bps: 5 means 0.05 % per unit of notional turned over.
    """
    cost = turnover * (cost_per_side_bps / 1e4)
    net = gross - cost
    net.name = "net_return"
    return net


# ---------------------------------------------------------------------------
# Public: single-factor sensitivity grid
# ---------------------------------------------------------------------------
def cost_sharpe_grid(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    factor_name: str,
    cost_bps_grid: Iterable[float] = (0, 2, 5, 10, 20),
    rebal_freqs: Iterable[str] = ("daily", "weekly", "monthly"),
    quantiles: int = 5,
) -> pd.DataFrame:
    """
    Build a Sharpe-ratio grid: rows = rebalance frequency, cols = cost bps.
    Extra columns report ann_return, ann_turnover, and gross_sharpe for
    context. Returns a tidy DataFrame with one row per (freq, cost) cell.
    """
    merged = prepare_alphalens_input(factor_df, prices, horizons=(1,),
                                     quantiles=quantiles, max_loss=0.50)
    rows = []
    for freq in rebal_freqs:
        w_freq = apply_rebalance(long_short_weights(merged, quantiles=quantiles),
                                 freq=freq)
        gross = gross_long_short_return(merged, w_freq)
        to = turnover_series(w_freq)
        ann_to = float(to.sum() / (len(to) / 252))  # annualised one-way turnover
        gross_sharpe = float(emp.sharpe_ratio(gross, annualization=252))
        for cost in cost_bps_grid:
            net = apply_cost(gross, to, cost)
            rows.append({
                "factor":        factor_name,
                "rebalance":     freq,
                "cost_bps":      float(cost),
                "ann_turnover":  ann_to,
                "gross_sharpe":  gross_sharpe,
                "net_sharpe":    float(emp.sharpe_ratio(net, annualization=252)),
                "net_ann_return": float(emp.annual_return(net)),
                "net_max_dd":    float(emp.max_drawdown((1 + net).cumprod())),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_cost_sharpe_heatmap(
    grid: pd.DataFrame,
    out_path: Path | str,
) -> Path:
    """
    For a single factor, plot a (rebalance freq × cost) heatmap of net Sharpe.
    """
    apply_style()
    factor_name = grid["factor"].iloc[0]
    piv = grid.pivot(index="rebalance", columns="cost_bps", values="net_sharpe")
    piv = piv.reindex(["daily", "weekly", "monthly"])
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    sns.heatmap(
        piv, cmap=DIVERGING_CMAP, center=0, annot=True, fmt=".2f",
        cbar_kws={"label": "Annualised Sharpe (net)"}, ax=ax, linewidths=0.5,
    )
    ax.set_title(f"{factor_name} — net Sharpe under cost × rebalance")
    ax.set_xlabel("Round-trip cost per side (bps)")
    ax.set_ylabel("Rebalance frequency")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def baseline_slice(
    grids: pd.DataFrame,
    rebalance: str = "monthly",
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Extract a cross-factor baseline slice, e.g. monthly @ 5 bps per side.
    """
    rows = grids[
        (grids["rebalance"] == rebalance)
        & (np.isclose(grids["cost_bps"].astype(float), float(cost_bps)))
    ].copy()
    if rows.empty:
        return rows
    return rows.sort_values("net_sharpe", ascending=False).reset_index(drop=True)


def plot_cost_baseline_bar(
    baseline: pd.DataFrame,
    out_path: Path | str,
    rebalance: str = "monthly",
    cost_bps: float = 5.0,
) -> Path:
    """
    Cleaner decision-view chart: one bar per factor at a fixed
    (rebalance, cost) scenario, sorted by net Sharpe.
    """
    apply_style()
    if baseline.empty:
        raise ValueError("baseline dataframe is empty")

    df = baseline.sort_values("net_sharpe", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(df) + 1.6))
    colors = [colour_for(f) for f in df["factor"]]
    ax.barh(df["factor"], df["net_sharpe"], color=colors, alpha=0.88)

    for i, row in df.reset_index(drop=True).iterrows():
        x = row["net_sharpe"]
        ha = "left" if x >= 0 else "right"
        x_text = x + (0.01 if x >= 0 else -0.01)
        ax.text(
            x_text,
            i,
            f"{row['net_sharpe']:+.2f}  ({row['net_ann_return']*100:+.1f}% ann)",
            va="center",
            ha=ha,
            fontsize=8.8,
            color="#333",
        )

    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Net Sharpe (annualised)")
    ax.set_ylabel("")
    ax.set_title(f"Cost sensitivity snapshot — {rebalance} rebalance @ {cost_bps:g} bps/side")
    ax.grid(True, axis="x")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
