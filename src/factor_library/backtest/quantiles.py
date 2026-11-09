"""
Quantile cumulative-return analysis — the canonical factor research picture.

Groups stocks into N equal-size buckets by factor_value on each day, holds
for one period, and accumulates the daily equal-weight return of each
bucket. The long-short (top minus bottom) bucket is overlaid. A monotone
ordering of cumulative curves is strong visual evidence that the factor
rank is predictive; a messy / non-monotonic ordering is a red flag even
if the Q5-Q1 spread happens to look profitable.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning, module="alphalens")

import alphalens as al  # noqa: E402

from factor_library.analysis.ic import prepare_alphalens_input
from factor_library.viz.style import apply_style, colour_for


def quantile_cumulative_returns(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    quantiles: int = 5,
    horizon_days: int = 1,
    compound: bool = True,
) -> pd.DataFrame:
    """
    Build the per-quantile cumulative return curve plus a long/short
    (Q_max − Q_1) series.

    Returns a wide DataFrame indexed by date with columns
    ``["Q1", "Q2", …, f"Q{quantiles}", "Long_Short"]``.
    """
    merged = prepare_alphalens_input(
        factor_df, prices,
        horizons=(horizon_days,), quantiles=quantiles, max_loss=0.50,
    )
    col = f"{horizon_days}D"
    dates = merged.index.get_level_values(0)
    daily = (merged
             .groupby([dates, "factor_quantile"], observed=True)[col]
             .mean()
             .unstack())
    daily.columns = [f"Q{int(c)}" for c in daily.columns]
    daily = daily.sort_index(axis=1)
    daily = daily.fillna(0.0)
    daily["Long_Short"] = daily[f"Q{quantiles}"] - daily["Q1"]
    if compound:
        equity = (1.0 + daily).cumprod() - 1.0
    else:
        equity = daily.cumsum()
    return equity


def _quantile_palette(n: int, base_hex: str) -> list[str]:
    """Distinguishable categorical palette for Q1..Qn.

    Switched from ``sns.light_palette`` (same-hue shades — hard to tell
    Q2 / Q3 / Q4 apart) to ``sns.color_palette('viridis_r', n)`` which is
    perceptually uniform AND uses a full hue rotation, so each quintile
    is clearly distinct without losing the natural Q1→Qn ordering. The
    ``base_hex`` arg is kept for signature compatibility but ignored.
    """
    return sns.color_palette("viridis_r", n_colors=n).as_hex()


def plot_quantile_cumulative_returns(
    equity: pd.DataFrame,
    factor_name: str,
    out_path: Path | str,
    quantiles: int = 5,
    log_scale: bool = True,
) -> Path:
    """
    Single chart with ``quantiles`` cumulative-return lines (Q1..Qn
    coloured by viridis_r) plus a dashed Long_Short overlay.
    """
    apply_style()
    colour = colour_for(factor_name)
    palette = _quantile_palette(quantiles, colour)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for i in range(1, quantiles + 1):
        col = f"Q{i}"
        if col not in equity.columns:
            continue
        ax.plot(equity.index, (1 + equity[col]).values, color=palette[i - 1],
                lw=1.6, label=col)
    if "Long_Short" in equity.columns:
        ax.plot(equity.index, (1 + equity["Long_Short"]).values,
                color="#111", lw=2.0, linestyle="--", label=f"Q{quantiles} − Q1")

    ax.axhline(1.0, color="grey", lw=0.8)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(f"{factor_name} — quantile cumulative return (compounded, gross)")
    ax.set_ylabel("Cumulative wealth (× initial, log scale)")
    ax.set_xlabel("")
    ax.legend(ncols=quantiles + 1, loc="upper left", fontsize=10,
              columnspacing=1.2, handlelength=1.6)
    ax.grid(True, axis="y", which="both")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def quantile_annual_summary(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    quantiles: int = 5,
    horizon_days: int = 1,
) -> pd.DataFrame:
    """
    Per-quantile annualised return, volatility, Sharpe, and hit rate.
    Handy as a small companion table to the cumulative-return chart.
    """
    merged = prepare_alphalens_input(
        factor_df, prices,
        horizons=(horizon_days,), quantiles=quantiles, max_loss=0.50,
    )
    col = f"{horizon_days}D"
    dates = merged.index.get_level_values(0)
    daily = (merged
             .groupby([dates, "factor_quantile"], observed=True)[col]
             .mean()
             .unstack())
    daily.columns = [f"Q{int(c)}" for c in daily.columns]
    daily = daily.sort_index(axis=1)
    daily["Long_Short"] = daily[f"Q{quantiles}"] - daily["Q1"]

    rows = []
    for col in daily.columns:
        s = daily[col].dropna()
        if s.empty:
            continue
        mu = s.mean()
        sd = s.std(ddof=1)
        rows.append({
            "bucket":         col,
            "ann_return":     float(((1 + mu) ** 252 - 1)),
            "ann_volatility": float(sd * np.sqrt(252)),
            "sharpe":         float(mu / sd * np.sqrt(252)) if sd else np.nan,
            "hit_rate":       float((s > 0).mean()),
            "n_days":         int(len(s)),
        })
    return pd.DataFrame(rows).set_index("bucket")
