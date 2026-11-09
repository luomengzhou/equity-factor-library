"""
Quantstats HTML tear sheets — one-page comprehensive factor performance report.

The tear sheet bundles cumulative return, rolling Sharpe, drawdown, monthly
return heatmap, worst-drawdown table, return distribution, EOY returns, and
much more into a single HTML file. Replaces the constellation of ad-hoc PNGs
the old pipeline produced per factor.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="alphalens")
warnings.filterwarnings("ignore", category=FutureWarning, module="quantstats")

import alphalens as al  # noqa: E402
import quantstats as qs  # noqa: E402


def long_short_daily_returns(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    quantiles: int = 5,
    horizon_days: int = 1,
    max_loss: float = 0.50,
) -> pd.Series:
    """
    Build the 1-day long-short daily return series for a factor:
    top-quantile mean return minus bottom-quantile mean return, equally
    weighted within each quantile.
    """
    factor = (factor_df
              .sort_values(["date", "fsym_primary_listing_id"])
              .set_index(["date", "fsym_primary_listing_id"])["factor_value"])
    merged = al.utils.get_clean_factor_and_forward_returns(
        factor=factor, prices=prices,
        periods=(horizon_days,), quantiles=quantiles, max_loss=max_loss,
    )
    ret_col = f"{horizon_days}D"
    date_index = merged.index.get_level_values(0)
    by_q = (merged
            .groupby([date_index, "factor_quantile"])[ret_col]
            .mean()
            .unstack())
    ls = (by_q[quantiles] - by_q[1]).dropna()
    ls.name = "long_short_return"
    ls.index.name = "date"
    return ls


def build_tearsheet(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    factor_name: str,
    output_html: Path | str,
    benchmark_returns: pd.Series | None = None,
    quantiles: int = 5,
) -> Path:
    """
    Produce a single-page HTML tear sheet for the factor's long-short return.

    Parameters
    ----------
    factor_df : DataFrame (fsym_primary_listing_id, date, factor_value)
    prices    : wide DataFrame (date × fsym_primary_listing_id)
    factor_name : display name, used in the report title
    output_html : destination path
    benchmark_returns : optional benchmark series; if omitted, the report is
                        self-benchmarked.
    """
    ls = long_short_daily_returns(factor_df, prices, quantiles=quantiles)

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    qs.reports.html(
        ls,
        benchmark=benchmark_returns,
        output=str(output_html),
        title=f"{factor_name} — Long/Short Q{quantiles}-Q1 Tear Sheet",
        download_filename=output_html.name,
    )
    return output_html


def tearsheet_summary_stats(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    quantiles: int = 5,
) -> dict:
    """Return a compact dict of key performance stats — useful for a README table."""
    ls = long_short_daily_returns(factor_df, prices, quantiles=quantiles)
    equity = (1.0 + ls).cumprod()  # compound equity curve
    return {
        "n_days":         int(len(ls)),
        "daily_mean_bps": float(ls.mean() * 1e4),
        "ann_return":     float(equity.iloc[-1] ** (252.0 / len(ls)) - 1.0),
        "sharpe":         float(qs.stats.sharpe(ls)),
        "sortino":        float(qs.stats.sortino(ls)),
        "max_drawdown":   float(qs.stats.max_drawdown(equity)),
        "volatility":     float(ls.std() * np.sqrt(252)),
        "skew":           float(ls.skew()),
        "kurtosis":       float(ls.kurtosis()),
    }
