"""
Signal-weighted long/short — mentor's Stage-2 suggestion.

The equal-weight Q5-Q1 book used in ``reports.tearsheets.long_short_daily_returns``
gives every top-quintile name the same weight. Mentor's comment:

    "another thing that you could do for example it would be like to try to
     assign a weight depending on that momentum signal so like if the
     momentum signal is higher then you allocate more weight"

This module implements that: each stock's weight is proportional to how far
its factor_value sits from 0.5 (the cross-sectional median). Long legs get
positive weights, short legs get negative weights, and the weights are
normalised so the gross (sum of absolute weights) equals 1 on every day.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="alphalens")
import alphalens as al  # noqa: E402


def signal_weighted_daily_returns(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    max_loss: float = 0.50,
) -> pd.Series:
    """
    Daily signal-weighted long/short return. Per-stock weight is
        w_i,t = (factor_value_i,t - 0.5) / Σ_j |factor_value_j,t - 0.5|
    so the sum of absolute weights is 1 each day. Names at the median
    get zero weight; names at the extremes get the largest weight.

    Parameters
    ----------
    factor_df : DataFrame with columns [fsym_primary_listing_id, date, factor_value]
    prices    : wide DataFrame (date × fsym_primary_listing_id)
    """
    factor = (factor_df
              .sort_values(["date", "fsym_primary_listing_id"])
              .set_index(["date", "fsym_primary_listing_id"])["factor_value"])
    merged = al.utils.get_clean_factor_and_forward_returns(
        factor=factor, prices=prices,
        periods=(1,), quantiles=5, max_loss=max_loss,
    )
    df = merged[["1D", "factor"]].reset_index()
    df.columns = ["date", "asset", "ret", "factor_value"]

    # Centre at the cross-sectional median each day; normalise to L1 = 1.
    df["signal"] = df["factor_value"] - 0.5
    l1 = df.groupby("date", sort=False)["signal"].transform(lambda s: s.abs().sum())
    df["weight"] = df["signal"] / l1
    df["contrib"] = df["weight"] * df["ret"]

    daily = df.groupby("date", sort=True)["contrib"].sum()
    daily.name = "signal_weighted_return"
    return daily


def compare_weighting(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    factor_name: str,
) -> pd.DataFrame:
    """
    Side-by-side summary of equal-weight Q5-Q1 vs signal-weighted L/S for
    a single factor. Returns a two-row DataFrame.
    """
    import empyrical as emp
    from factor_library.reports.tearsheets import long_short_daily_returns

    def _stats(ret: pd.Series, label: str) -> dict:
        equity = (1 + ret).cumprod()
        log_eq = np.log1p(ret).cumsum()
        max_dd = float((np.exp(log_eq - log_eq.cummax()) - 1).min())
        ann = float(emp.annual_return(ret))
        return {
            "factor":       factor_name,
            "weighting":    label,
            "n_days":       int(len(ret)),
            "ann_return":   ann,
            "ann_vol":      float(ret.std(ddof=1) * np.sqrt(252)),
            "sharpe":       float(emp.sharpe_ratio(ret, annualization=252)),
            "sortino":      float(emp.sortino_ratio(ret, annualization=252)),
            "max_drawdown": max_dd,
            "calmar":       float(ann / abs(max_dd)) if max_dd < 0 else np.nan,
            "skew":         float(ret.skew()),
            "kurtosis":     float(ret.kurtosis()),
        }

    eq = long_short_daily_returns(factor_df, prices)
    sw = signal_weighted_daily_returns(factor_df, prices)
    return pd.DataFrame([
        _stats(eq, "equal-weight Q5-Q1"),
        _stats(sw, "signal-weighted"),
    ])
