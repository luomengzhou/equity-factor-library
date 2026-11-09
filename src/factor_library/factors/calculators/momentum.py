"""
Momentum factor metrics aligned with preliminary_factor_library.xlsx (Momentum rows only):

- mom_12m_1m: 12M return skipping the most recent month
- mom_6m_1m: 6M return skipping the most recent month
- mom_1m_reversal: −1 × past one-month return (short-horizon reversal)
- mom_52w_high: price / trailing 52-week high

Winsorization is applied on derived series, not on raw prices.
"""
import numpy as np
import pandas as pd

from factor_library.factors.calculators.base import winsorize_global

# 1M = 21d, 6M = 147d, 12M = 273d; 52W = 252d
MOMENTUM_METRICS = ["mom_12m_1m", "mom_6m_1m", "mom_1m_reversal", "mom_52w_high"]


def calculate_momentum_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum matrix: 12M-1M, 6M-1M, 1M reversal, 52-week high proximity.

    Convention: 21 trading days = 1 month; 6M/12M legs omit the most recent month.
    """
    df = df.sort_values(["fsym_primary_listing_id", "date"]).reset_index(drop=True)
    g = df.groupby("fsym_primary_listing_id")["price"]

    df["mom_12m_1m"] = g.shift(21) / g.shift(273) - 1
    df["mom_6m_1m"] = g.shift(21) / g.shift(147) - 1
    df["mom_1m_reversal"] = -g.pct_change(21)

    rolling_52w = g.transform(lambda s: s.rolling(252, min_periods=21).max())
    df["mom_52w_high"] = df["price"] / rolling_52w
    df["mom_52w_high"] = df["mom_52w_high"].replace([np.inf, -np.inf], np.nan)

    for c in ["mom_6m_1m", "mom_12m_1m", "mom_1m_reversal", "mom_52w_high"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df


def compute_momentum_metrics(
    df: pd.DataFrame,
    winsorize: bool = True,
) -> pd.DataFrame:
    """Compute the four library-aligned momentum metrics."""
    df = calculate_momentum_matrix(df)
    if winsorize:
        for col in MOMENTUM_METRICS:
            if col in df.columns:
                df[col] = winsorize_global(df[col])
    return df
