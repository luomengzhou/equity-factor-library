"""
Yield factor metrics (dividend growth, payout ratio).

METRIC_TABLES: dividend_growth (ff_dps), payout_ratio_eps (ff_eps), payout_ratio_dps (ff_dps).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_library.factors.calculators.base import winsorize_global


def compute_dividend_growth(
    df: pd.DataFrame,
    years: int = 5,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Dividend Growth = CAGR(DPS) over N years.

    CAGR = (DPS_t / DPS_{t-N})^(1/N) - 1

    METRIC_TABLES: dividend_growth -> ff_dps (fds.ff_v3_ff_basic_ltm).
    """
    df = df.copy()
    if "ff_dps" not in df.columns:
        df["dividend_growth"] = np.nan
        return df
    if years <= 0:
        raise ValueError("years must be positive")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    # This panel is daily. Use a years-long trading-day lag and compute true
    # CAGR, not a short-horizon pct_change.
    lag_periods = int(years * periods_per_year)
    dps_now = df["ff_dps"]
    dps_lag = df.groupby("fsym_primary_listing_id")["ff_dps"].shift(lag_periods)

    out = pd.Series(np.nan, index=df.index, dtype=float)
    valid = (dps_now > 0) & (dps_lag > 0)
    ratio = dps_now[valid] / dps_lag[valid]
    out.loc[valid] = np.power(ratio, 1.0 / years) - 1.0
    df["dividend_growth"] = out
    df["dividend_growth"] = df["dividend_growth"].replace([np.inf, -np.inf], np.nan)
    return df


def compute_payout_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Payout Ratio = Dividends / Earnings = DPS / EPS.

    METRIC_TABLES: payout_ratio_dps (ff_dps), payout_ratio_eps (ff_eps).
    """
    df = df.copy()
    if "ff_dps" not in df.columns or "ff_eps" not in df.columns:
        df["payout_ratio"] = np.nan
        return df
    # Restrict to economically meaningful cases. Negative/zero earnings make
    # payout ratio hard to interpret and can flip sign mechanically.
    valid = (df["ff_eps"] > 0) & (df["ff_dps"] >= 0)
    num = df["ff_dps"].where(valid)
    den = df["ff_eps"].where(valid)
    df["payout_ratio"] = num / den
    df["payout_ratio"] = df["payout_ratio"].replace([np.inf, -np.inf], np.nan)
    return df


def compute_yield_metrics(
    df: pd.DataFrame,
    winsorize: bool = True,
) -> pd.DataFrame:
    """Compute yield metrics."""
    df = compute_dividend_growth(df)
    df = compute_payout_ratio(df)
    if winsorize:
        for col in ["dividend_growth", "payout_ratio"]:
            if col in df.columns and df[col].notna().any():
                df[col] = winsorize_global(df[col])
    return df
