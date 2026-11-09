"""
Quality factor metrics.

METRIC_TABLES: roe, roa, gross_profitability_*, profit_margin_*, leverage_*,
earnings_variability, accruals_*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_library.factors.calculators.base import winsorize_global


def compute_roe(df: pd.DataFrame) -> pd.DataFrame:
    """
    ROE = Net Income(TTM) / Avg Equity.

    METRIC_TABLES: roe -> ff_roe / fe_roe.
    """
    df = df.copy()
    if "ff_roe" in df.columns:
        df["roe"] = df["ff_roe"]
    elif "fe_roe" in df.columns:
        df["roe"] = df["fe_roe"]
    else:
        df["roe"] = np.nan
    return df


def compute_roa(df: pd.DataFrame) -> pd.DataFrame:
    """
    ROA = Net Income(TTM) / Avg Assets.

    METRIC_TABLES: roa -> ff_roa.
    """
    df = df.copy()
    if "ff_roa" in df.columns:
        df["roa"] = df["ff_roa"]
    else:
        df["roa"] = np.nan
    return df


def compute_gross_profitability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gross Profitability = Gross Profit / Total Assets = (Sales - COGS) / Assets.

    METRIC_TABLES: gross_profitability_sales (ff_sales), gross_profitability_cogs (ff_cogs),
    gross_profitability_assets (ff_assets).
    """
    df = df.copy()
    need = ["ff_sales", "ff_cogs", "ff_assets"]
    if not all(c in df.columns for c in need):
        df["gross_profitability"] = np.nan
        return df
    gp = df["ff_sales"] - df["ff_cogs"]
    df["gross_profitability"] = gp / df["ff_assets"].replace(0, np.nan)
    df["gross_profitability"] = df["gross_profitability"].replace([np.inf, -np.inf], np.nan)
    return df


def compute_profit_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Profit Margin = Net Income / Revenue.

    METRIC_TABLES: profit_margin_sales (ff_sales), profit_margin_net_inc (ff_net_inc).
    """
    df = df.copy()
    if "ff_net_inc" not in df.columns or "ff_sales" not in df.columns:
        df["profit_margin"] = np.nan
        return df
    df["profit_margin"] = df["ff_net_inc"] / df["ff_sales"].replace(0, np.nan)
    df["profit_margin"] = df["profit_margin"].replace([np.inf, -np.inf], np.nan)
    return df


def compute_leverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leverage = Total Debt / Equity (D/E).
    Lower is better for quality; factor direction uses -1.

    METRIC_TABLES: leverage_debt (ff_debt), leverage_eq (ff_shldrs_eq).
    """
    df = df.copy()
    if "ff_debt" not in df.columns or "ff_shldrs_eq" not in df.columns:
        df["leverage"] = np.nan
        return df
    eq = df["ff_shldrs_eq"].replace(0, np.nan)
    df["leverage"] = df["ff_debt"] / eq
    df["leverage"] = df["leverage"].replace([np.inf, -np.inf], np.nan)
    return df


def compute_earnings_variability(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Earnings Variability = StdDev(EPS or NI) over rolling window.
    Lower = more stable = better quality.

    METRIC_TABLES: earnings_variability -> ff_eps.
    """
    df = df.copy()
    if "ff_eps" not in df.columns:
        df["earnings_variability"] = np.nan
        return df
    df["earnings_variability"] = (
        df.groupby("fsym_primary_listing_id")["ff_eps"]
        .transform(lambda s: s.rolling(window, min_periods=2).std())
    )
    return df


def compute_accruals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accruals = (Net Income - Free Cash Flow) / Total Assets.

    METRIC_TABLES: accruals_net_inc (ff_net_inc), accruals_cf (ff_free_ps_cf),
    accruals_assets (ff_assets).
    """
    df = df.copy()
    need = ["ff_net_inc", "ff_free_ps_cf", "ff_assets"]
    if not all(c in df.columns for c in need):
        df["accruals"] = np.nan
        return df
    acc = df["ff_net_inc"] - df["ff_free_ps_cf"]
    df["accruals"] = acc / df["ff_assets"].replace(0, np.nan)
    df["accruals"] = df["accruals"].replace([np.inf, -np.inf], np.nan)
    return df


def compute_piotroski_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Piotroski-style proxy score from available quality signals.
    NaN-aware: each check contributes only where the underlying metric has actual data.
    Rows with no valid quality data → NaN (avoids artificial 0-score clustering).
    """
    df = df.copy()
    parts = []

    def _check(series: pd.Series, cond: pd.Series) -> pd.Series:
        """Return 1/0 where series is not NaN, NaN where series is NaN."""
        out = cond.astype(float)
        out[series.isna()] = np.nan
        return out

    if "roa" in df.columns and df["roa"].notna().any():
        parts.append(_check(df["roa"], df["roa"] > 0))
    if "roe" in df.columns and df["roe"].notna().any():
        parts.append(_check(df["roe"], df["roe"] > 0))
    if "profit_margin" in df.columns and df["profit_margin"].notna().any():
        parts.append(_check(df["profit_margin"], df["profit_margin"] > 0))
    if "leverage" in df.columns and df["leverage"].notna().any():
        lev_prev = df.groupby("fsym_primary_listing_id")["leverage"].shift(1)
        cond = df["leverage"] < lev_prev
        mask = df["leverage"].notna() & lev_prev.notna()
        s = cond.astype(float)
        s[~mask] = np.nan
        parts.append(s)
    if "accruals" in df.columns and df["accruals"].notna().any():
        parts.append(_check(df["accruals"], df["accruals"] < 0))
    if "gross_profitability" in df.columns and df["gross_profitability"].notna().any():
        parts.append(_check(df["gross_profitability"], df["gross_profitability"] > 0))

    if not parts:
        df["piotroski_fscore"] = np.nan
    else:
        stacked = pd.concat(parts, axis=1)
        n_valid = stacked.notna().sum(axis=1)
        df["piotroski_fscore"] = stacked.sum(axis=1, skipna=True)
        df.loc[n_valid == 0, "piotroski_fscore"] = np.nan
    return df


def compute_quality_metrics(
    df: pd.DataFrame,
    winsorize: bool = True,
) -> pd.DataFrame:
    """Compute all quality metrics."""
    df = compute_roe(df)
    df = compute_roa(df)
    df = compute_gross_profitability(df)
    df = compute_profit_margin(df)
    df = compute_leverage(df)
    df = compute_earnings_variability(df)
    df = compute_accruals(df)
    df = compute_piotroski_proxy(df)
    if winsorize:
        for col in [
            "roe",
            "roa",
            "gross_profitability",
            "profit_margin",
            "leverage",
            "earnings_variability",
            "accruals",
        ]:
            if col in df.columns and df[col].notna().any():
                df[col] = winsorize_global(df[col])
    return df


# ---------------------------------------------------------------------------
# Profitability (FF RMW) — merged from profitability.py
# ---------------------------------------------------------------------------

def compute_rmw_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    RMW proxy: Operating Profitability = EBIT_oper / Shareholders' Equity.

    Uses ff_ebit_oper directly or reconstructs from ff_ebit_oper_ps * shares.
    """
    df = df.copy()
    if "ff_ebit_oper" in df.columns and "ff_shldrs_eq" in df.columns:
        df["rmw_proxy"] = df["ff_ebit_oper"] / df["ff_shldrs_eq"].replace(0, np.nan)
    elif "ff_ebit_oper_ps" in df.columns and "ff_com_shs_out_secs" in df.columns and "ff_shldrs_eq" in df.columns:
        ebit = df["ff_ebit_oper_ps"] * df["ff_com_shs_out_secs"].replace(0, np.nan)
        df["rmw_proxy"] = ebit / df["ff_shldrs_eq"].replace(0, np.nan)
    else:
        df["rmw_proxy"] = np.nan
    df["rmw_proxy"] = df["rmw_proxy"].replace([np.inf, -np.inf], np.nan)
    return df


def compute_profitability_metrics(
    df: pd.DataFrame,
    winsorize: bool = True,
) -> pd.DataFrame:
    """Compute profitability (RMW) metrics."""
    df = compute_rmw_proxy(df)
    if winsorize and "rmw_proxy" in df.columns and df["rmw_proxy"].notna().any():
        df["rmw_proxy"] = winsorize_global(df["rmw_proxy"])
    return df
