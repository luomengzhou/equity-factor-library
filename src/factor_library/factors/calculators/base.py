"""
Base utilities: winsorize, percentile_rank, zscore, composite building.

Winsorization is applied on derived signals (e.g. growth rates) after calculation,
not on raw level data. Applied globally (across all companies).
"""
from typing import Literal

import numpy as np
import pandas as pd


def winsorize_global(
    series: pd.Series,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.Series:
    """
    Winsorize at global quantiles (across all firms, all dates).

    Apply on derived signals (e.g. YoY growth, asset_growth_rate) AFTER
    calculation, not on raw level data.

    Parameters
    ----------
    series : pd.Series
        The derived signal to winsorize
    lower_q, upper_q : float
        Quantile bounds (default 1%-99%)

    Returns
    -------
    pd.Series
        Winsorized values
    """
    valid = series.dropna()
    if len(valid) < 5:
        return series
    lo = valid.quantile(lower_q)
    hi = valid.quantile(upper_q)
    return series.clip(lower=lo, upper=hi)


def percentile_rank(series: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank (0-1) for a single date slice."""
    return series.rank(pct=True, method="average")


def cross_sectional_zscore(series: pd.Series, clip: float = 5.0) -> pd.Series:
    """
    Cross-sectional Z-score for a single date slice. Optional clip to [-clip, clip].
    """
    valid = series.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=series.index)
    std = valid.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=series.index)
    out = (series - valid.mean()) / std
    if clip > 0:
        out = out.clip(-clip, clip)
    return out


def apply_rank(
    df: pd.DataFrame,
    col: str,
    method: Literal["percentile", "zscore"] = "percentile",
) -> pd.Series:
    """
    Apply cross-sectional rank by date.

    Parameters
    ----------
    df : DataFrame with 'date' column
    col : column name to rank
    method : 'percentile' or 'zscore'

    Returns
    -------
    pd.Series
        Ranked values (same index as df)
    """
    if method == "percentile":
        return df.groupby("date")[col].transform(percentile_rank)
    else:
        return df.groupby("date")[col].transform(
            lambda s: cross_sectional_zscore(s, clip=5.0)
        )


def build_composite(
    df: pd.DataFrame,
    rank_cols: list[str],
    method: Literal["equal_weight", "inverse_volatility"] = "equal_weight",
    min_count: int = 2,
) -> pd.Series:
    """
    Build factor composite from ranked metric columns.

    Parameters
    ----------
    df : DataFrame containing rank_cols
    rank_cols : list of column names (already ranked: percentile or zscore)
    method : 'equal_weight' or 'inverse_volatility'
    min_count : minimum non-NaN ranked metrics required; rows below threshold → NaN.
                Prevents spike artifacts from sparse data. Set to 1 for single-metric factors.

    Returns
    -------
    pd.Series
        Composite factor value
    """
    if not rank_cols:
        return pd.Series(np.nan, index=df.index)
    available = [c for c in rank_cols if c in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index)

    # Enforce minimum non-NaN count to avoid spike from sparse fundamental data
    actual_min = min(min_count, len(available))
    non_nan_count = df[available].notna().sum(axis=1)

    if method == "equal_weight":
        raw = df[available].mean(axis=1, skipna=True)
        raw[non_nan_count < actual_min] = np.nan
        return raw

    # inverse_volatility: weight by 1/std of each rank column (full sample)
    vols = df[available].std(skipna=True, ddof=0).replace(0, np.nan)
    inv_vol = 1.0 / vols
    w = inv_vol / inv_vol.sum(skipna=True)
    weights = {c: float(w[c]) for c in available if pd.notna(w[c])}
    if not weights:
        raw = df[available].mean(axis=1, skipna=True)
        raw[non_nan_count < actual_min] = np.nan
        return raw
    x = df[available].to_numpy(dtype=float)
    w_arr = np.array([weights.get(c, np.nan) for c in available], dtype=float)
    valid = ~np.isnan(x)
    numer = np.nansum(x * w_arr, axis=1)
    denom = np.nansum(np.where(valid, w_arr, np.nan), axis=1)
    out = np.full(len(df), np.nan, dtype=float)
    np.divide(numer, denom, out=out, where=denom > 0)
    result = pd.Series(out, index=df.index)
    result[non_nan_count < actual_min] = np.nan
    return result
