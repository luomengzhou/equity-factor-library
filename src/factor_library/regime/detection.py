from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


RegimeLabel = Literal["trend", "chop"]


@dataclass(frozen=True)
class RegimeConfig:
    # Rolling windows over daily market returns
    mean_window: int = 60
    vol_window: int = 20
    # |annualized rolling_mean / rolling_std| >= threshold => trend
    sharpe_threshold: float = 1.5
    # market return aggregation across stocks
    market_ret_agg: Literal["median", "mean"] = "median"


@dataclass(frozen=True)
class VolBullBearConfig:
    # Volatility regime: rolling std of market returns
    vol_window: int = 20
    vol_quantile: float = 0.5  # <= quantile => low_vol else high_vol

    # Bull/bear: rolling cumulative return of market returns
    bullbear_window: int = 126  # ~6 months
    bullbear_quantile: float = 0.5  # <= quantile => bear else bull


def compute_market_return_from_prices(
    prices: pd.DataFrame,
    *,
    id_col: str = "fsym_primary_listing_id",
    date_col: str = "date",
    price_col: str = "price",
    agg: Literal["median", "mean"] = "median",
) -> pd.DataFrame:
    """
    Build a simple daily market return series from per-stock prices.

    This intentionally avoids external benchmarks; it aggregates cross-sectional
    returns across the available universe each day.
    """
    df = prices[[id_col, date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([id_col, date_col])
    df["ret_1d"] = df.groupby(id_col, sort=False)[price_col].pct_change()

    g = df.groupby(date_col, sort=True)["ret_1d"]
    if agg == "mean":
        mkt_ret = g.mean()
    else:
        mkt_ret = g.median()

    out = mkt_ret.rename("mkt_ret_1d").reset_index()
    return out


def label_regime_from_market_return(
    market_ret: pd.DataFrame,
    *,
    date_col: str = "date",
    ret_col: str = "mkt_ret_1d",
    cfg: RegimeConfig | None = None,
) -> pd.DataFrame:
    """
    Label each date as trend/chop using an annualized rolling Sharpe-like statistic.
    """
    if cfg is None:
        cfg = RegimeConfig()

    df = market_ret[[date_col, ret_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    roll_mean = df[ret_col].rolling(cfg.mean_window, min_periods=max(5, cfg.vol_window)).mean()
    roll_std = df[ret_col].rolling(cfg.vol_window, min_periods=max(5, cfg.vol_window)).std(ddof=1)
    score = (roll_mean / roll_std) * np.sqrt(252)

    df["regime_score"] = score.replace([np.inf, -np.inf], np.nan)
    df["regime"] = np.where(df["regime_score"].abs() >= cfg.sharpe_threshold, "trend", "chop")
    df["regime"] = df["regime"].astype("string")
    return df[[date_col, "regime", "regime_score"]]


def label_volatility_regime(
    market_ret: pd.DataFrame,
    *,
    date_col: str = "date",
    ret_col: str = "mkt_ret_1d",
    cfg: VolBullBearConfig | None = None,
) -> pd.DataFrame:
    """
    Label dates as low_vol / high_vol based on rolling volatility level.
    Uses an expanding quantile threshold on the rolling vol series to avoid peeking.
    """
    if cfg is None:
        cfg = VolBullBearConfig()
    df = market_ret[[date_col, ret_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    vol = df[ret_col].rolling(cfg.vol_window, min_periods=max(5, cfg.vol_window)).std(ddof=1)
    # expanding quantile threshold (uses history up to t)
    thr = vol.expanding(min_periods=max(20, cfg.vol_window)).quantile(cfg.vol_quantile)
    df["mkt_vol"] = vol
    df["vol_regime"] = np.where(df["mkt_vol"] <= thr, "low_vol", "high_vol")
    df["vol_regime"] = df["vol_regime"].astype("string")
    return df[[date_col, "vol_regime", "mkt_vol"]]


def label_bull_bear_regime(
    market_ret: pd.DataFrame,
    *,
    date_col: str = "date",
    ret_col: str = "mkt_ret_1d",
    cfg: VolBullBearConfig | None = None,
) -> pd.DataFrame:
    """
    Label dates as bull / bear using rolling cumulative market return.
    Threshold is an expanding quantile to avoid peeking.
    """
    if cfg is None:
        cfg = VolBullBearConfig()
    df = market_ret[[date_col, ret_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    roll_cum = (1.0 + df[ret_col]).rolling(cfg.bullbear_window, min_periods=max(20, cfg.bullbear_window)).apply(
        lambda x: float(np.prod(x) - 1.0),
        raw=True,
    )
    thr = roll_cum.expanding(min_periods=max(60, cfg.bullbear_window)).quantile(cfg.bullbear_quantile)
    df["mkt_roll_ret"] = roll_cum
    df["bullbear"] = np.where(df["mkt_roll_ret"] <= thr, "bear", "bull")
    df["bullbear"] = df["bullbear"].astype("string")
    return df[[date_col, "bullbear", "mkt_roll_ret"]]


def compute_mktcap_bucket(
    panel: pd.DataFrame,
    *,
    id_col: str = "fsym_primary_listing_id",
    date_col: str = "date",
    mktcap_col: str = "market_cap",
    n_buckets: int = 3,
) -> pd.DataFrame:
    """
    Per-date market cap buckets. Requires a `market_cap` column in the input panel.
    Returns columns: id_col, date_col, mktcap_bucket (1..n_buckets).
    """
    if mktcap_col not in panel.columns:
        raise ValueError(f"market cap column missing: {mktcap_col}")

    df = panel[[id_col, date_col, mktcap_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Fast per-date quantile-like bucketing using rank and group size:
    # bucket = floor((rank-1) * n_buckets / n_in_group) + 1
    s = df[mktcap_col]
    r = s.groupby(df[date_col], sort=False).rank(method="average", na_option="keep")
    n = s.groupby(df[date_col], sort=False).transform("count")
    b = np.floor((r - 1) * n_buckets / n) + 1
    b = b.where(s.notna(), pd.NA)
    df["mktcap_bucket"] = b.astype("Int64")
    return df[[id_col, date_col, "mktcap_bucket"]]


def compute_quantile_bucket_from_wide(
    wide: pd.DataFrame,
    *,
    bucket_col: str,
    out_col: str,
    id_col: str = "fsym_primary_listing_id",
    date_col: str = "date",
    n_buckets: int = 3,
) -> pd.DataFrame:
    """
    Per-date quantile buckets computed from a numeric column in a wide panel.
    Returns id+date level buckets (Int64 1..n_buckets).
    """
    if bucket_col not in wide.columns:
        raise ValueError(f"bucket source column missing: {bucket_col}")
    df = wide[[id_col, date_col, bucket_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])

    s = df[bucket_col]
    r = s.groupby(df[date_col], sort=False).rank(method="average", na_option="keep")
    n = s.groupby(df[date_col], sort=False).transform("count")
    b = np.floor((r - 1) * n_buckets / n) + 1
    b = b.where(s.notna(), pd.NA)
    df[out_col] = b.astype("Int64")
    return df[[id_col, date_col, out_col]]

