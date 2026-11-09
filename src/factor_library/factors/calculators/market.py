"""
Market factor metrics (beta, market excess return).

Uses panel price to derive daily stock return and market return proxy.
Risk-free: bundled ``app/data/DTB3.xlsx`` (FRED DTB3, % p.a.); daily rate ≈
(annual_pct/100)/252. If the file is missing, falls back to ``rf_daily=0`` with
a warning (``market_excess_return`` ≈ ``market_ret``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from factor_library.config import RF_DTB3_XLSX_PATH


def load_dtb3_rf_daily(
    path: Path | str | None = None,
    *,
    sheet_name: str = "Daily",
) -> pd.Series:
    """
    Load FRED DTB3 Excel → calendar-date → daily risk-free rate (decimal).

    ``DTB3`` column is **percent per annum** (e.g. 5.25 → 5.25%). We use a
    simple working-day split: ``rf_daily = (pct/100) / 252`` to align with
    common 252-day equity conventions (same order of magnitude as continuous
    approximation for low rates).

    Returns
    -------
    Series indexed by normalized ``date`` (midnight UTC-naive).
    Empty series if file missing or sheet invalid.
    """
    p = Path(path) if path is not None else Path(RF_DTB3_XLSX_PATH)
    if not p.is_file():
        return pd.Series(dtype=float)

    try:
        raw = pd.read_excel(p, sheet_name=sheet_name, usecols=["observation_date", "DTB3"])
    except (ValueError, KeyError, OSError):
        return pd.Series(dtype=float)

    raw = raw.dropna(subset=["DTB3"])
    if raw.empty:
        return pd.Series(dtype=float)

    d = pd.to_datetime(raw["observation_date"], errors="coerce").dt.normalize()
    pct = pd.to_numeric(raw["DTB3"], errors="coerce").clip(lower=0.0)
    ok = d.notna() & pct.notna()
    d, pct = d[ok], pct[ok]
    rf_annual_dec = pct / 100.0
    rf_daily = rf_annual_dec / 252.0
    s = pd.Series(rf_daily.values, index=d)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def _rolling_beta(
    ret: pd.Series,
    mkt: pd.Series,
    window: int = 252,
    min_periods: int = 63,
) -> pd.Series:
    """Rolling CAPM beta = Cov(R_i, R_m) / Var(R_m)."""
    cov = ret.rolling(window=window, min_periods=min_periods).cov(mkt)
    var = mkt.rolling(window=window, min_periods=min_periods).var()
    beta = cov / var.replace(0, np.nan)
    return beta.replace([np.inf, -np.inf], np.nan)


def compute_beta(
    df: pd.DataFrame,
    window: int = 252,
    min_periods: int = 63,
    rf_xlsx_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Beta = Cov(R_i^e, R_m^e) / Var(R_m^e) with daily excess returns vs DTB3 rf.

    ``R_m`` is cross-sectional mean of ``ret_1d``; ``R_m^e = R_m - rf_daily``;
    ``R_i^e = ret_1d - rf_daily``. If DTB3 file is absent, rf_daily is 0.
    """
    import warnings

    df = df.copy()
    if "price" not in df.columns:
        df["beta"] = np.nan
        df["market_ret"] = np.nan
        df["market_excess_return"] = np.nan
        df["rf_daily"] = np.nan
        return df

    df = df.sort_values(["fsym_primary_listing_id", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["ret_1d"] = df.groupby("fsym_primary_listing_id")["price"].pct_change()
    df["market_ret"] = df.groupby("date")["ret_1d"].transform("mean")

    rf_series = load_dtb3_rf_daily(rf_xlsx_path)
    dn = df["date"].dt.normalize()
    if rf_series.empty:
        warnings.warn(
            "DTB3 risk-free file missing or empty; using rf_daily=0 (market_excess_return = market_ret). "
            "Set config.RF_DTB3_XLSX_PATH or pass rf_xlsx_path to compute_beta.",
            UserWarning,
            stacklevel=2,
        )
        df["rf_daily"] = 0.0
    else:
        df["rf_daily"] = dn.map(rf_series)
        n_miss = int(df["rf_daily"].isna().sum())
        if n_miss:
            # FRED vs trading calendar: sort by time, ffill/bfill along the panel
            df = df.sort_values(["date", "fsym_primary_listing_id"])
            df["rf_daily"] = df["rf_daily"].ffill().bfill()
            df = df.sort_values(["fsym_primary_listing_id", "date"]).reset_index(drop=True)
            if df["rf_daily"].isna().any():
                warnings.warn(
                    "Some rows still lack rf_daily after ffill/bfill; filled with 0.",
                    UserWarning,
                    stacklevel=2,
                )
                df["rf_daily"] = df["rf_daily"].fillna(0.0)

    df["market_excess_return"] = df["market_ret"] - df["rf_daily"]
    df["ret_excess"] = df["ret_1d"] - df["rf_daily"]

    beta_parts = []
    for _, g in df.groupby("fsym_primary_listing_id", sort=False):
        b = _rolling_beta(
            g["ret_excess"],
            g["market_excess_return"],
            window=window,
            min_periods=min_periods,
        )
        beta_parts.append(b)
    if beta_parts:
        df["beta"] = pd.concat(beta_parts).sort_index()
    else:
        df["beta"] = np.nan
    return df


def compute_market_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market metrics from price panel (+ DTB3 rf when available)."""
    return compute_beta(df)


# ---------------------------------------------------------------------------
# Low Beta / BAB — merged from low_beta.py
# ---------------------------------------------------------------------------


def compute_low_beta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Low beta = raw beta from CAPM. Factor favors low-beta stocks.
    Requires: beta column (from compute_beta).
    """
    df = df.copy()
    df["low_beta"] = df["beta"] if "beta" in df.columns else np.nan
    return df


def compute_bab_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """BAB proxy: -beta. Higher factor value = lower beta = long side."""
    df = df.copy()
    df["bab_proxy"] = -df["beta"] if "beta" in df.columns else np.nan
    return df


def compute_low_beta_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute low beta / BAB metrics."""
    df = compute_low_beta(df)
    df = compute_bab_proxy(df)
    return df
