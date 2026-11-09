"""
Liquidity factor metrics.

ADV = Average Daily Dollar Volume = mean(Price * Volume)
Turnover = Volume / Shares Outstanding
Amihud Illiquidity = mean(|R_t| / DollarVol_t)
Pastor-Stambaugh proxy = rolling exposure to aggregate liquidity shocks

Uses daily panel columns loaded in data_loader:
- price
- volume (from the daily price table, see data_loader.load_volume)
- high_price / low_price (for spread proxy)
- ff_com_shs_out_secs
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import warnings

try:
    import statsmodels.api as sm
    from statsmodels.regression.rolling import RollingOLS
except Exception:  # pragma: no cover - defensive import guard
    sm = None
    RollingOLS = None


def _id_col(df: pd.DataFrame) -> str:
    """Pick security ID column."""
    return "fsym_primary_listing_id" if "fsym_primary_listing_id" in df.columns else "fsym_id"


def compute_adv_dollar(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    ADV = Average Daily Dollar Volume = mean(Price * Volume).

    Requires: price, volume.
    """
    df = df.copy()
    if "price" in df.columns and "volume" in df.columns:
        sec = _id_col(df)
        df["dollar_volume"] = df["price"] * df["volume"]
        df["adv_dollar"] = (
            df.groupby(sec)["dollar_volume"]
            .transform(lambda s: s.rolling(window, min_periods=5).mean())
        )
    else:
        df["adv_dollar"] = np.nan
    return df


def compute_turnover(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Turnover = Volume / Shares Outstanding (avg over period).

    Requires: volume, ff_com_shs_out_secs.
    """
    df = df.copy()
    if "volume" in df.columns and "ff_com_shs_out_secs" in df.columns:
        sec = _id_col(df)
        raw = df["volume"] / df["ff_com_shs_out_secs"].replace(0, np.nan)
        df["turnover"] = (
            raw.groupby(df[sec]).transform(lambda s: s.rolling(window, min_periods=5).mean())
        )
    else:
        df["turnover"] = np.nan
    return df


def compute_amihud(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Amihud Illiquidity = mean(|R_t| / DollarVol_t).
    Higher = more illiquid.

    Requires: returns and dollar volume.
    """
    df = df.copy()
    sec = _id_col(df)
    if "price" not in df.columns or "volume" not in df.columns:
        df["amihud_illiquidity"] = np.nan
        return df
    ret = df.groupby(sec)["price"].pct_change()
    dollar_vol = (df["price"] * df["volume"]).replace(0, np.nan)
    raw = ret.abs() / dollar_vol
    df["amihud_illiquidity"] = (
        raw.groupby(df[sec]).transform(lambda s: s.rolling(window, min_periods=5).mean())
    )
    df["amihud_illiquidity"] = df["amihud_illiquidity"].replace([np.inf, -np.inf], np.nan)
    return df


def compute_bid_ask_spread_proxy(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Bid-ask spread proxy using daily high-low range:
    spread_proxy = (high - low) / close, then rolling average.
    """
    df = df.copy()
    sec = _id_col(df)
    if not {"high_price", "low_price", "price"}.issubset(df.columns):
        df["bid_ask_spread"] = np.nan
        return df
    raw = (df["high_price"] - df["low_price"]) / df["price"].replace(0, np.nan)
    df["bid_ask_spread"] = (
        raw.groupby(df[sec]).transform(lambda s: s.rolling(window, min_periods=5).mean())
    )
    df["bid_ask_spread"] = df["bid_ask_spread"].replace([np.inf, -np.inf], np.nan)
    return df


def _standardize_series(s: pd.Series) -> pd.Series:
    """Global z-score with NaN-safe fallback."""
    s = pd.to_numeric(s, errors="coerce")
    sigma = float(s.std(skipna=True, ddof=0))
    if (not np.isfinite(sigma)) or sigma <= 0:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return ((s - float(s.mean(skipna=True))) / sigma).astype(float)


def compute_pastor_stambaugh_proxy(
    df: pd.DataFrame,
    window: int = 252,
    min_periods: int = 126,
) -> pd.DataFrame:
    """
    Pastor-Stambaugh style liquidity-risk exposure proxy.

    Steps:
    1) Cross-sectional median Amihud per date => L_t
    2) Liquidity shock u_t = diff(log(L_t)), then z-scored
    3) Rolling OLS per stock:
         r_i,t = a + b*r_m,t + c*u_t + e_t
       where c is the exposure proxy.
    """
    df = df.copy()
    sec = _id_col(df)
    if "price" not in df.columns:
        df["pastor_stambaugh"] = np.nan
        return df

    if "amihud_illiquidity" not in df.columns:
        df = compute_amihud(df)

    liq_level = (
        df.groupby("date")["amihud_illiquidity"]
        .median()
        .replace([np.inf, -np.inf], np.nan)
    )
    liq_level = liq_level.where(liq_level > 0, np.nan)
    liq_shock = _standardize_series(np.log(liq_level).diff())
    liq_shock.name = "liq_shock"

    ret_col = "_ret_1d"
    df[ret_col] = df.groupby(sec)["price"].pct_change()
    market_ret = df.groupby("date")[ret_col].mean().rename("market_return")
    df = df.merge(
        pd.concat([market_ret, liq_shock], axis=1),
        left_on="date",
        right_index=True,
        how="left",
    )

    df["pastor_stambaugh"] = np.nan
    if RollingOLS is None or sm is None:
        warnings.warn(
            "statsmodels not available; pastor_stambaugh set to NaN.",
            UserWarning,
            stacklevel=2,
        )
        return df.drop(columns=[ret_col, "market_return", "liq_shock"], errors="ignore")

    for _, idx in df.groupby(sec, sort=False).groups.items():
        sub = (
            df.loc[idx, ["date", ret_col, "market_return", "liq_shock"]]
            .sort_values("date")
            .dropna(subset=[ret_col, "market_return", "liq_shock"])
        )
        if len(sub) < min_periods:
            continue
        y = sub[ret_col].astype(float)
        x = sm.add_constant(
            sub[["market_return", "liq_shock"]].astype(float),
            has_constant="add",
        )
        try:
            model = RollingOLS(y, x, window=window, min_nobs=min_periods)
            fit = model.fit(params_only=True)
            c = fit.params["liq_shock"].astype(float)
        except Exception:
            continue
        if c.notna().any():
            df.loc[c.index, "pastor_stambaugh"] = c.values

    return df.drop(columns=[ret_col, "market_return", "liq_shock"], errors="ignore")


def compute_liquidity_metrics(
    df: pd.DataFrame,
    *,
    include_bid_ask_proxy: bool = False,
    include_pastor_stambaugh: bool = False,
) -> pd.DataFrame:
    """Compute liquidity metrics from panel volume + price."""
    df = df.sort_values([_id_col(df), "date"]).reset_index(drop=True)
    df = compute_adv_dollar(df)
    df = compute_turnover(df)
    df = compute_amihud(df)
    if include_bid_ask_proxy:
        df = compute_bid_ask_spread_proxy(df)
    if include_pastor_stambaugh:
        df = compute_pastor_stambaugh_proxy(df)
    return df
