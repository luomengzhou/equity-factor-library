"""
Volatility factor metrics: historical, downside, idiosyncratic volatility.

Winsorization on derived vol signals (after calculation).
"""
import numpy as np
import pandas as pd

from factor_library.factors.calculators.base import winsorize_global

HIST_VOL_WINDOW = 252
DOWN_VOL_WINDOW = 252
IDIO_VOL_WINDOW = 60
ANNUAL_FACTOR = np.sqrt(252)

VOLATILITY_METRICS = [
    "historical_volatility",
    "downside_volatility",
    "idiosyncratic_volatility",
]


def calculate_volatility_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute historical, downside, idiosyncratic volatility."""
    df = df.sort_values(["fsym_primary_listing_id", "date"]).reset_index(drop=True)
    df["daily_ret"] = df.groupby("fsym_primary_listing_id")["price"].pct_change()

    min_per_hist = max(HIST_VOL_WINDOW // 2, 20)
    df["historical_volatility"] = (
        df.groupby("fsym_primary_listing_id")["daily_ret"]
        .transform(
            lambda s: s.rolling(window=HIST_VOL_WINDOW, min_periods=min_per_hist).std()
            * ANNUAL_FACTOR
        )
    )

    min_per_down = max(DOWN_VOL_WINDOW // 2, 20)
    neg_sq = df["daily_ret"].clip(upper=0) ** 2
    df["downside_volatility"] = (
        neg_sq.groupby(df["fsym_primary_listing_id"])
        .transform(
            lambda s: np.sqrt(
                s.rolling(window=DOWN_VOL_WINDOW, min_periods=min_per_down).mean()
            )
            * ANNUAL_FACTOR
        )
    )

    min_per_idio = max(IDIO_VOL_WINDOW // 2, 20)
    market_ret = df.groupby("date")["daily_ret"].transform("mean")
    residual = df["daily_ret"] - market_ret
    df["idiosyncratic_volatility"] = (
        residual.groupby(df["fsym_primary_listing_id"])
        .transform(
            lambda s: s.rolling(window=IDIO_VOL_WINDOW, min_periods=min_per_idio).std()
            * ANNUAL_FACTOR
        )
    )
    return df


def compute_volatility_metrics(
    df: pd.DataFrame,
    winsorize: bool = True,
) -> pd.DataFrame:
    """Compute vol metrics and optionally winsorize (on derived signals)."""
    df = calculate_volatility_metrics(df)
    if winsorize:
        for col in VOLATILITY_METRICS:
            if col in df.columns:
                df[col] = winsorize_global(df[col])
    return df
