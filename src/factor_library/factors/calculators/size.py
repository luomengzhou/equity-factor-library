"""
Size factor metric: market_cap = price × shares.

Market cap is a level; winsorization on log(market_cap) or raw is optional
per plan (apply on derived where relevant). For consistency we winsorize
the final metric used in ranking.
"""
import numpy as np
import pandas as pd

from factor_library.factors.calculators.base import winsorize_global

METRIC_KEYS = ["market_cap"]


def compute_size_metrics(
    df: pd.DataFrame,
    winsorize: bool = True,
) -> pd.DataFrame:
    """
    Compute market_cap = price × ff_com_shs_out_secs.

    Winsorize on the level (plan: use winsorization on derived; market_cap
    is a level but we apply global winsorize for consistency in ranking).
    """
    if "price" not in df.columns or "ff_com_shs_out_secs" not in df.columns:
        df = df.copy()
        df["market_cap"] = np.nan
        return df

    df = df.copy()
    df["market_cap"] = df["price"] * df["ff_com_shs_out_secs"]
    df["market_cap"] = df["market_cap"].replace([np.inf, -np.inf], np.nan)

    if winsorize and df["market_cap"].notna().sum() >= 5:
        df["market_cap"] = winsorize_global(df["market_cap"])
    return df
