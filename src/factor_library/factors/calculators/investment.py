"""
Investment factor metrics: asset_growth_rate (QoQ) + cma_proxy (YoY).

asset_growth_rate: quarter-over-quarter asset growth (short-term investment signal).
cma_proxy: year-over-year asset growth (structural signal; FF CMA uses similar inputs).
In get_factor, both use METRIC_DIRECTION -1: lower growth -> higher factor percentile
(FF CMA / investment-anomaly literature: low-growth firms outperform high-growth).
"""
import numpy as np
import pandas as pd

from factor_library.factors.calculators.base import winsorize_global

METRIC_KEYS = ["asset_growth_rate", "cma_proxy"]


def calculate_asset_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute asset growth rate on quarterly report sequence, then map to daily panel.

    Reuses Week 5 logic. Winsorization is applied in the caller (on the derived
    asset_growth_rate), not here.
    """
    if "ff_assets" not in df.columns or "asset_date" not in df.columns:
        df = df.copy()
        df["asset_growth_rate"] = np.nan
        return df

    reports = (
        df[["fsym_primary_listing_id", "asset_date", "ff_assets"]]
        .dropna(subset=["asset_date", "ff_assets"])
        .drop_duplicates(subset=["fsym_primary_listing_id", "asset_date"], keep="last")
        .sort_values(["fsym_primary_listing_id", "asset_date"])
    )

    reports["assets_lag_1q"] = reports.groupby("fsym_primary_listing_id")["ff_assets"].shift(1)
    reports["asset_growth_rate"] = (
        (reports["ff_assets"] - reports["assets_lag_1q"]) / reports["assets_lag_1q"].abs()
    )
    reports["asset_growth_rate"] = reports["asset_growth_rate"].replace(
        [np.inf, -np.inf], np.nan
    )

    metric_cols = ["fsym_primary_listing_id", "asset_date", "asset_growth_rate"]
    out = df.merge(
        reports[metric_cols],
        on=["fsym_primary_listing_id", "asset_date"],
        how="left",
    )

    return out


def calculate_cma_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    CMA proxy: YoY (4-quarter lag) asset growth rate.

    Captures structural investment conservatism vs aggressiveness over a full year,
    complementary to the QoQ asset_growth_rate.
    """
    if "ff_assets" not in df.columns or "asset_date" not in df.columns:
        df = df.copy()
        df["cma_proxy"] = np.nan
        return df

    reports = (
        df[["fsym_primary_listing_id", "asset_date", "ff_assets"]]
        .dropna(subset=["asset_date", "ff_assets"])
        .drop_duplicates(subset=["fsym_primary_listing_id", "asset_date"], keep="last")
        .sort_values(["fsym_primary_listing_id", "asset_date"])
    )

    reports["assets_lag_4q"] = reports.groupby("fsym_primary_listing_id")["ff_assets"].shift(4)
    reports["cma_proxy"] = (
        (reports["ff_assets"] - reports["assets_lag_4q"]) / reports["assets_lag_4q"].abs()
    )
    reports["cma_proxy"] = reports["cma_proxy"].replace([np.inf, -np.inf], np.nan)

    metric_cols = ["fsym_primary_listing_id", "asset_date", "cma_proxy"]
    out = df.merge(
        reports[metric_cols],
        on=["fsym_primary_listing_id", "asset_date"],
        how="left",
    )

    return out


def compute_investment_metrics(
    df: pd.DataFrame,
    winsorize: bool = True,
) -> pd.DataFrame:
    """
    Compute investment metrics: QoQ asset_growth_rate + YoY cma_proxy.

    Direction is applied in get_factor (METRIC_DIRECTION -1: lower growth ranks higher,
    aligning with FF CMA / investment anomaly literature).
    """
    df = calculate_asset_growth(df)
    df = calculate_cma_proxy(df)
    if winsorize:
        for col in ("asset_growth_rate", "cma_proxy"):
            if col in df.columns and df[col].notna().any():
                df[col] = winsorize_global(df[col])
    return df
