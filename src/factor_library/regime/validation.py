from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd


def summarize_ic_series(
    ic_series: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    ic_col: str = "ic",
) -> pd.DataFrame:
    """
    Aggregate an IC time-series table into ic_mean/ic_std/ICIR (ir)/n_days.

    Expects columns: factor, date, ic + group_cols.
    """
    df = ic_series.copy()
    df = df.dropna(subset=["factor", ic_col, *group_cols])
    if df.empty:
        return pd.DataFrame(columns=["factor", *group_cols, "ic_mean", "ic_std", "ir", "n_days"])

    agg = (
        df.groupby(["factor", *group_cols], sort=True)[ic_col]
        .agg(ic_mean="mean", ic_std="std", n_days="count")
        .reset_index()
    )
    agg["ir"] = agg["ic_mean"] / agg["ic_std"]
    return agg


def _daily_cross_sectional_corr(x: pd.Series, y: pd.Series, method: Literal["spearman", "pearson"]) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    if method == "pearson":
        return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="pearson"))
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))


def compute_ic_summary_vs_forward_return_grouped(
    wide_factor_df: pd.DataFrame,
    factor_columns: Sequence[str],
    forward_returns_df: pd.DataFrame,
    *,
    group_df: pd.DataFrame,
    group_col: str,
    id_col: str = "fsym_primary_listing_id",
    date_col: str = "date",
    ret_col: str,
    min_cross_section: int = 10,
    method: Literal["spearman", "pearson"] = "spearman",
) -> pd.DataFrame:
    """
    Grouped IC/IR summary (per factor, per group).

    - wide_factor_df: columns [id_col, date_col, factor_1..factor_k]
    - forward_returns_df: columns [id_col, date_col, ret_col]
    - group_df: columns include [id_col, date_col, group_col] OR [date_col, group_col]
    """
    f = wide_factor_df[[id_col, date_col] + list(factor_columns)].copy()
    f[date_col] = pd.to_datetime(f[date_col])
    r = forward_returns_df[[id_col, date_col, ret_col]].copy()
    r[date_col] = pd.to_datetime(r[date_col])
    g = group_df.copy()
    g[date_col] = pd.to_datetime(g[date_col])

    # Determine join keys
    join_keys = [date_col]
    if id_col in g.columns:
        join_keys = [id_col, date_col]

    merged = f.merge(r, on=[id_col, date_col], how="inner").merge(g[[*join_keys, group_col]], on=join_keys, how="left")
    merged = merged.dropna(subset=[group_col])

    rows: list[dict[str, object]] = []
    for fac in factor_columns:
        sub = merged[[id_col, date_col, group_col, fac, ret_col]].copy()
        # group-by group label, then per-day ICs
        for grp, grp_df in sub.groupby(group_col, sort=True):
            ics: list[float] = []
            for _, day_df in grp_df.groupby(date_col, sort=True):
                if len(day_df) < min_cross_section:
                    continue
                ic = _daily_cross_sectional_corr(day_df[fac], day_df[ret_col], method)
                if not np.isnan(ic):
                    ics.append(float(ic))
            if not ics:
                rows.append(
                    {
                        "factor": fac,
                        "group_col": group_col,
                        "group": str(grp),
                        "ic_mean": np.nan,
                        "ic_std": np.nan,
                        "ir": np.nan,
                        "n_days": 0,
                        "ret_col": ret_col,
                    }
                )
                continue
            arr = np.array(ics, dtype=float)
            mu = float(np.nanmean(arr))
            sd = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else float("nan")
            ir = mu / sd if sd and not np.isnan(sd) and sd > 0 else float("nan")
            rows.append(
                {
                    "factor": fac,
                    "group_col": group_col,
                    "group": str(grp),
                    "ic_mean": mu,
                    "ic_std": sd,
                    "ir": ir,
                    "n_days": int(len(ics)),
                    "ret_col": ret_col,
                }
            )
    return pd.DataFrame(rows)

