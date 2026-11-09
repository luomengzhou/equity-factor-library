"""
Main API: get_factor(universe, factor_name, ...)

Returns factor values for the given universe. Supports cache, configurable
rank_method, composite_method, winsorization bounds.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from factor_library.data.cache import get_cached_factor, save_factor_to_cache
from factor_library.data.loader import load_merged_panel, resolve_universe_to_primary
from factor_library.factors.registry import get_factor_to_metrics_map
from factor_library.factors.calculators.base import apply_rank, build_composite, winsorize_global
from factor_library.factors.calculators.investment import compute_investment_metrics
from factor_library.factors.calculators.liquidity import compute_liquidity_metrics
from factor_library.factors.calculators.market import compute_market_metrics, compute_low_beta_metrics
from factor_library.factors.calculators.momentum import compute_momentum_metrics
from factor_library.factors.calculators.quality import (
    compute_quality_metrics,
    compute_profitability_metrics,
)
from factor_library.factors.calculators.size import compute_size_metrics
from factor_library.factors.calculators.value import compute_value_metrics
from factor_library.factors.calculators.volatility import compute_volatility_metrics
from factor_library.factors.calculators.yield_module import compute_yield_metrics

# ---------------------------------------------------------------------------
# Metrics we can compute (formulas implemented; some need DB data)
# ---------------------------------------------------------------------------
IMPLEMENTED_METRICS = frozenset({
    "asset_growth_rate", "cma_proxy",
    "mom_12m_1m", "mom_6m_1m", "mom_1m_reversal", "mom_52w_high",
    "historical_volatility", "downside_volatility", "idiosyncratic_volatility",
    "market_cap",
    "earnings_yield", "book_to_price", "sales_yield", "cf_yield", "ev_ebitda",
    "dividend_yield", "shareholder_yield", "peg",
    "roe", "roa", "gross_profitability", "profit_margin", "leverage",
    "earnings_variability", "accruals", "piotroski_fscore", "qmj_proxy",
    "rmw_proxy",
    "dividend_growth", "payout_ratio",
    "beta", "market_excess_return",
    "low_beta", "bab_proxy",
    "adv_dollar", "turnover", "amihud_illiquidity", "bid_ask_spread",
    "pastor_stambaugh",
})

# Direction: -1 means lower raw value -> higher factor (e.g. size).
# Investment (-1): lower asset growth -> higher factor, following Fama-French CMA
# (Conservative-Minus-Aggressive); investment anomaly literature ranks low-growth
# names higher.
# Volatility / Low Risk (-1): lower realised volatility -> higher factor, following
# the low-volatility anomaly (Ang-Hodrick-Xing-Zhang 2006; Frazzini-Pedersen BAB).
# All factor directions in this dict are aligned to the academic-literature sign.
METRIC_DIRECTION = {
    "asset_growth_rate": -1, "cma_proxy": -1,
    "mom_12m_1m": 1, "mom_6m_1m": 1, "mom_1m_reversal": 1, "mom_52w_high": 1,
    "historical_volatility": -1, "downside_volatility": -1,
    "idiosyncratic_volatility": -1,
    "market_cap": -1,
    "earnings_yield": 1, "book_to_price": 1, "sales_yield": 1, "cf_yield": 1,
    "ev_ebitda": 1, "dividend_yield": 1, "shareholder_yield": 1, "peg": -1,
    "roe": 1, "roa": 1, "gross_profitability": 1, "profit_margin": 1,
    "leverage": -1, "earnings_variability": -1, "accruals": -1,
    "piotroski_fscore": 1, "qmj_proxy": 1, "rmw_proxy": 1,
    # Yield: reward faster dividend growth but penalize extreme payout ratio
    # (often reflects low/negative earnings quality rather than healthy income).
    "dividend_growth": 1, "payout_ratio": -1,
    # beta +1: "Market" factor scores high-beta names (market sensitivity).
    # bab_proxy is (-beta) with +1 → ranks low-beta high; distinct from Market.
    "beta": 1, "market_excess_return": 1,
    "low_beta": 1, "bab_proxy": 1,
    # Liquidity direction flipped 2026-04-20: original convention ranked
    # high-liquidity stocks high (long ADV / long turnover / short Amihud /
    # short bid-ask), which the multi-factor attribution showed earns
    # annualised alpha -8.6% (t=-3.77). Flipping to rank LOW-liquidity stocks
    # high aligns with the Amihud (2002) illiquidity premium and Pastor-
    # Stambaugh (2003) liquidity-risk literature.
    "adv_dollar": -1, "turnover": -1, "amihud_illiquidity": 1,
    "bid_ask_spread": 1, "pastor_stambaugh": 1,
}

# Metrics that should NOT be winsorized (discrete scores / already composites)
_NO_WINSORIZE = {"piotroski_fscore", "qmj_proxy"}

# Metrics that are constant within each date (no cross-sectional variation) — skip ranking
_SKIP_CROSS_SECTIONAL_RANK = {"market_excess_return"}

_STANDARD_RANK_METHOD = "percentile"
_ALLOWED_COMPOSITE_METHODS = frozenset({"equal_weight", "inverse_volatility"})
_MIN_METRIC_COVERAGE_RATIO = 0.5
_QMJ_MIN_REGRESSION_OBS = 10

# ---------------------------------------------------------------------------
# Metric group definitions for dispatch
# ---------------------------------------------------------------------------
_INVESTMENT_KEYS = {"asset_growth_rate", "cma_proxy"}
_MOMENTUM_KEYS = {"mom_12m_1m", "mom_6m_1m", "mom_1m_reversal", "mom_52w_high"}
_VOLATILITY_KEYS = {"historical_volatility", "downside_volatility",
                     "idiosyncratic_volatility"}
_VALUE_KEYS = {"earnings_yield", "book_to_price", "sales_yield", "cf_yield",
               "ev_ebitda", "dividend_yield", "shareholder_yield", "peg"}
_QUALITY_KEYS = {"roe", "roa", "gross_profitability", "profit_margin",
                 "leverage", "earnings_variability", "accruals",
                 "piotroski_fscore"}
_YIELD_KEYS = {"dividend_growth", "payout_ratio"}
_MARKET_BETA_KEYS = {"beta", "market_excess_return", "low_beta", "bab_proxy"}
_LIQUIDITY_KEYS = {"adv_dollar", "turnover", "amihud_illiquidity",
                   "bid_ask_spread"}


def _get_metrics_for_factor(factor_name: str, path: Path | str | None) -> list[str]:
    """Return implemented metric_keys for this factor."""
    mapping = get_factor_to_metrics_map(path)
    ft_key = None
    for k in mapping:
        if k.strip().lower() == factor_name.strip().lower():
            ft_key = k
            break
    if ft_key is None:
        raise ValueError(
            f"Unknown factor: '{factor_name}'. "
            f"Available: {list(mapping.keys())[:10]}..."
        )
    requested = mapping[ft_key]
    implemented = [m for m in requested if m in IMPLEMENTED_METRICS]
    missing = [m for m in requested if m not in IMPLEMENTED_METRICS]
    if not implemented:
        raise NotImplementedError(
            f"Factor '{factor_name}' has no implemented metrics. "
            f"Requested: {requested}. Missing: {missing}"
        )
    return implemented


def _directional_nanaware_mean(
    df: pd.DataFrame,
    positive_legs: list[str],
    negative_legs: list[str],
    *,
    min_valid: int = 2,
) -> pd.Series:
    """NaN-aware mean of direction-aligned legs."""
    parts: list[pd.Series] = []
    for col in positive_legs:
        if col in df.columns:
            parts.append(df[col])
    for col in negative_legs:
        if col in df.columns:
            parts.append(-df[col])

    if not parts:
        return pd.Series(np.nan, index=df.index, dtype=float)

    stacked = pd.concat(parts, axis=1)
    out = stacked.mean(axis=1, skipna=True)
    out[stacked.notna().sum(axis=1) < min_valid] = np.nan
    return out.astype(float)


def _cross_sectional_residual_by_date(
    raw: pd.Series,
    anchor: pd.Series,
    date: pd.Series,
    *,
    min_obs: int = _QMJ_MIN_REGRESSION_OBS,
) -> pd.Series:
    """Per-date OLS residual for raw ~ 1 + anchor, with de-mean fallback."""
    panel = pd.DataFrame(
        {"date": pd.to_datetime(date), "_raw": raw, "_anchor": anchor},
        index=raw.index,
    )
    residual = pd.Series(np.nan, index=panel.index, dtype=float)

    for _, grp in panel.groupby("date", sort=False):
        y = grp["_raw"]
        x = grp["_anchor"]
        y_valid = y.notna()
        if not y_valid.any():
            continue

        # Small-sample / degenerate-anchor fallback: de-meaned cross-section.
        y_demeaned = y - y[y_valid].mean()
        reg_mask = y_valid & x.notna()
        if reg_mask.sum() < min_obs:
            residual.loc[y.index[y_valid]] = y_demeaned[y_valid]
            continue

        x_reg = x[reg_mask].astype(float)
        if x_reg.nunique(dropna=True) < 2:
            residual.loc[y.index[y_valid]] = y_demeaned[y_valid]
            continue

        x_centered = x_reg - x_reg.mean()
        var = float((x_centered * x_centered).sum())
        if np.isclose(var, 0.0):
            residual.loc[y.index[y_valid]] = y_demeaned[y_valid]
            continue

        y_reg = y[reg_mask].astype(float)
        beta = float((x_centered * (y_reg - y_reg.mean())).sum() / var)
        alpha = float(y_reg.mean() - beta * x_reg.mean())
        residual.loc[y_reg.index] = y_reg - (alpha + beta * x_reg)

        missing_anchor = y_valid & ~reg_mask
        if missing_anchor.any():
            residual.loc[y.index[missing_anchor]] = y_demeaned[missing_anchor]

    return residual


def _compute_panel_with_metrics(
    df: pd.DataFrame,
    metric_keys: list[str],
    winsorize: bool = True,
    winsorize_bounds: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """
    Compute all requested metric columns. Winsorization is applied centrally
    at the end (not inside individual calculators) for configurable bounds.
    """
    mk_set = set(metric_keys)

    # Investment (QoQ growth + YoY CMA)
    if _INVESTMENT_KEYS & mk_set:
        needed = [m for m in _INVESTMENT_KEYS & mk_set if m not in df.columns]
        if needed:
            df = compute_investment_metrics(df, winsorize=False)

    # Momentum
    if _MOMENTUM_KEYS & mk_set:
        needed = [m for m in _MOMENTUM_KEYS & mk_set if m not in df.columns]
        if needed:
            df = compute_momentum_metrics(df, winsorize=False)

    # Volatility
    if _VOLATILITY_KEYS & mk_set:
        needed = [m for m in _VOLATILITY_KEYS & mk_set if m not in df.columns]
        if needed:
            df = compute_volatility_metrics(df, winsorize=False)

    # Size
    if "market_cap" in mk_set and "market_cap" not in df.columns:
        df = compute_size_metrics(df, winsorize=False)

    # Value
    if _VALUE_KEYS & mk_set:
        needed = [m for m in _VALUE_KEYS & mk_set if m not in df.columns]
        if needed:
            df = compute_value_metrics(df, winsorize=False)

    # Quality (includes profitability/rmw)
    need_quality = _QUALITY_KEYS & mk_set
    if need_quality or "qmj_proxy" in mk_set:
        if any(m not in df.columns for m in need_quality) or "qmj_proxy" in mk_set:
            df = compute_quality_metrics(df, winsorize=False)
    if "rmw_proxy" in mk_set and "rmw_proxy" not in df.columns:
        df = compute_profitability_metrics(df, winsorize=False)

    # Yield
    if _YIELD_KEYS & mk_set:
        needed = [m for m in _YIELD_KEYS & mk_set if m not in df.columns]
        if needed:
            df = compute_yield_metrics(df, winsorize=False)

    # Market / Beta / Low Beta
    if _MARKET_BETA_KEYS & mk_set:
        if "beta" not in df.columns:
            df = compute_market_metrics(df)
        if {"low_beta", "bab_proxy"} & mk_set:
            df = compute_low_beta_metrics(df)

    # QMJ proxy: orthogonalized residual (not raw QMJ leg mean).
    if "qmj_proxy" in mk_set:
        pos_legs = ["roe", "roa", "gross_profitability", "profit_margin"]
        neg_legs = ["leverage", "earnings_variability", "accruals"]

        # Keep the historical qmj_raw definition: positive quality legs plus
        # sign-flipped negative legs, aggregated in a NaN-aware way.
        qmj_raw = _directional_nanaware_mean(
            df,
            positive_legs=pos_legs,
            negative_legs=neg_legs,
            min_valid=2,
        )

        # Quality anchor is direction-consistent and NaN-aware; include
        # Piotroski when available so residualization removes broad quality.
        quality_anchor_raw = _directional_nanaware_mean(
            df,
            positive_legs=pos_legs + ["piotroski_fscore"],
            negative_legs=neg_legs,
            min_valid=2,
        )

        if qmj_raw.notna().any():
            # "Quality (Composite)" is defined as this orthogonalized residual,
            # not as the raw leg-average qmj signal.
            # Final qmj_proxy is orthogonalized residual per date:
            # qmj_raw ~ 1 + quality_anchor_raw.
            df["qmj_proxy"] = _cross_sectional_residual_by_date(
                raw=qmj_raw,
                anchor=quality_anchor_raw,
                date=df["date"],
                min_obs=_QMJ_MIN_REGRESSION_OBS,
            )
        else:
            df["qmj_proxy"] = np.nan

    # Liquidity
    if _LIQUIDITY_KEYS & mk_set:
        df = compute_liquidity_metrics(df)

    # --- Centralized winsorization ---
    if winsorize:
        lo, hi = winsorize_bounds
        for mk in metric_keys:
            if (mk in df.columns
                    and mk not in _NO_WINSORIZE
                    and df[mk].notna().any()):
                df[mk] = winsorize_global(df[mk], lower_q=lo, upper_q=hi)

    return df


def _build_factor_composite(
    df: pd.DataFrame,
    metric_keys: list[str],
    rank_method: Literal["percentile"],
    composite_method: Literal["equal_weight", "inverse_volatility"],
) -> pd.Series:
    """Apply standard factor construction pipeline.

    Pipeline:
    1) raw metrics
    2) direction-adjusted cross-sectional percentile rank per date
    3) equal-weight average of metric percentiles
    4) final cross-sectional percentile rank of the composite per date
    """
    composite, _ = _build_factor_composite_with_intermediates(
        df, metric_keys, rank_method, composite_method
    )
    return composite


def _build_factor_composite_with_intermediates(
    df: pd.DataFrame,
    metric_keys: list[str],
    rank_method: Literal["percentile"],
    composite_method: Literal["equal_weight", "inverse_volatility"],
) -> tuple[pd.Series, pd.DataFrame]:
    """Same as _build_factor_composite but returns (factor_percentile, work_df).

    `work_df` contains:
    - raw metrics
    - `_rank_{metric}`: direction-adjusted metric percentile
    - `_composite`: aggregated metric percentiles (equal-weight or inverse-vol weighted)
    - `_factor_percentile`: final factor percentile per date
    """
    if rank_method != _STANDARD_RANK_METHOD:
        raise ValueError(
            f"Week 6 standard now requires rank_method='{_STANDARD_RANK_METHOD}', "
            f"got '{rank_method}'."
        )
    if composite_method not in _ALLOWED_COMPOSITE_METHODS:
        raise ValueError(
            f"composite_method must be one of {sorted(_ALLOWED_COMPOSITE_METHODS)}, "
            f"got '{composite_method}'."
        )

    rankable = [mk for mk in metric_keys if mk not in _SKIP_CROSS_SECTIONAL_RANK]
    work = df.copy()
    rank_cols = []
    for mk in rankable:
        if mk not in work.columns:
            continue
        direction = METRIC_DIRECTION.get(mk, 1)
        col_adj = f"_adj_{mk}"
        work[col_adj] = work[mk] * direction
        work[f"_rank_{mk}"] = apply_rank(work, col_adj, method=rank_method)
        rank_cols.append(f"_rank_{mk}")

    if rank_cols:
        min_c = max(1, math.ceil(len(rank_cols) * _MIN_METRIC_COVERAGE_RATIO))
    else:
        min_c = 1

    composite = build_composite(work, rank_cols, method=composite_method, min_count=min_c)
    work["_composite"] = composite
    work["_factor_percentile"] = apply_rank(work, "_composite", method="percentile")
    return work["_factor_percentile"], work


def get_factor(
    universe: list[str],
    factor_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    rank_method: Literal["percentile"] = "percentile",
    composite_method: Literal["equal_weight", "inverse_volatility"] = "equal_weight",
    winsorize: bool = True,
    winsorize_bounds: tuple[float, float] = (0.01, 0.99),
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
    factor_library_path: str | Path | None = None,
    use_precomputed_step2: bool = True,
    base_parquet: "Path | None" = None,
    membership_parquet: "Path | None" = None,
) -> pd.DataFrame:
    """
    Return factor values for the given universe and factor.

    Parameters
    ----------
    universe : list of fsym_primary_listing_id (or fsym_id; -S mapped to -R)
    factor_name : Factor Type from preliminary_factor_library
    start_date, end_date : optional date filter
    rank_method : only 'percentile' is supported by the Week 6 standard
    composite_method : default 'equal_weight' (course standard). 'inverse_volatility'
        is optional (weights rank columns by 1/std in ``build_composite``).
    winsorize : apply global winsorization on derived signals
    winsorize_bounds : quantile bounds, e.g. (0.01, 0.99) for 1%-99%
                       or (0.05, 0.95) for 5%-95%
    use_cache : reuse cached result if available
    cache_dir : override default cache directory
    factor_library_path : override factor library Excel path
    use_precomputed_step2 : if True, use ``final/data/*/step2_factors.parquet`` when present
    base_parquet : optional custom step1 panel path
    membership_parquet : optional PIT index membership parquet (pit3000)

    Returns
    -------
    pd.DataFrame
        Columns: fsym_primary_listing_id, date, factor_value
        where factor_value is the final cross-sectional factor percentile.
    """
    universe = list(universe)
    primary = resolve_universe_to_primary(universe)

    if use_cache:
        cached = get_cached_factor(
            primary, factor_name, start_date, end_date,
            rank_method, composite_method, winsorize, cache_dir,
            base_parquet=base_parquet,
            use_precomputed_step2=use_precomputed_step2,
            membership_parquet=membership_parquet,
        )
        if cached is not None and len(cached) > 0:
            return cached

    metric_keys = _get_metrics_for_factor(factor_name, factor_library_path)

    df = load_merged_panel(
        universe=primary,
        start_date=start_date,
        end_date=end_date,
        factor_type=factor_name,
        use_precomputed_step2=use_precomputed_step2,
        metric_keys=metric_keys,
        base_parquet=base_parquet,
        membership_parquet=membership_parquet,
    )

    df = _compute_panel_with_metrics(
        df, metric_keys,
        winsorize=winsorize,
        winsorize_bounds=winsorize_bounds,
    )

    factor_values = _build_factor_composite(
        df, metric_keys, rank_method, composite_method,
    )

    out = df[["fsym_primary_listing_id", "date"]].copy()
    out["factor_value"] = factor_values
    out = out.dropna(subset=["factor_value"])
    if out.empty:
        warnings.warn(
            f"get_factor({factor_name!r}): no rows with valid factor_value "
            f"(sparse metrics, failed DB enrich, or min_count not met).",
            UserWarning,
            stacklevel=2,
        )

    if use_cache:
        save_factor_to_cache(
            out, primary, factor_name, start_date, end_date,
            rank_method, composite_method, winsorize, cache_dir,
            base_parquet=base_parquet,
            use_precomputed_step2=use_precomputed_step2,
            membership_parquet=membership_parquet,
        )

    return out.reset_index(drop=True)
