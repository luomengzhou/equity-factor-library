"""
Value factor metrics.

Formulas reference Week2_Nvidia_Value Factor/shareholder_yield.py and
table_fsym_mapping (eps_ltm, bps_q, sales_ps_ltm, fcfe_ltm, ev_q, ebitda_ltm,
div_paid_ltm, shares, buyback).

Required columns from METRIC_TABLES:
- eps_ltm: ff_eps (fds.ff_v3_ff_advanced_ltm)
- bps_q: ff_bps
- sales_ps_ltm: ff_sales_ps
- fcfe_ltm: ff_free_cf_fcfe
- ev_q: ff_entrpr_value
- ebitda_ltm: ff_ebitda_oper
- div_paid_ltm: ff_div_cf
- buyback: ff_stk_purch_cf, ff_stk_sale_cf
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from factor_library.factors.calculators.base import winsorize_global

logger = logging.getLogger(__name__)

# Column name mappings (raw DB names vs internal)
_EPS = "ff_eps"
_BPS = "ff_bps"
_SALES_PS = "ff_sales_ps"
_FCFE = "ff_free_cf_fcfe"
_EV = "ff_entrpr_value"
_EV_ALT = "ff_entrpr_val"
_EBITDA = "ff_ebitda_oper"
_DIV_CF = "ff_div_cf"
_MV = "market_cap"  # or full_mv
_PRICE = "price"
_STK_PURCH = "ff_stk_purch_cf"
_STK_SALE = "ff_stk_sale_cf"


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    """Return a/b; NaN where denominator is non-positive or missing."""
    denom = b.where(b > 0)
    out = a / denom
    return out.replace([np.inf, -np.inf], np.nan)


def _get_price_or_mcap_per_share(df: pd.DataFrame) -> pd.Series:
    """Price per share: use price if available, else market_cap/shares."""
    if "price" in df.columns:
        return df["price"]
    if "market_cap" in df.columns and "ff_com_shs_out_secs" in df.columns:
        return df["market_cap"] / df["ff_com_shs_out_secs"].replace(0, np.nan)
    return pd.Series(np.nan, index=df.index)


def compute_earnings_yield(df: pd.DataFrame) -> pd.DataFrame:
    """
    E/P = 1/(P/E). Earnings Yield = eps_ltm / price.

    METRIC_TABLES: eps_ltm -> ff_eps (fds.ff_v3_ff_advanced_ltm).
    Requires: ff_eps, price (or market_cap/shares as price proxy).
    """
    df = df.copy()
    denom = _get_price_or_mcap_per_share(df)
    if _EPS not in df.columns:
        df["earnings_yield"] = np.nan
        logger.debug("earnings_yield: missing ff_eps, returning NaN")
        return df
    df["earnings_yield"] = _safe_divide(df[_EPS], denom)
    return df


def compute_book_to_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    B/P = 1/(P/B). Book-to-Price = bps / price.

    METRIC_TABLES: bps_q -> ff_bps.
    """
    df = df.copy()
    denom = _get_price_or_mcap_per_share(df)
    if _BPS not in df.columns:
        df["book_to_price"] = np.nan
        return df
    df["book_to_price"] = _safe_divide(df[_BPS], denom)
    return df


def compute_sales_yield(df: pd.DataFrame) -> pd.DataFrame:
    """
    S/P = 1/(P/S). Sales Yield = sales_ps_ltm / price.

    METRIC_TABLES: sales_ps_ltm -> ff_sales_ps.
    """
    df = df.copy()
    denom = _get_price_or_mcap_per_share(df)
    if _SALES_PS not in df.columns:
        df["sales_yield"] = np.nan
        return df
    df["sales_yield"] = _safe_divide(df[_SALES_PS], denom)
    return df


def _get_market_cap(df: pd.DataFrame) -> pd.Series:
    """Market cap: use market_cap column or price * shares."""
    if "market_cap" in df.columns:
        return df["market_cap"]
    if "price" in df.columns and "ff_com_shs_out_secs" in df.columns:
        return df["price"] * df["ff_com_shs_out_secs"].replace(0, np.nan)
    return pd.Series(np.nan, index=df.index)


def compute_cf_yield(df: pd.DataFrame) -> pd.DataFrame:
    """
    CF/P. Cash-Flow Yield = fcfe_ltm / market_cap.

    METRIC_TABLES: fcfe_ltm -> ff_free_cf_fcfe.
    """
    df = df.copy()
    denom = _get_market_cap(df)
    if _FCFE not in df.columns:
        df["cf_yield"] = np.nan
        return df
    df["cf_yield"] = _safe_divide(df[_FCFE], denom)
    return df


def compute_ev_ebitda(df: pd.DataFrame) -> pd.DataFrame:
    """
    ev_ebitda = EBITDA/EV (yield; higher = cheaper = better value).
    Factor formula: 1/(EV/EBITDA) = EBITDA/EV.

    METRIC_TABLES: ev_q -> ff_entrpr_value, ebitda_ltm -> ff_ebitda_oper.
    """
    df = df.copy()
    ev_col = _EV if _EV in df.columns else (_EV_ALT if _EV_ALT in df.columns else None)
    if ev_col is None or _EBITDA not in df.columns:
        df["ev_ebitda"] = np.nan
        return df
    df["ev_ebitda"] = _safe_divide(df[_EBITDA], df[ev_col])
    return df


def compute_dividend_yield(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dividend Yield = div_cf / market_cap.

    METRIC_TABLES: div_paid_ltm -> ff_div_cf.
    """
    df = df.copy()
    denom = _get_market_cap(df)
    if _DIV_CF not in df.columns:
        df["dividend_yield"] = np.nan
        return df
    df["dividend_yield"] = _safe_divide(df[_DIV_CF], denom)
    return df


def compute_shareholder_yield(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shareholder Yield = Dividend Yield + Net Buyback Yield.

    METRIC_TABLES: div_paid_ltm, buyback (ff_stk_purch_cf, ff_stk_sale_cf).
    """
    df = df.copy()
    denom = _get_market_cap(df)
    has_div = _DIV_CF in df.columns
    has_bb = _STK_PURCH in df.columns and _STK_SALE in df.columns
    if not has_div and not has_bb:
        df["shareholder_yield"] = np.nan
        return df
    div_y = _safe_divide(df[_DIV_CF], denom) if has_div else pd.Series(np.nan, index=df.index)
    if _STK_PURCH in df.columns and _STK_SALE in df.columns:
        net_bb = df[_STK_PURCH] - df[_STK_SALE]
        buy_y = _safe_divide(net_bb, denom)
    else:
        buy_y = pd.Series(np.nan, index=df.index)
    if has_div and has_bb:
        df["shareholder_yield"] = div_y + buy_y
    elif has_div:
        df["shareholder_yield"] = div_y
    else:
        df["shareholder_yield"] = buy_y
    return df


def compute_peg(df: pd.DataFrame) -> pd.DataFrame:
    """
    PEG proxy = PE / EPS growth where:
    PE = price / EPS, EPS growth ≈ pct_change(EPS, 4 periods).
    """
    df = df.copy()
    if _EPS not in df.columns:
        df["peg"] = np.nan
        return df
    px = _get_price_or_mcap_per_share(df)
    pe = _safe_divide(px, df[_EPS].replace(0, np.nan))
    eps_g = (
        df.groupby("fsym_primary_listing_id")[_EPS]
        .transform(lambda s: s.pct_change(252))
    )
    df["peg"] = _safe_divide(pe, eps_g.replace(0, np.nan))
    df["peg"] = df["peg"].replace([np.inf, -np.inf], np.nan)
    return df


def compute_value_metrics(
    df: pd.DataFrame,
    winsorize: bool = True,
) -> pd.DataFrame:
    """Compute all value metrics. Adds columns: earnings_yield, book_to_price, sales_yield, cf_yield, ev_ebitda, dividend_yield, shareholder_yield."""
    df = compute_earnings_yield(df)
    df = compute_book_to_price(df)
    df = compute_sales_yield(df)
    df = compute_cf_yield(df)
    df = compute_ev_ebitda(df)
    df = compute_dividend_yield(df)
    df = compute_shareholder_yield(df)
    df = compute_peg(df)
    if winsorize:
        for col in [
            "earnings_yield",
            "book_to_price",
            "sales_yield",
            "cf_yield",
            "ev_ebitda",
            "dividend_yield",
            "shareholder_yield",
            "peg",
        ]:
            if col in df.columns and df[col].notna().any():
                df[col] = winsorize_global(df[col])
    return df
