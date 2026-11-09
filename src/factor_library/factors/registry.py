"""
Factor Registry - Load Factor Type → metrics mapping from preliminary_factor_library.xlsx

Builds Factor Type → [metric_key] mapping and validates against implemented calculators.
"""
from __future__ import annotations

from pathlib import Path
import unicodedata
import pandas as pd

from factor_library.config import (
    FACTOR_LIBRARY_PATH,
    FACTOR_METRICS_SHEET,
    FACTOR_TYPE_COL,
    METRIC_FORMULA_COL,
)


def _normalize_factor_type(s: str) -> str:
    """Normalize factor type for lookup (strip whitespace)."""
    return str(s).strip()


def _normalize_text(s: str) -> str:
    """
    Unicode-safe lowercase normalization for rule matching.

    Handles diacritics/typography variants such as "Pástor–Stambaugh".
    """
    text = unicodedata.normalize("NFKD", str(s))
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.lower().strip()


def _metric_formula_to_key(formula: str, factor_type: str) -> str | None:
    """
    Map library 'Metric / Formula' text to internal metric_key.
    Covers all Factor Types from preliminary_factor_library.xlsx.
    """
    f = _normalize_text(formula)
    ft = _normalize_text(factor_type)

    # Market
    if "market" in ft and "cross" not in f:
        if "excess" in f or "r_mkt" in f or "r_f" in f:
            return "market_excess_return"
        if "beta" in f or "cov" in f or "var" in f:
            return "beta"

    # Value
    if "value" in ft and "cross" not in f:
        if "e/p" in f or "earnings yield" in f or "1/(p/e)" in f:
            return "earnings_yield"
        if "b/p" in f or "book-to-price" in f or "1/(p/b)" in f:
            return "book_to_price"
        if "s/p" in f or "sales yield" in f or "1/(p/s)" in f:
            return "sales_yield"
        if "cf/p" in f or "cash-flow" in f or "free cf" in f:
            return "cf_yield"
        if "ev/ebitda" in f or "ev " in f or "enterprise" in f:
            return "ev_ebitda"
        if "dividend yield" in f and "shareholder" not in f:
            return "dividend_yield"
        if "shareholder yield" in f or "div yield + buyback" in f:
            return "shareholder_yield"
        if "peg" in f or "expected eps growth" in f:
            return "peg"

    # Size
    if "size" in ft:
        if "market cap" in f or "mktcap" in f or "shares" in f or "log(" in f:
            return "market_cap"

    # Investment
    if "investment" in ft:
        if "cma" in f or ("asset growth" in f and "rank" in f):
            return "cma_proxy"
        if "asset growth" in f:
            return "asset_growth_rate"

    # Momentum (preliminary_factor_library.xlsx: 12M skip 1M, 6M, 1M reversal, 52W high)
    if "momentum" in ft:
        if "12m" in f or "12 m" in f:
            return "mom_12m_1m"
        if "6m" in f or "6 m" in f:
            return "mom_6m_1m"
        if "1m" in f and "reversal" in f:
            return "mom_1m_reversal"
        if "52" in f or "52-week" in f or "week high" in f:
            return "mom_52w_high"

    # Volatility / Low Risk
    if "volatility" in ft or "low risk" in ft:
        if "historical" in f and "idio" not in f and "downside" not in f:
            return "historical_volatility"
        if "downside" in f:
            return "downside_volatility"
        if "idiosyncratic" in f or "residual" in f:
            return "idiosyncratic_volatility"

    # Low Beta / BAB
    if "low beta" in ft or "bab" in ft:
        if "betting" in f or "bab" in f or "levered" in f:
            return "bab_proxy"
        if "beta" in f:
            return "low_beta"

    # Quality
    if "quality" in ft and "composite" not in f and "qmj" not in f:
        if "roe" in f and "fe_roe" not in f:
            return "roe"
        if "roa" in f:
            return "roa"
        if "gross profit" in f or "gross profitability" in f:
            return "gross_profitability"
        if "profit margin" in f or "operating margin" in f:
            return "profit_margin"
        if "leverage" in f or "d/e" in f or "debt" in f:
            return "leverage"
        if "earnings variability" in f or "stddev" in f:
            return "earnings_variability"
        if "accruals" in f:
            return "accruals"
        if "piotroski" in f or "f-score" in f:
            return "piotroski_fscore"

    # Quality (Composite) / QMJ
    if "quality" in ft and "composite" in ft:
        return "qmj_proxy"

    # Profitability (FF)
    if "profitability" in ft and "ff" in ft:
        if "rmw" in f or "robust" in f or "operating profit" in f:
            return "rmw_proxy"

    # Liquidity (Systematic)
    if "liquidity" in ft and "systematic" in ft:
        if (
            "pastor" in f
            or "stambaugh" in f
            or "liquidity shock" in f
            or "aggregate liquidity" in f
        ):
            return "pastor_stambaugh"
        return "pastor_stambaugh"

    # Liquidity
    if "liquidity" in ft and "systematic" not in ft:
        if "adv" in f or "dollar volume" in f or "average daily" in f:
            return "adv_dollar"
        if "turnover" in f:
            return "turnover"
        if "amihud" in f or "illiquidity" in f:
            return "amihud_illiquidity"

    # Yield
    if "yield" in ft and "shareholder" not in f and "dividend" in f:
        if "growth" in f or "cagr" in f:
            return "dividend_growth"
        if "payout" in f:
            return "payout_ratio"

    return None


def load_factor_metrics(
    path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Load Factor_Metrics sheet from preliminary_factor_library.xlsx
    (``final/data/preliminary_factor_library.xlsx``).

    Returns
    -------
    pd.DataFrame
        Columns: Factor Type, Metric / Formula, metric_key (added)
    """
    p = Path(path) if path is not None else None
    if p is None or not Path(p).exists():
        p = FACTOR_LIBRARY_PATH
    if not Path(p).exists():
        raise FileNotFoundError(
            f"Factor library not found: {FACTOR_LIBRARY_PATH}"
        )

    df = pd.read_excel(p, sheet_name=FACTOR_METRICS_SHEET)
    if FACTOR_TYPE_COL not in df.columns or METRIC_FORMULA_COL not in df.columns:
        raise ValueError(
            f"Expected columns '{FACTOR_TYPE_COL}' and '{METRIC_FORMULA_COL}' "
            f"in sheet {FACTOR_METRICS_SHEET}"
        )

    df[FACTOR_TYPE_COL] = df[FACTOR_TYPE_COL].astype(str).apply(_normalize_factor_type)
    df["metric_key"] = df.apply(
        lambda r: _metric_formula_to_key(r[METRIC_FORMULA_COL], r[FACTOR_TYPE_COL]),
        axis=1,
    )
    return df


def get_factor_metrics(
    factor_name: str,
    path: Path | str | None = None,
) -> dict[str, list[str]]:
    """
    Get metrics for a Factor Type. Returns {factor_type: [metric_keys]}.

    Deduplicates metric_keys (e.g. Asset Growth and CMA proxy both map to asset_growth_rate).
    Only includes metric_keys that are not None.

    Returns
    -------
    dict
        {factor_type: [unique metric_keys]}
    """
    df = load_factor_metrics(path)
    target = _normalize_text(factor_name)
    df = df[df[FACTOR_TYPE_COL].astype(str).apply(_normalize_text) == target]
    keys = df["metric_key"].dropna().unique().tolist()
    # Deduplicate preserving order
    seen = set()
    unique = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return {factor_name.strip(): unique}


def list_factor_types(path: Path | str | None = None) -> list[str]:
    """List all Factor Types in the library."""
    df = load_factor_metrics(path)
    return df[FACTOR_TYPE_COL].unique().tolist()


def get_factor_to_metrics_map(path: Path | str | None = None) -> dict[str, list[str]]:
    """
    Build full Factor Type → [metric_keys] mapping for all factors.

    Returns
    -------
    dict
        {factor_type: [metric_keys]}
    """
    df = load_factor_metrics(path)
    result: dict[str, list[str]] = {}
    for ft in df[FACTOR_TYPE_COL].unique():
        sub = df[df[FACTOR_TYPE_COL] == ft]["metric_key"].dropna().unique().tolist()
        seen = set()
        unique = [k for k in sub if k not in seen and not seen.add(k)]  # type: ignore
        result[ft] = unique
    return result
