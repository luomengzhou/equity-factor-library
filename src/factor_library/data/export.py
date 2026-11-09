#!/usr/bin/env python3
"""
Export full factor wide table: raw metrics, metric percentiles, factor percentile.

Outputs a CSV (wide format):
  Date | Security | Metric1, ... | Metric1_rank, ... | {Factor}_Score | {Factor}_Rank

Default: **all** securities on one date (no row cap). Use ``--all-dates`` for the full time series.

Usage:
  python app/scripts/01_export_factor_wide.py --factor Value
  python app/scripts/01_export_factor_wide.py --factor Momentum --date 2024-01-31
  python app/scripts/01_export_factor_wide.py --factor Value --all-dates
  python app/scripts/01_export_factor_wide.py --factor Momentum --output /path/to/custom.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factor_library.config import DEFAULT_EXPORT_DIR
from factor_library.data.loader import load_merged_panel, load_symbol_coverage
from factor_library.factors.get_factor import (
    _build_factor_composite_with_intermediates,
    _compute_panel_with_metrics,
    _get_metrics_for_factor,
)

# Metric internal name -> display name for CSV headers
METRIC_DISPLAY_NAMES: dict[str, str] = {
    "earnings_yield": "E/P",
    "book_to_price": "B/P",
    "sales_yield": "S/P",
    "cf_yield": "CF/P",
    "ev_ebitda": "EV_EBITDA",
    "dividend_yield": "Div_Yield",
    "shareholder_yield": "Shareholder_Yield",
    "peg": "PEG",
    "asset_growth_rate": "Asset_Growth_QoQ",
    "cma_proxy": "CMA_Proxy_YoY",
    "mom_12m_1m": "Mom_12m1m",
    "mom_6m_1m": "Mom_6m1m",
    "mom_1m_reversal": "Mom_1m_Reversal",
    "mom_52w_high": "Mom_52wHigh",
    "historical_volatility": "Hist_Vol",
    "downside_volatility": "Downside_Vol",
    "idiosyncratic_volatility": "Idio_Vol",
    "market_cap": "Market_Cap",
    "roe": "ROE",
    "roa": "ROA",
    "gross_profitability": "Gross_Prof",
    "profit_margin": "Profit_Margin",
    "leverage": "Leverage",
    "earnings_variability": "Earn_Var",
    "accruals": "Accruals",
    "piotroski_fscore": "Piotroski",
    "qmj_proxy": "QMJ_Proxy",
    "rmw_proxy": "RMW_Proxy",
    "dividend_growth": "Div_Growth",
    "payout_ratio": "Payout_Ratio",
    "beta": "Beta",
    "low_beta": "Low_Beta",
    "bab_proxy": "BAB_Proxy",
    "adv_dollar": "ADV_Dollar",
    "turnover": "Turnover",
    "amihud_illiquidity": "Amihud",
    "bid_ask_spread": "BidAsk_Spread",
}


def _metric_display_name(metric: str) -> str:
    return METRIC_DISPLAY_NAMES.get(metric, metric)


def _security_name_map() -> pd.DataFrame:
    """Build Security ID -> human-readable name mapping from symbol coverage."""
    cov = load_symbol_coverage()
    if cov.empty or "fsym_primary_listing_id" not in cov.columns:
        return pd.DataFrame(columns=["Security", "Security_Name"])

    name_candidates = [
        "proper_name",
        "ticker",
        "symbol",
        "security_name",
        "company_name",
    ]
    name_col = next((c for c in name_candidates if c in cov.columns), None)
    if name_col is None:
        return pd.DataFrame(columns=["Security", "Security_Name"])

    mapped = cov[["fsym_primary_listing_id", name_col]].copy()
    mapped[name_col] = mapped[name_col].astype(str).str.strip()
    mapped = mapped.dropna(subset=["fsym_primary_listing_id"])
    mapped = mapped[mapped[name_col] != ""]
    mapped = mapped.drop_duplicates(subset=["fsym_primary_listing_id"], keep="first")
    mapped = mapped.rename(
        columns={"fsym_primary_listing_id": "Security", name_col: "Security_Name"}
    )
    return mapped


def _pick_auto_date(work: pd.DataFrame, min_rows: int) -> pd.Timestamp:
    """Latest date with at least ``min_rows`` valid factor percentile rows."""
    valid = work.dropna(subset=["_factor_percentile"])
    if valid.empty:
        raise ValueError("No valid factor percentile values found for this factor.")

    coverage = valid.groupby("date")["_factor_percentile"].count().sort_index()
    enough = coverage[coverage >= min_rows]
    if not enough.empty:
        return pd.Timestamp(enough.index.max())
    return pd.Timestamp(coverage.idxmax())


def _select_rows_for_export(
    date_work: pd.DataFrame,
    raw_cols: list[str],
    rank_cols: list[str],
    limit: int | None,
) -> pd.DataFrame:
    """If ``limit`` is None, return all rows (sorted by Security). Else top ``limit`` by completeness."""
    selected = date_work.dropna(subset=["_factor_percentile"]).copy()
    if selected.empty:
        return selected

    if limit is None:
        return selected.sort_values("fsym_primary_listing_id", kind="mergesort")

    if raw_cols:
        selected["_raw_non_na"] = selected[raw_cols].notna().sum(axis=1)
    else:
        selected["_raw_non_na"] = 0
    if rank_cols:
        selected["_rank_non_na"] = selected[rank_cols].notna().sum(axis=1)
    else:
        selected["_rank_non_na"] = 0

    target = min(limit, len(selected))
    if raw_cols:
        threshold = int(selected["_raw_non_na"].max())
        while threshold > 1:
            subset = selected[selected["_raw_non_na"] >= threshold]
            if len(subset) >= target:
                selected = subset
                break
            threshold -= 1

    return selected.sort_values(
        ["_raw_non_na", "_rank_non_na", "_factor_percentile"],
        ascending=[False, False, False],
    ).head(limit)


def _build_wide_table(
    date_work: pd.DataFrame,
    factor_name: str,
    raw_cols: list[str],
    rank_cols: list[str],
) -> pd.DataFrame:
    """Build wide CSV frame from a work slice (one or many dates)."""
    ranked_metrics = [rcol.replace("_rank_", "") for rcol in rank_cols]

    out = date_work[["date", "fsym_primary_listing_id"]].copy()
    out = out.rename(columns={"fsym_primary_listing_id": "Security"})
    out["Date"] = out["date"].dt.strftime("%Y-%m-%d")
    out = out.drop(columns=["date"])
    name_map = _security_name_map()
    if not name_map.empty:
        out = out.merge(name_map, on="Security", how="left")
    else:
        out["Security_Name"] = pd.NA

    for metric in raw_cols:
        out[_metric_display_name(metric)] = date_work[metric].values

    for metric in ranked_metrics:
        rcol = f"_rank_{metric}"
        out[f"{_metric_display_name(metric)}_rank"] = date_work[rcol].values

    safe_factor = factor_name.replace(" ", "_").replace("/", "_")
    score_col = f"{safe_factor}_Score"
    rank_col = f"{safe_factor}_Rank"
    out[score_col] = date_work["_factor_percentile"].values
    out[rank_col] = date_work["_rank_pct"].round(0).astype("Int64").values

    raw_display_cols = [
        _metric_display_name(m)
        for m in raw_cols
        if _metric_display_name(m) in out.columns
    ]
    rank_display_cols = [
        f"{_metric_display_name(m)}_rank"
        for m in ranked_metrics
        if f"{_metric_display_name(m)}_rank" in out.columns
    ]
    return out[
        ["Date", "Security", "Security_Name"]
        + raw_display_cols
        + rank_display_cols
        + [score_col, rank_col]
    ]


def export_factor_wide(
    factor_name: str,
    sample_date: str | None = None,
    limit: int | None = None,
    universe: list[str] | None = None,
    output_path: Path | None = None,
    rank_method: str = "percentile",
    composite_method: str = "equal_weight",
    all_dates: bool = False,
) -> pd.DataFrame:
    """
    Export a wide CSV with raw metrics, metric percentiles, factor percentile.

    By default exports **all** securities on one date (``limit=None``).
    With ``all_dates=True``, exports all (date, security) rows with valid factor scores.
    """
    metric_keys = _get_metrics_for_factor(factor_name, None)
    rankable = [m for m in metric_keys if m != "market_excess_return"]

    df = load_merged_panel(
        universe=universe,
        start_date=None,
        end_date=None,
        factor_type=factor_name,
        use_precomputed_step2=True,
        metric_keys=metric_keys,
    )
    if df.empty:
        raise ValueError("Empty panel. Check data/DB.")

    df = _compute_panel_with_metrics(df, metric_keys, winsorize=True)
    factor_pct, work = _build_factor_composite_with_intermediates(
        df, metric_keys, rank_method, composite_method
    )
    work["_factor_percentile"] = factor_pct
    work["_rank_pct"] = work["_factor_percentile"] * 100
    work["date"] = pd.to_datetime(work["date"])

    raw_cols = [m for m in rankable if m in work.columns]
    rank_cols = [f"_rank_{m}" for m in rankable if f"_rank_{m}" in work.columns]

    if all_dates:
        if limit is not None:
            raise ValueError("--limit cannot be used with --all-dates")
        date_work = work.dropna(subset=["_factor_percentile"]).copy()
        date_work = date_work.sort_values(["date", "fsym_primary_listing_id"], kind="mergesort")
    else:
        if sample_date:
            pick_date = pd.to_datetime(sample_date)
        else:
            min_rows = max(1, limit or 1)
            pick_date = _pick_auto_date(work, min_rows=min_rows)

        date_work = work[work["date"] == pick_date].copy()
        if date_work.empty:
            raise ValueError(
                f"No data for date {pick_date.date()}. Try a different date or omit --date."
            )

        date_work = _select_rows_for_export(
            date_work=date_work,
            raw_cols=raw_cols,
            rank_cols=rank_cols,
            limit=limit,
        )
        if date_work.empty:
            raise ValueError(
                f"No valid rows with factor score on {pick_date.date()} for factor '{factor_name}'."
            )

    out = _build_wide_table(date_work, factor_name, raw_cols, rank_cols)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False, float_format="%.4f")
        n_dates = out["Date"].nunique()
        print(
            f"Saved: {output_path} ({len(out)} rows, {n_dates} distinct date(s))"
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export full factor wide CSV (metrics, ranks, factor score)"
    )
    parser.add_argument("--factor", type=str, required=True, help="Factor name (e.g. Value, Momentum)")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Single date (YYYY-MM-DD). Default: latest date with enough rows (or use --all-dates)",
    )
    parser.add_argument(
        "--all-dates",
        action="store_true",
        help="Export every date × security (full panel); ignores --date/--limit",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cap rows on the chosen single date only (default: no cap = all names)",
    )
    parser.add_argument("--universe", nargs="*", default=None, help="Optional universe IDs")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output CSV path. Default: {DEFAULT_EXPORT_DIR}/{{Factor}}_factor_wide.csv",
    )
    parser.add_argument(
        "--rank-method",
        choices=["percentile"],
        default="percentile",
        help="Week 6 standard fixes metric ranking to percentile.",
    )
    parser.add_argument(
        "--composite-method",
        choices=["equal_weight"],
        default="equal_weight",
        help="Week 6 standard fixes composite aggregation to equal weight.",
    )
    args = parser.parse_args()

    safe_name = args.factor.replace(" ", "_").replace("/", "_")
    default_path = DEFAULT_EXPORT_DIR / f"{safe_name}_factor_wide.csv"
    out_path = args.output or default_path

    export_factor_wide(
        factor_name=args.factor,
        sample_date=args.date,
        limit=args.limit,
        universe=args.universe,
        output_path=out_path,
        rank_method=args.rank_method,
        composite_method=args.composite_method,
        all_dates=args.all_dates,
    )


if __name__ == "__main__":
    main()
