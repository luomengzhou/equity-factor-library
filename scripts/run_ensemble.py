#!/usr/bin/env python3
"""
Stage 2F — multi-factor ensemble. Combines the library's factors into a
single composite signal using three weighting schemes and compares the
resulting long/short return against the strongest single-factor book.

Defaults:
  - **11-factor set** excludes only Low Beta / BAB (near mirror of Market in
    joint regression; keeping both double-counts market beta). Market and
    Liquidity are included; Liquidity sign was corrected 2026-04-20.
  - Optional `--all-factors` uses the full 12 names; `--pruned-9` reproduces
    an older 9-factor research slice.
  - Spearman **signal** overlap heat-maps land in `ensemble/`; long-leg
    **Jaccard** overlap is `scripts/run_factor_overlap.py` (different object).
  - Benchmark against Volatility / Low Risk (single-factor Sharpe ~0.72).

Usage:
    python scripts/run_ensemble.py
    python scripts/run_ensemble.py --pruned-9
    python scripts/run_ensemble.py --all-factors
    python scripts/run_ensemble.py --include "Momentum" "Quality" "Investment"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factor_library.analysis.ensemble import (
    build_ensemble_factor,
    compare_ensembles,
    factor_overlap_matrix,
    plot_factor_overlap_heatmap,
    plot_ensemble_comparison,
)
from factor_library.analysis.ic import build_price_panel
from factor_library.config import DATA_DIR, DEFAULT_OUTPUT_DIR
from factor_library.reports.tearsheets import long_short_daily_returns


FACTOR_FILES = {
    "Momentum":              "Momentum_composite.parquet",
    "Volatility / Low Risk": "Volatility___Low_Risk_composite.parquet",
    "Investment":            "Investment_composite.parquet",
    "Value":                 "Value_composite.parquet",
    "Quality":               "Quality_composite.parquet",
    "Quality (Composite)":   "Quality_(Composite)_composite.parquet",
    "Yield":                 "Yield_composite.parquet",
    "Size":                  "Size_composite.parquet",
    "Low Beta / BAB":        "Low_Beta___BAB_composite.parquet",
    "Liquidity":             "Liquidity_composite.parquet",
    "Market":                "Market_composite.parquet",
    "Profitability (FF)":    "Profitability_(FF)_composite.parquet",
}
ALL_FACTORS = list(FACTOR_FILES.keys())
# Same 11-factor universe as `run_factor_overlap.py` (Low Beta excluded).
DEFAULT_INCLUDE = [
    "Momentum", "Volatility / Low Risk", "Investment",
    "Value", "Quality", "Quality (Composite)",
    "Yield", "Size", "Profitability (FF)",
    "Market", "Liquidity",
]
# Pruned research set used in prior deck versions:
# keep 9 factors after removing the three attribution-flagged
# (Low Beta ≈ -Market, Liquidity historically negative alpha in old run).
PRUNED_9 = [
    "Momentum", "Volatility / Low Risk", "Investment",
    "Value", "Quality", "Quality (Composite)",
    "Yield", "Size", "Profitability (FF)",
]


def load_composites(output_dir: Path, names: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for n in names:
        fn = FACTOR_FILES.get(n)
        if not fn:
            continue
        p = output_dir / fn
        if not p.exists():
            print(f"  [skip] {n}: {p.name} not found")
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        out[n] = df
    return out


def _display_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[1]
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def _resolve_include(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    if args.include is not None:
        if args.pruned_9 or args.all_factors:
            parser.error("--include may not be combined with --pruned-9 or --all-factors")
        return list(args.include)
    if args.pruned_9:
        return list(PRUNED_9)
    if args.all_factors:
        return list(ALL_FACTORS)
    return list(DEFAULT_INCLUDE)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2F — multi-factor ensemble")
    ap.add_argument(
        "--include",
        nargs="+",
        default=None,
        help="Explicit factor list (overrides default 11-factor set).",
    )
    ap.add_argument(
        "--all-factors",
        action="store_true",
        help="Use all 12 factors including Low Beta / BAB.",
    )
    ap.add_argument(
        "--pruned-9",
        action="store_true",
        help="Use the historical 9-factor pruned set instead of the default 11.",
    )
    ap.add_argument("--benchmark", default="Volatility / Low Risk")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()
    args.include = _resolve_include(args, ap)

    print("[1/5] Loading price panel + factor composites…")
    step1 = pd.read_parquet(DATA_DIR / "step1_merged_data.parquet")
    # if the extended panel is present, refresh_prices_yfinance.py put it
    # alongside; loader.build_price_panel takes whichever we pass here.
    ext = DATA_DIR / "step1_merged_data_extended.parquet"
    if ext.exists():
        step1 = pd.read_parquet(ext)
    prices = build_price_panel(step1)
    factor_dfs = load_composites(args.output_dir, args.include)
    print(f"    factors included ({len(factor_dfs)}): {list(factor_dfs)}")

    print("\n[2/5] Factor overlap diagnostics (Spearman, signal space)…")
    out = args.output_dir / "ensemble"
    out.mkdir(parents=True, exist_ok=True)
    overlap, n_points = factor_overlap_matrix(factor_dfs, method="spearman")
    overlap_csv = out / "factor_overlap_spearman.csv"
    overlap_png = out / "factor_overlap_spearman.png"
    overlap.to_csv(overlap_csv)
    plot_factor_overlap_heatmap(overlap, n_points, overlap_png, method="spearman")
    print(f"    overlap -> {_display_path(overlap_csv)}")
    print(f"    overlap -> {_display_path(overlap_png)}")

    print("\n[3/5] Building ensembles (equal / inverse_vol / ic_weighted)…")
    ensembles: dict[str, pd.DataFrame] = {}
    for method in ("equal", "inverse_vol", "ic_weighted"):
        print(f"    - {method}")
        ensembles[method] = build_ensemble_factor(factor_dfs, prices, method=method)
        # persist as a regular composite parquet
        ensembles[method].to_parquet(
            out / f"ensemble_{method}_composite.parquet", index=False
        )

    print("\n[4/5] Computing long/short return series…")
    series: dict[str, pd.Series] = {}
    # benchmark single factor
    if args.benchmark in factor_dfs:
        series[f"single: {args.benchmark}"] = long_short_daily_returns(
            factor_dfs[args.benchmark], prices
        )
    for method, df in ensembles.items():
        series[f"ensemble: {method}"] = long_short_daily_returns(df, prices)

    summary = compare_ensembles(series)
    summary.to_csv(out / "ensemble_summary.csv")
    print(summary.round(4).to_string())

    print("\n[5/5] Writing comparison chart…")
    chart = plot_ensemble_comparison(
        series, out / "ensemble_comparison.png",
        title=(f"Multi-factor ensemble vs {args.benchmark} — growth of $1 "
               f"({len(factor_dfs)} factors)"),
    )
    print(f"    chart -> {_display_path(chart)}")
    print(f"    summary -> {_display_path(out / 'ensemble_summary.csv')}")


if __name__ == "__main__":
    main()
