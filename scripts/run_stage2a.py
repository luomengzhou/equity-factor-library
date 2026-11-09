#!/usr/bin/env python3
"""
Stage 2A — publication-quality charts & tear sheets.

For each selected factor (default: all 12; use --headline-only for the
three headline factors) this script produces:
    * IC decay bar chart (1d / 5d / 20d / 60d Spearman IC + IR)
    * Rolling 20-day IC with 95 % CI band
    * Quantstats long/short HTML tear sheet

Cross-factor charts produced once:
    * Correlation clustermap (inner-join on common (id, date) points)
    * Coverage timeline (horizontal bar per factor)

Usage:
    python scripts/run_stage2a.py
    python scripts/run_stage2a.py --headline-only
    python scripts/run_stage2a.py --factors Momentum --skip-tearsheet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factor_library.config import DATA_DIR, DEFAULT_OUTPUT_DIR, PROJECT_ROOT
from factor_library.analysis.ic import build_price_panel, run_ic_analysis
from factor_library.reports.tearsheets import build_tearsheet, tearsheet_summary_stats
from factor_library.viz.overview import (
    plot_correlation_clustermap,
    plot_coverage_timeline,
)


# Map on-disk parquet stems to display names used in titles and palette lookups.
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
HEADLINE_FACTORS = ["Momentum", "Volatility / Low Risk", "Investment"]
ALL_FACTORS = list(FACTOR_FILES.keys())


def _slug(name: str) -> str:
    return name.replace(" / ", "_").replace(" ", "_")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_composites(output_dir: Path, factor_names: list[str]) -> dict[str, pd.DataFrame]:
    composites = {}
    for name in factor_names:
        if name not in FACTOR_FILES:
            print(f"  [skip] {name}: unknown factor name")
            continue
        p = output_dir / FACTOR_FILES[name]
        if not p.exists():
            print(f"  [skip] {name}: {p.name} not found")
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        composites[name] = df
    return composites


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2A: charts + tear sheets")
    ap.add_argument("--factors", nargs="+", default=ALL_FACTORS,
                    help="Factors for IC + tear sheet. Default: all factors.")
    ap.add_argument("--headline-only", action="store_true",
                    help=f"Shortcut for --factors {' '.join(HEADLINE_FACTORS)}")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help="Root for artefacts (default: runtime/output)")
    ap.add_argument("--skip-tearsheet", action="store_true")
    ap.add_argument("--skip-overview", action="store_true",
                    help="Skip cross-factor correlation / coverage charts")
    ap.add_argument(
        "--charts-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "gallery" / "charts",
        help="Directory for overview charts (default: docs/gallery/charts)",
    )
    args = ap.parse_args()
    if args.headline_only:
        args.factors = HEADLINE_FACTORS

    output_dir: Path = args.output_dir

    # Load the daily price panel once — shared by IC and tear-sheet pipelines.
    print("[1/4] Loading step1 price panel…")
    step1 = pd.read_parquet(DATA_DIR / "step1_merged_data.parquet")
    prices_wide = build_price_panel(step1)
    print(f"      prices: {prices_wide.shape[0]:,} dates × {prices_wide.shape[1]} tickers")

    # ---------------- per-factor IC + tearsheet ----------------
    per_factor = load_composites(output_dir, args.factors)
    summary_stats = []
    for name, df in per_factor.items():
        slug = _slug(name)
        print(f"\n[2/4] {name} — IC decay + rolling IC…")
        ic_bundle = run_ic_analysis(df, prices_wide, name, output_dir)
        print(f"      decay  -> {_display_path(ic_bundle['decay_png'])}")
        print(f"      rolling-> {_display_path(ic_bundle['rolling_png'])}")
        print(f"      summary:\n{ic_bundle['summary'].round(4).to_string()}")

        if not args.skip_tearsheet:
            print(f"\n[3/4] {name} — quantstats tear sheet…")
            ts_path = build_tearsheet(
                df, prices_wide, name,
                output_dir / "tearsheets" / f"{slug}.html",
            )
            stats = tearsheet_summary_stats(df, prices_wide)
            summary_stats.append({"factor": name, **stats})
            print(f"      tearsheet -> {_display_path(ts_path)}")

    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_csv = output_dir / "tearsheets" / "summary_stats.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n      summary CSV -> {_display_path(summary_csv)}")

    # ---------------- cross-factor overview ----------------
    if not args.skip_overview:
        print("\n[4/4] Cross-factor overview charts…")
        all_composites = load_composites(output_dir, list(FACTOR_FILES.keys()))
        charts = args.charts_dir
        p1 = plot_correlation_clustermap(all_composites, charts / "factor_correlation_clustermap.png")
        p2 = plot_coverage_timeline(all_composites, charts / "factor_coverage_timeline.png")
        print(f"      {_display_path(p1)}")
        print(f"      {_display_path(p2)}")

    print("\nStage 2A complete.")


if __name__ == "__main__":
    main()
