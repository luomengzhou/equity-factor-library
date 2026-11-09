#!/usr/bin/env python3
"""
Quantile cumulative-return charts — one chart per headline factor, plus a
per-quantile annualised-stats CSV.

Usage:
    python scripts/run_quantiles.py
    python scripts/run_quantiles.py --quantiles 10 --factors Momentum
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factor_library.analysis.ic import build_price_panel
from factor_library.backtest.quantiles import (
    plot_quantile_cumulative_returns,
    quantile_annual_summary,
    quantile_cumulative_returns,
)
from factor_library.config import DATA_DIR, DEFAULT_OUTPUT_DIR


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


def _slug(name: str) -> str:
    return name.replace(" / ", "_").replace(" ", "_").replace("(", "").replace(")", "")


def _display_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[1]
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantile cumulative-return charts")
    ap.add_argument("--factors", nargs="+", default=HEADLINE_FACTORS)
    ap.add_argument("--quantiles", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=1,
                    help="Forward-return horizon in days (default 1 = daily rebalance)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--linear", action="store_true",
                    help="Linear y-axis (default is log-scale)")
    args = ap.parse_args()

    print("[1/3] Loading price panel…")
    step1 = pd.read_parquet(DATA_DIR / "step1_merged_data.parquet")
    prices = build_price_panel(step1)

    out = args.output_dir / "quantiles"
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name in args.factors:
        fn = FACTOR_FILES.get(name)
        if fn is None:
            print(f"  [skip] {name}: no parquet mapping")
            continue
        fp = args.output_dir / fn
        if not fp.exists():
            print(f"  [skip] {name}: {fp.name} not found")
            continue
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["date"])

        print(f"\n[2/3] {name} — compounding Q1–Q{args.quantiles} + long-short…")
        equity = quantile_cumulative_returns(
            df, prices, quantiles=args.quantiles, horizon_days=args.horizon,
        )
        slug = _slug(name)
        png = plot_quantile_cumulative_returns(
            equity, name, out / f"{slug}_quantiles.png",
            quantiles=args.quantiles, log_scale=not args.linear,
        )
        csv = out / f"{slug}_quantile_equity.csv"
        equity.tail(1).to_csv(csv)  # only the final value per bucket
        print(f"      chart -> {_display_path(png)}")

        summary = quantile_annual_summary(
            df, prices, quantiles=args.quantiles, horizon_days=args.horizon,
        )
        summary["factor"] = name
        summaries.append(summary.reset_index())
        print(summary.round(4).to_string())

    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        summary_csv = out / "quantile_summary_all.csv"
        combined.to_csv(summary_csv, index=False)
        print(f"\n[3/3] Summary CSV -> {_display_path(summary_csv)}")

    print("\nQuantile analysis complete.")


if __name__ == "__main__":
    main()
