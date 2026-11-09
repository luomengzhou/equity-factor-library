#!/usr/bin/env python3
"""
Apply volatility targeting to long/short Q5-Q1 return series for selected
single factors and (optionally) ensemble composites. Writes a comparison CSV
and a two-panel chart per strategy.

Usage:
    python scripts/run_vol_targeting.py
    python scripts/run_vol_targeting.py --all-factors --include-ensemble
    python scripts/run_vol_targeting.py --target-vols 0.08 0.12 0.18 --factors Volatility/LowRisk
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factor_library.analysis.ic import build_price_panel
from factor_library.backtest.vol_targeting import (
    compare_vol_targets,
    plot_vol_targeting_comparison,
)
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
HEADLINE_FACTORS = ["Momentum", "Volatility / Low Risk", "Investment"]
ALL_FACTORS = list(FACTOR_FILES.keys())


def _slug(name: str) -> str:
    return (
        name.replace(" / ", "_")
        .replace(" ", "_")
        .replace(":", "")
        .replace("(", "")
        .replace(")", "")
    )


def _display_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[1]
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Volatility targeting overlay")
    ap.add_argument(
        "--factors",
        nargs="+",
        default=HEADLINE_FACTORS,
        help="Single factors to evaluate (default: headline 3).",
    )
    ap.add_argument("--all-factors", action="store_true",
                    help="Use all single factors in FACTOR_FILES.")
    ap.add_argument("--include-ensemble", action="store_true",
                    help="Also evaluate ensemble composites from runtime/output/ensemble/.")
    ap.add_argument(
        "--ensemble-methods",
        nargs="+",
        default=["equal", "inverse_vol", "ic_weighted"],
        choices=["equal", "inverse_vol", "ic_weighted"],
        help="Ensemble composite methods to include when --include-ensemble is set.",
    )
    ap.add_argument("--target-vols", nargs="+", type=float,
                    default=[0.10, 0.15, 0.20])
    ap.add_argument("--vol-lookback", type=int, default=60)
    ap.add_argument("--max-leverage", type=float, default=3.0)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()
    if args.all_factors:
        args.factors = ALL_FACTORS

    print("[1/3] Loading price panel…")
    step1 = pd.read_parquet(DATA_DIR / "step1_merged_data.parquet")
    prices = build_price_panel(step1)

    out = args.output_dir / "vol_targeting"
    out.mkdir(parents=True, exist_ok=True)

    strategy_files: dict[str, Path] = {}
    for name in args.factors:
        fn = FACTOR_FILES.get(name)
        if fn is None:
            print(f"  [skip] {name}: no mapping")
            continue
        strategy_files[name] = args.output_dir / fn
    if args.include_ensemble:
        for method in args.ensemble_methods:
            name = f"Ensemble: {method}"
            strategy_files[name] = args.output_dir / "ensemble" / f"ensemble_{method}_composite.parquet"

    all_rows = []
    for name, fp in strategy_files.items():
        if not fp.exists():
            print(f"  [skip] {name}: {fp.name} missing")
            continue
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["date"])

        print(f"\n[2/3] {name} — long/short → vol target compare…")
        ls = long_short_daily_returns(df, prices)
        table = compare_vol_targets(
            ls, target_vols=tuple(args.target_vols),
            vol_lookback=args.vol_lookback, max_leverage=args.max_leverage,
        )
        table["strategy_name"] = name
        table["strategy_type"] = "ensemble" if name.startswith("Ensemble:") else "single_factor"
        all_rows.append(table.reset_index())
        slug = _slug(name)
        table.to_csv(out / f"{slug}_vol_target_stats.csv")
        print(table.round(4).to_string())

        png = plot_vol_targeting_comparison(
            ls, name, out / f"{slug}_vol_targeting.png",
            target_vols=tuple(args.target_vols),
            vol_lookback=args.vol_lookback, max_leverage=args.max_leverage,
        )
        print(f"      chart -> {_display_path(png)}")

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        path = out / "vol_target_summary_all.csv"
        combined.to_csv(path, index=False)
        print(f"\n[3/3] Summary -> {_display_path(path)}")


if __name__ == "__main__":
    main()
