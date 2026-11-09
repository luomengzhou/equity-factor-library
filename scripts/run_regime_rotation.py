#!/usr/bin/env python3
"""
Regime-driven factor rotation. Walk-forward IR-weighted allocation
conditional on the current market regime. Compares against:
  * equal-weight ensemble of the same factors (no regime overlay)
  * the strongest static single factor (Vol / Low Risk)

Uses the regime labels produced by scripts/run_regime.py (stored in
runtime/output/regime/regime_labels.parquet). Runs the rotation on
three regime dimensions — vol_regime, bullbear, trend_chop — and
saves a comparison chart for each.

Outputs under runtime/output/regime_rotation/:
  rotation_summary.csv          one row per strategy × regime dimension
  {dimension}_comparison.png    growth-of-$1 chart
  {dimension}_weights.png       monthly stacked-area weight history
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factor_library.analysis.ic import build_price_panel
from factor_library.analysis.regime_rotation import (
    build_factor_return_frame,
    equal_weight_ensemble,
    plot_rotation_comparison,
    plot_weight_stack,
    regime_rotation_returns,
    summarise_strategies,
)
from factor_library.config import (
    DEFAULT_OUTPUT_DIR,
    STEP1_EXTENDED_PARQUET,
    STEP1_MERGED_PARQUET,
)


# Exclude Low Beta / BAB (mirror of Market — joint-regression R²=1) and
# Yield (per-factor α near zero in attribution). Market is in: of the
# Market/Low-Beta pair we keep Market (the natural +β direction) and
# drop Low Beta. Liquidity is in — its 2026-04-20 sign flip lifted alpha
# to +8.3% (t=+3.62).
DEFAULT_FACTORS = [
    "Momentum", "Volatility / Low Risk", "Investment",
    "Value", "Quality", "Quality (Composite)",
    "Size", "Profitability (FF)", "Market", "Liquidity",
]
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


def _display_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[1]
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Regime-driven factor rotation")
    ap.add_argument("--factors", nargs="+", default=DEFAULT_FACTORS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument(
        "--weight-freq",
        default="QE",
        help="Resample frequency for weight charts, e.g. ME, QE, 2QE, YE.",
    )
    args = ap.parse_args()

    out = args.output_dir / "regime_rotation"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Load prices + regime labels + factor composites…")
    step1_path = (STEP1_EXTENDED_PARQUET if STEP1_EXTENDED_PARQUET.exists()
                  else STEP1_MERGED_PARQUET)
    step1 = pd.read_parquet(step1_path)
    prices = build_price_panel(step1)
    regime_path = args.output_dir / "regime" / "regime_labels.parquet"
    if not regime_path.exists():
        raise FileNotFoundError(
            f"missing {regime_path} — run scripts/run_regime.py first"
        )
    regime_labels = pd.read_parquet(regime_path)
    regime_labels["date"] = pd.to_datetime(regime_labels["date"])

    composites = {}
    for n in args.factors:
        fp = args.output_dir / FACTOR_FILES[n]
        if not fp.exists():
            print(f"    [skip] {n}")
            continue
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["date"])
        composites[n] = df
    print(f"    using {len(composites)} factors: {list(composites)}")

    print(f"\n[2/4] Building factor L/S return matrix (lookback={args.lookback} d)…")
    ret_frame = build_factor_return_frame(composites, prices)
    ret_frame = ret_frame.sort_index()
    print(f"    ret_frame: {ret_frame.shape}")

    baseline_eq = equal_weight_ensemble(ret_frame)
    single_best = ret_frame["Volatility / Low Risk"].rename("single_Vol/LowRisk")

    all_rows = []
    for dim in ("vol_regime", "bullbear", "trend_chop"):
        print(f"\n[3/4] Rotation on {dim}…")
        rotated, weights = regime_rotation_returns(
            ret_frame, regime_labels, dimension=dim,
            lookback_days=args.lookback,
        )
        print(f"    active days: {(weights.sum(axis=1) > 0).sum():,} / {len(weights):,}")

        stats = summarise_strategies({
            "rotation":                 rotated,
            "equal-weight ensemble":    baseline_eq.loc[rotated.index],
            "single factor Vol/LowRisk": single_best.loc[rotated.index],
        })
        stats["regime_dimension"] = dim
        all_rows.append(stats.reset_index())
        print(stats.round(4).to_string())

        plot_rotation_comparison(
            {f"rotation ({dim})": rotated,
             "equal-weight ensemble": baseline_eq.loc[rotated.index],
             "single factor Vol/LowRisk": single_best.loc[rotated.index]},
            out / f"{dim}_comparison.png",
            title=f"Regime rotation on {dim} — growth of $1",
        )
        plot_weight_stack(
            weights,
            out / f"{dim}_weights.png",
            title=f"Regime-rotated factor weights ({dim}, {args.weight_freq} mean)",
            freq=args.weight_freq,
        )

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(out / "rotation_summary.csv", index=False)

    print(f"\n[4/4] Summary -> {_display_path(out / 'rotation_summary.csv')}")


if __name__ == "__main__":
    main()
