#!/usr/bin/env python3
"""
Regress the 100-stock equal-weight benchmark daily return on the library's
12 long/short factor returns, both one-factor-at-a-time and jointly.
Mentor's Stage-2 action item #1.

Outputs under runtime/output/benchmark_regression/:
  benchmark_returns.parquet          daily benchmark series
  per_factor_regression.csv          one row per factor (α, β, R²)
  joint_regression_summary.csv       single-row summary (R², adj-R², α)
  joint_regression_betas.csv         per-factor conditional β in the joint fit
  per_factor_r2.png                  horizontal bar chart
  joint_betas.png                    horizontal β bar chart with significance
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factor_library.analysis.benchmark_regression import (
    build_equal_weight_benchmark,
    long_only_top_quintile_returns,
    partial_r2_per_factor,
    plot_joint_betas,
    plot_partial_r2,
    plot_per_factor_r2,
    regress_benchmark_on_all,
    regress_benchmark_on_factor,
)
from factor_library.analysis.ic import build_price_panel
from factor_library.config import (
    DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    STEP1_EXTENDED_PARQUET,
    STEP1_MERGED_PARQUET,
)
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


def _display_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[1]
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark regression")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    out = args.output_dir / "benchmark_regression"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Load price panel + build equal-weight 100-stock benchmark…")
    step1_path = (STEP1_EXTENDED_PARQUET if STEP1_EXTENDED_PARQUET.exists()
                  else STEP1_MERGED_PARQUET)
    step1 = pd.read_parquet(step1_path)
    prices = build_price_panel(step1)
    bench = build_equal_weight_benchmark(step1)
    bench.to_frame("benchmark_return").to_parquet(out / "benchmark_returns.parquet")
    print(f"    benchmark: {len(bench):,} days, "
          f"{bench.index.min().date()} → {bench.index.max().date()}, "
          f"mean {bench.mean()*1e4:+.2f} bps/day, "
          f"ann {((1+bench).prod()**(252/len(bench))-1)*100:.2f}%")

    print(f"\n[2/4] Build 12 factor long/short return series…")
    factor_returns: dict[str, pd.Series] = {}
    for name, fn in FACTOR_FILES.items():
        fp = args.output_dir / fn
        if not fp.exists():
            print(f"    [skip] {name}: {fp.name} missing")
            continue
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["date"])
        ls = long_short_daily_returns(df, prices)
        # alphalens labels the forward return at the signal date t (return
        # from t to t+1). Our benchmark is indexed by the realised date
        # (return from t-1 to t). Shift by one business day so both series
        # represent the return realised on the same labelled date.
        ls = ls.shift(1).dropna()
        ls.name = name
        factor_returns[name] = ls
    print(f"    built {len(factor_returns)} factor L/S return series "
          f"(shifted to align with benchmark's backward-return convention)")

    print(f"\n[3/4] Per-factor regression (benchmark ~ α + β · factor)…")
    per_factor = pd.DataFrame([
        regress_benchmark_on_factor(bench, ret, name)
        for name, ret in factor_returns.items()
    ])
    per_factor = per_factor.sort_values("r_squared", ascending=False).reset_index(drop=True)
    per_factor.to_csv(out / "per_factor_regression.csv", index=False)
    print("\n    Per-factor R² (sorted desc):")
    print(per_factor[["factor", "n_obs", "alpha_ann_bps", "beta", "beta_t",
                      "r_squared"]].round(4).to_string(index=False))
    plot_per_factor_r2(per_factor, out / "per_factor_r2.png")

    print(f"\n[4/4] Joint regression (benchmark ~ α + Σ βᵢ · factorᵢ)…")
    summary, betas = regress_benchmark_on_all(bench, factor_returns)
    pd.DataFrame([summary]).to_csv(out / "joint_regression_summary.csv", index=False)
    betas.to_csv(out / "joint_regression_betas.csv", index=False)
    print(f"    R²       = {summary['r_squared']*100:.2f}%")
    print(f"    adj-R²   = {summary['adj_r_squared']*100:.2f}%")
    print(f"    ann α    = {summary['alpha_ann_bps']:+.1f} bps (t={summary['alpha_t']:+.2f})")
    print(f"    n obs    = {summary['n_obs']:,}")
    print(f"    n factors= {summary['n_factors']}")
    print(f"\n    Joint β (sorted by |β|):")
    print(betas.round(4).to_string(index=False))
    plot_joint_betas(betas, summary, out / "joint_betas.png")

    print(f"\n    Partial R² (leave-one-out feature importance)…")
    partial = partial_r2_per_factor(bench, factor_returns)
    partial.to_csv(out / "partial_r2.csv", index=False)
    print(partial.round(5).to_string(index=False))
    plot_partial_r2(partial, summary, out / "partial_r2.png")

    # ---------------- Long-only top quintile variant ----------------
    print(f"\n[5/5] Long-only top quintile variant (has full market beta)…")
    lo_returns: dict[str, pd.Series] = {}
    for name, fn in FACTOR_FILES.items():
        fp = args.output_dir / fn
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["date"])
        top = long_only_top_quintile_returns(df, prices)
        top = top.shift(1).dropna()  # align with benchmark's backward-return convention
        top.name = name
        lo_returns[name] = top

    lo_per_factor = pd.DataFrame([
        regress_benchmark_on_factor(bench, ret, name)
        for name, ret in lo_returns.items()
    ]).sort_values("r_squared", ascending=False).reset_index(drop=True)
    lo_per_factor.to_csv(out / "per_factor_regression_long_only.csv", index=False)
    print("\n    Per-factor R² (long-only top quintile):")
    print(lo_per_factor[["factor", "n_obs", "beta", "beta_t",
                         "r_squared"]].round(4).to_string(index=False))

    lo_summary, lo_betas = regress_benchmark_on_all(bench, lo_returns)
    pd.DataFrame([lo_summary]).to_csv(out / "joint_regression_summary_long_only.csv", index=False)
    lo_betas.to_csv(out / "joint_regression_betas_long_only.csv", index=False)
    print(f"\n    Joint long-only R² = {lo_summary['r_squared']*100:.2f}%, "
          f"adj = {lo_summary['adj_r_squared']*100:.2f}%, "
          f"α ann = {lo_summary['alpha_ann_bps']:+.1f} bps (t={lo_summary['alpha_t']:+.2f})")

    plot_per_factor_r2(lo_per_factor, out / "per_factor_r2_long_only.png")
    plot_joint_betas(lo_betas, lo_summary, out / "joint_betas_long_only.png")

    print(f"\n    Partial R² long-only (leave-one-out feature importance)…")
    lo_partial = partial_r2_per_factor(bench, lo_returns)
    lo_partial.to_csv(out / "partial_r2_long_only.csv", index=False)
    print(lo_partial.round(5).to_string(index=False))
    plot_partial_r2(lo_partial, lo_summary, out / "partial_r2_long_only.png")

    print(f"\n    Written:")
    for p in sorted(out.iterdir()):
        print(f"      {_display_path(p)}")


if __name__ == "__main__":
    main()
