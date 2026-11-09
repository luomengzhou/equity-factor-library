#!/usr/bin/env python3
"""
Stage 2C — regime-conditional factor validation.

For each of three market regimes — trend/chop, bull/bear, high-vol/low-vol —
computes per-factor IC and IR (factor_value vs 20-day forward return)
restricted to days in each regime.

Outputs under runtime/output/regime/:
    regime_labels.parquet                 daily regime classification
    ic_by_trend_chop.csv                  12 factors × 2 regimes
    ic_by_bull_bear.csv                   12 factors × 2 regimes
    ic_by_vol.csv                         12 factors × 2 regimes
    heatmap_{trend_chop,bull_bear,vol}.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from factor_library.config import DATA_DIR, DEFAULT_OUTPUT_DIR
from factor_library.regime import (
    RegimeConfig,
    VolBullBearConfig,
    compute_ic_summary_vs_forward_return_grouped,
    compute_market_return_from_prices,
    label_bull_bear_regime,
    label_regime_from_market_return,
    label_volatility_regime,
)
from factor_library.regime.plots import plot_regime_ir_heatmap


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


def load_long_factor_frame(output_dir: Path) -> pd.DataFrame:
    """Concatenate every composite parquet into one wide frame suitable for
    the grouped-IC routine: columns [fsym_primary_listing_id, date, factor_1..k]."""
    frames = []
    for display, fn in FACTOR_FILES.items():
        p = output_dir / fn
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        col = display.replace(" / ", "_").replace(" ", "_")
        df = df.rename(columns={"factor_value": col})
        frames.append(df.set_index(["fsym_primary_listing_id", "date"])[[col]])
    wide = pd.concat(frames, axis=1, join="outer").reset_index()
    return wide


def build_forward_returns(step1: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Build a long DataFrame of [id, date, fwd_ret_{h}d]."""
    step1 = step1.sort_values(["fsym_primary_listing_id", "date"]).copy()
    step1[f"fwd_ret_{horizon}d"] = (
        step1.groupby("fsym_primary_listing_id")["price"]
             .pct_change(horizon).shift(-horizon)
    )
    return step1[["fsym_primary_listing_id", "date", f"fwd_ret_{horizon}d"]].dropna()


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2C — regime-conditional IC")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--horizon", type=int, default=20,
                    help="Forward-return horizon for IC (days)")
    args = ap.parse_args()

    out = args.output_dir / "regime"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Load price panel + factor frame…")
    step1 = pd.read_parquet(DATA_DIR / "step1_merged_data.parquet")
    factors = load_long_factor_frame(args.output_dir)
    print(f"      factor frame: {factors.shape[0]:,} rows × {factors.shape[1]} cols")

    print(f"[2/4] Compute forward returns (h={args.horizon} d)…")
    fwd = build_forward_returns(step1, horizon=args.horizon)
    print(f"      forward returns: {len(fwd):,} rows")

    print("[3/4] Label regimes…")
    mkt = compute_market_return_from_prices(step1)
    trend   = label_regime_from_market_return(mkt, cfg=RegimeConfig())
    bullbr  = label_bull_bear_regime(mkt,         cfg=VolBullBearConfig())
    volreg  = label_volatility_regime(mkt,        cfg=VolBullBearConfig())
    labels = trend[["date", "regime"]].merge(
        bullbr[["date", "bullbear"]], on="date", how="outer"
    ).merge(
        volreg[["date", "vol_regime"]], on="date", how="outer"
    ).rename(columns={"regime": "trend_chop"}).sort_values("date")
    labels.to_parquet(out / "regime_labels.parquet", index=False)
    print(f"      regime_labels.parquet written: {len(labels):,} rows")
    print(labels.tail(5).to_string(index=False))

    print(f"\n[4/4] Group IC — 12 factors × 3 regime dimensions…")
    factor_cols = [c for c in factors.columns if c not in {"fsym_primary_listing_id", "date"}]
    ret_col = f"fwd_ret_{args.horizon}d"

    runs = [
        ("trend_chop", ["chop", "trend"]),
        ("bullbear",   ["bear", "bull"]),
        ("vol_regime", ["low_vol", "high_vol"]),
    ]
    for group_col, order in runs:
        grouped = compute_ic_summary_vs_forward_return_grouped(
            wide_factor_df=factors, factor_columns=factor_cols,
            forward_returns_df=fwd, group_df=labels,
            group_col=group_col, ret_col=ret_col,
        )
        csv = out / f"ic_by_{group_col}.csv"
        grouped.to_csv(csv, index=False)
        png = plot_regime_ir_heatmap(
            grouped, out / f"heatmap_{group_col}.png",
            metric="ir",
            title=f"Factor IR by {group_col.replace('_', ' / ')} regime  (h={args.horizon}d)",
            regime_order=order,
        )
        # Quick console preview
        piv = grouped.pivot(index="factor", columns="group", values="ir")
        print(f"\n      {group_col}:")
        print(piv.round(3).to_string())
        print(f"      -> {_display_path(csv)}")
        print(f"      -> {_display_path(png)}")

    print("\nStage 2C complete.")


if __name__ == "__main__":
    main()
