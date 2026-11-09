#!/usr/bin/env python3
"""
Compare equal-weight Q5-Q1 vs signal-weighted long/short for each factor.
Mentor's Stage-2 suggestion: "if momentum signal is higher then you allocate
more weight." Here "signal" means the factor's cross-sectional percentile
``factor_value``.

Outputs under runtime/output/signal_weighted/:
  signal_weighted_summary.csv        two-row-per-factor comparison
  {factor}_equity.png                growth-of-$1 chart, eq vs sw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factor_library.analysis.ic import build_price_panel
from factor_library.backtest.signal_weighted import (
    compare_weighting,
    signal_weighted_daily_returns,
)
from factor_library.config import (
    DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    STEP1_EXTENDED_PARQUET,
    STEP1_MERGED_PARQUET,
)
from factor_library.reports.tearsheets import long_short_daily_returns
from factor_library.viz.style import apply_style, colour_for


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


def _slug(name: str) -> str:
    return name.replace(" / ", "_").replace(" ", "_").replace("(", "").replace(")", "")


def _display_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[1]
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Equal-weight vs signal-weighted L/S")
    ap.add_argument("--factors", nargs="+", default=list(FACTOR_FILES.keys()))
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    out = args.output_dir / "signal_weighted"
    out.mkdir(parents=True, exist_ok=True)
    summary_csv = out / "signal_weighted_summary.csv"

    print("[1/3] Load price panel…")
    step1_path = (STEP1_EXTENDED_PARQUET if STEP1_EXTENDED_PARQUET.exists()
                  else STEP1_MERGED_PARQUET)
    step1 = pd.read_parquet(step1_path)
    prices = build_price_panel(step1)

    summaries = []
    for name in args.factors:
        fn = FACTOR_FILES.get(name)
        fp = args.output_dir / fn if fn else None
        if not fp or not fp.exists():
            print(f"  [skip] {name}: missing composite")
            continue
        df = pd.read_parquet(fp)
        df["date"] = pd.to_datetime(df["date"])

        print(f"\n[2/3] {name} — comparing weightings…")
        table = compare_weighting(df, prices, name)
        summaries.append(table)
        pd.concat(summaries, ignore_index=True).to_csv(summary_csv, index=False)
        print(table.round(4).to_string(index=False))

        # Equity chart
        apply_style()
        eq_ret = long_short_daily_returns(df, prices)
        sw_ret = signal_weighted_daily_returns(df, prices)
        eq_curve = (1 + eq_ret).cumprod()
        sw_curve = (1 + sw_ret).cumprod()

        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        c = colour_for(name)
        ax.plot(eq_curve.index, eq_curve.values, color=c, lw=1.3, alpha=0.7,
                label="Equal-weight Q5−Q1")
        ax.plot(sw_curve.index, sw_curve.values, color=c, lw=1.6,
                linestyle="--", label="Signal-weighted")
        ax.set_yscale("log")
        ax.axhline(1, color="grey", lw=0.7)
        ax.set_title(f"{name} — equal-weight vs signal-weighted L/S (growth of $1, log)")
        ax.set_ylabel("Growth of $1")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, axis="y", which="both")
        fig.tight_layout()
        fig.savefig(out / f"{_slug(name)}_equity.png")
        plt.close(fig)

    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        combined.to_csv(summary_csv, index=False)
        print(f"\n[3/3] Summary -> {_display_path(summary_csv)}")


if __name__ == "__main__":
    main()
