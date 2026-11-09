#!/usr/bin/env python3
"""
Build four beta diagnostics objects to separate long-only baskets from spreads:

  1) L_low           : long-only top quintile of Low Beta / BAB factor
  2) L_high          : long-only top quintile of Market (high-beta) factor
  3) BAB_spread      : L_low - L_high
  4) HML_beta_spread : L_high - L_low

Outputs under runtime/output/beta_four_objects/:
  four_objects_daily_returns.csv
  four_objects_summary.csv
  benchmark_regression.csv
  four_objects_corr.csv
  four_objects_overview.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from factor_library.analysis.benchmark_regression import (
    long_only_top_quintile_returns,
    regress_benchmark_on_factor,
)
from factor_library.analysis.ic import build_price_panel
from factor_library.config import (
    DEFAULT_OUTPUT_DIR,
    STEP1_EXTENDED_PARQUET,
    STEP1_MERGED_PARQUET,
)
from factor_library.viz.style import DIVERGING_CMAP, apply_style


FACTOR_FILES = {
    "Low Beta / BAB": "Low_Beta___BAB_composite.parquet",
    "Market": "Market_composite.parquet",
}


def _display_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[1]
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def _annual_return(ret: pd.Series) -> float:
    if ret.empty:
        return float("nan")
    gross = (1.0 + ret).prod()
    if gross <= 0:
        return float("nan")
    return float(gross ** (252.0 / len(ret)) - 1.0)


def _annual_vol(ret: pd.Series) -> float:
    if ret.empty:
        return float("nan")
    return float(ret.std(ddof=0) * np.sqrt(252.0))


def _sharpe(ret: pd.Series) -> float:
    if ret.empty:
        return float("nan")
    vol = ret.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float(ret.mean() / vol * np.sqrt(252.0))


def _summary_row(name: str, ret: pd.Series, benchmark: pd.Series) -> dict:
    merged = pd.concat({"ret": ret, "benchmark": benchmark}, axis=1).dropna()
    return {
        "object": name,
        "n_obs": int(len(merged)),
        "ann_return": _annual_return(merged["ret"]),
        "ann_vol": _annual_vol(merged["ret"]),
        "sharpe": _sharpe(merged["ret"]),
        "corr_to_benchmark": float(merged["ret"].corr(merged["benchmark"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Beta four-object diagnostics")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    out = args.output_dir / "beta_four_objects"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Load price panel and benchmark...")
    step1_path = STEP1_EXTENDED_PARQUET if STEP1_EXTENDED_PARQUET.exists() else STEP1_MERGED_PARQUET
    step1 = pd.read_parquet(step1_path)
    prices = build_price_panel(step1)
    bench = (
        step1.pivot_table(
            index="date",
            columns="fsym_primary_listing_id",
            values="price",
            aggfunc="last",
        )
        .sort_index()
        .pct_change()
        .mean(axis=1, skipna=True)
        .dropna()
        .rename("benchmark")
    )

    print("[2/4] Build L_low and L_high long-only returns...")
    composites: dict[str, pd.DataFrame] = {}
    for name, file_name in FACTOR_FILES.items():
        fp = args.output_dir / file_name
        if not fp.exists():
            raise FileNotFoundError(f"Missing composite file: {fp}")
        df = pd.read_parquet(fp).copy()
        df["date"] = pd.to_datetime(df["date"])
        composites[name] = df

    l_low = long_only_top_quintile_returns(composites["Low Beta / BAB"], prices).shift(1).dropna()
    l_low.name = "L_low"
    l_high = long_only_top_quintile_returns(composites["Market"], prices).shift(1).dropna()
    l_high.name = "L_high"

    objects = {
        "L_low": l_low,
        "L_high": l_high,
    }
    objects["BAB_spread"] = (objects["L_low"] - objects["L_high"]).rename("BAB_spread")
    objects["HML_beta_spread"] = (objects["L_high"] - objects["L_low"]).rename("HML_beta_spread")

    panel = pd.concat({"benchmark": bench, **objects}, axis=1).dropna()
    panel.to_csv(out / "four_objects_daily_returns.csv", index_label="date")

    print("[3/4] Write summary tables...")
    summary = pd.DataFrame([
        _summary_row(name, panel[name], panel["benchmark"])
        for name in objects
    ])
    summary.to_csv(out / "four_objects_summary.csv", index=False)

    regressions = pd.DataFrame([
        regress_benchmark_on_factor(panel["benchmark"], panel[name], name)
        for name in objects
    ])
    regressions.to_csv(out / "benchmark_regression.csv", index=False)

    corr = panel[["benchmark", "L_low", "L_high", "BAB_spread", "HML_beta_spread"]].corr()
    corr.to_csv(out / "four_objects_corr.csv")

    print("[4/4] Plot one-page overview...")
    apply_style()
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.0),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    plot_cols = ["benchmark", "L_low", "L_high", "BAB_spread", "HML_beta_spread"]
    plot_colors = {
        "benchmark": "#111111",
        "L_low": "#AF601A",
        "L_high": "#707B7C",
        "BAB_spread": "#1A5276",
        "HML_beta_spread": "#C0392B",
    }
    for col in plot_cols:
        curve = (1.0 + panel[col]).cumprod()
        ax1.plot(curve.index, curve.values, label=col, color=plot_colors[col], linewidth=1.5)
    ax1.set_yscale("log")
    ax1.axhline(1.0, color="grey", linewidth=0.7)
    ax1.set_title("Growth of $1 (log) on aligned sample")
    ax1.set_ylabel("Growth of $1")
    ax1.grid(True, axis="y", which="both")
    ax1.legend(loc="upper left", ncol=1, fontsize=9)

    sns.heatmap(
        corr,
        ax=ax2,
        cmap=DIVERGING_CMAP,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.82},
    )
    ax2.set_title("Daily return correlation")

    fig.suptitle("Beta four-object diagnostics: baskets vs spreads", y=1.02, fontsize=13, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out / "four_objects_overview.png")
    plt.close(fig)

    print("\nWritten:")
    for p in sorted(out.iterdir()):
        print(f"  {_display_path(p)}")


if __name__ == "__main__":
    main()
