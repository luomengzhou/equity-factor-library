#!/usr/bin/env python3
"""
Stage 2G — concentration & leave-one-out diagnostics.

For each selected factor (default: all 12), reuses long_short_attribution to produce:
  * Top-5 contribution share & effective-N time series
  * Leave-one-out table (drop each id, rebuild long/short, measure
    delta_mean and delta_IR) — sorted so the most impactful names appear
    at the top
  * "Leave-top-5" summary: what happens to Sharpe if the five
    single-day-avg top contributors are excluded altogether
  * Charts: concentration time series + LOO bar plot (Δ mean) +
    leave-top-5 cumulative return comparison

Answers the question "is this factor riding on a handful of
names or does it work across the universe?"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factor_library.backtest.long_short_attribution import (
    LongShortConfig,
    compute_long_short_returns_and_contributions,
    compute_next_day_returns_from_prices,
    leave_one_out_summary,
    summarize_concentration,
)
from factor_library.config import (
    DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    STEP1_EXTENDED_PARQUET,
    STEP1_MERGED_PARQUET,
)
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
    return name.replace(" / ", "_").replace(" ", "_")


def _plot_concentration_timeseries(conc: pd.DataFrame, factor: str, out: Path) -> Path:
    apply_style()
    conc = conc.dropna(subset=["top5_share_abs", "eff_n"])
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 5.5),
                                    gridspec_kw={"height_ratios": [1, 1]})
    col = colour_for(factor)
    ax1.plot(conc["date"], conc["top5_share_abs"] * 100, color=col, lw=1.0)
    ax1.set_ylabel("Top-5 share (%)")
    ax1.set_title(f"{factor} — concentration over time")
    ax1.axhline(30, color="grey", lw=0.6, linestyle=":")
    ax1.grid(True, axis="y")
    ax2.plot(conc["date"], conc["eff_n"], color=col, lw=1.0)
    ax2.set_ylabel("Effective N (= 1/HHI)")
    ax2.axhline(20, color="grey", lw=0.6, linestyle=":")
    ax2.grid(True, axis="y")
    ax2.set_xlabel("")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_loo_delta_mean(loo: pd.DataFrame, factor: str, out: Path, top_n: int = 20) -> Path:
    apply_style()
    hurts = loo.sort_values("delta_mean").head(top_n)  # most negative = biggest contributors
    helps = loo.sort_values("delta_mean").tail(top_n)  # biggest drag
    fig, ax = plt.subplots(figsize=(10, 6.5))
    col = colour_for(factor)
    combined = pd.concat([hurts, helps])
    combined["bp_per_day"] = combined["delta_mean"] * 1e4
    combined["id_label"] = combined["fsym_primary_listing_id"]
    ax.barh(range(len(combined)),
            combined["bp_per_day"].values,
            color=[col if v < 0 else "#888" for v in combined["bp_per_day"]],
            alpha=0.85)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(combined["id_label"], fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Δ daily mean if this id is removed  (bps / day)")
    ax.set_title(f"{factor} — leave-one-out impact (bottom/top {top_n})")
    ax.grid(True, axis="x")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Concentration + LOO diagnostics (default: all factors)")
    ap.add_argument(
        "--factors",
        nargs="+",
        default=list(FACTOR_FILES.keys()),
        help="Factor names to run. Default: all factors in FACTOR_FILES.",
    )
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    print("[1/3] Load price panel (extended if available)…")
    step1_path = STEP1_EXTENDED_PARQUET if STEP1_EXTENDED_PARQUET.exists() else STEP1_MERGED_PARQUET
    step1 = pd.read_parquet(step1_path)
    fwd = compute_next_day_returns_from_prices(step1)

    out_root = args.output_dir / "concentration"
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = LongShortConfig()
    summaries = []

    for name in args.factors:
        if name not in FACTOR_FILES:
            print(f"[skip] {name}: unknown factor name")
            continue
        fn = FACTOR_FILES[name]
        fp = args.output_dir / fn
        if not fp.exists():
            print(f"[skip] {name}: {fp.name} missing")
            continue
        df = pd.read_parquet(fp).rename(columns={"factor_value": name})
        df["date"] = pd.to_datetime(df["date"])

        print(f"\n[2/3] {name} — base strategy + concentration…")
        strat, contrib = compute_long_short_returns_and_contributions(
            df, factor_col=name, forward_1d_ret=fwd, cfg=cfg,
        )
        base_mean = float(strat["long_short_ret"].mean())
        base_std = float(strat["long_short_ret"].std(ddof=1))
        base_ir = base_mean / base_std if base_std else np.nan
        print(f"    base mean {base_mean*1e4:.2f} bps/day   IR {base_ir:.4f}   n={len(strat):,}")

        conc = summarize_concentration(contrib)
        conc.to_csv(out_root / f"{_slug(name)}_concentration_timeseries.csv", index=False)
        _plot_concentration_timeseries(conc, name,
                                       out_root / f"{_slug(name)}_concentration.png")

        print(f"    avg top-5 share: {conc['top5_share_abs'].mean()*100:.1f}%")
        print(f"    avg effective N: {conc['eff_n'].mean():.1f}")

        print(f"\n    LOO across {df['fsym_primary_listing_id'].nunique()} ids…")
        loo = leave_one_out_summary(df, factor_col=name, forward_1d_ret=fwd, cfg=cfg)
        loo.to_csv(out_root / f"{_slug(name)}_loo.csv", index=False)
        _plot_loo_delta_mean(loo, name, out_root / f"{_slug(name)}_loo_delta_mean.png")

        # Leave-top-5: drop the 5 most impactful ids (worst delta_mean) altogether
        top5_ids = loo.sort_values("delta_mean").head(5)["fsym_primary_listing_id"].tolist()
        strat_top5, _ = compute_long_short_returns_and_contributions(
            df, factor_col=name, forward_1d_ret=fwd, cfg=cfg,
            exclude_ids=top5_ids,
        )
        top5_mean = float(strat_top5["long_short_ret"].mean())
        top5_std = float(strat_top5["long_short_ret"].std(ddof=1))
        top5_ir = top5_mean / top5_std if top5_std else np.nan

        # Leave-top-1
        top1_id = loo.sort_values("delta_mean").head(1)["fsym_primary_listing_id"].tolist()
        strat_top1, _ = compute_long_short_returns_and_contributions(
            df, factor_col=name, forward_1d_ret=fwd, cfg=cfg,
            exclude_ids=top1_id,
        )
        top1_mean = float(strat_top1["long_short_ret"].mean())
        top1_std = float(strat_top1["long_short_ret"].std(ddof=1))
        top1_ir = top1_mean / top1_std if top1_std else np.nan

        summaries.append({
            "factor":                   name,
            "base_daily_mean_bps":      base_mean * 1e4,
            "base_ir":                  base_ir,
            "avg_top5_share_pct":       conc["top5_share_abs"].mean() * 100,
            "avg_effective_n":          conc["eff_n"].mean(),
            "drop_top1_id":             top1_id[0],
            "drop_top1_daily_mean_bps": top1_mean * 1e4,
            "drop_top1_ir":             top1_ir,
            "drop_top5_ids":            "|".join(top5_ids),
            "drop_top5_daily_mean_bps": top5_mean * 1e4,
            "drop_top5_ir":             top5_ir,
        })
        print(f"    drop top-1 ({top1_id[0]}): mean {top1_mean*1e4:.2f} bps/day  IR {top1_ir:.4f}"
              f"   Δ IR {(top1_ir - base_ir):+.4f}")
        print(f"    drop top-5: mean {top5_mean*1e4:.2f} bps/day  IR {top5_ir:.4f}"
              f"   Δ IR {(top5_ir - base_ir):+.4f}")

    print("\n[3/3] Writing summary table…")
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_root / "concentration_summary.csv", index=False)
    print(summary_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
