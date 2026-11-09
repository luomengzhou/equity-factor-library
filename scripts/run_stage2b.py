#!/usr/bin/env python3
"""
Stage 2B — transaction-cost, multi-factor attribution and out-of-sample
validation artefacts. Builds on Stage 2A (alphalens-based IC + quantstats
tear sheets).

Produces under runtime/output/:
    cost_sensitivity/{factor}_net_sharpe_grid.csv + .png
        Rebalance-frequency × cost-bps heat-map of net Sharpe.
    attribution/ff_attribution_table.csv + heatmap.png
        12-factor OLS attribution — every factor regressed on the others.
    oos/{factor}_oos_comparison*.csv + .png
        In-sample vs out-of-sample IC and Sharpe for one or more split dates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factor_library.config import DATA_DIR, DEFAULT_OUTPUT_DIR
from factor_library.analysis.ic import build_price_panel
from factor_library.analysis.multi_factor import (
    alpha_summary,
    ff_attribution_table,
    plot_attribution_heatmap,
)
from factor_library.analysis.out_of_sample import (
    oos_validation,
    plot_oos_ic_comparison,
)
from factor_library.backtest.costs import (
    baseline_slice,
    cost_sharpe_grid,
    plot_cost_baseline_bar,
    plot_cost_sharpe_heatmap,
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
HEADLINE_FACTORS = ["Momentum", "Volatility / Low Risk", "Investment"]


def _slug(name: str) -> str:
    return name.replace(" / ", "_").replace(" ", "_").replace("(", "").replace(")", "")


def _display_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[1]
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def load_composites(output_dir: Path, names: list[str]) -> dict[str, pd.DataFrame]:
    d = {}
    for n in names:
        if n not in FACTOR_FILES:
            print(f"  [skip] {n}: unknown factor name")
            continue
        p = output_dir / FACTOR_FILES[n]
        if not p.exists():
            print(f"  [skip] {n}: {p.name} not found")
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        d[n] = df
    return d


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _split_slug(split_date: str) -> str:
    return str(pd.to_datetime(split_date).date())


def _oos_split_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Condense the tidy OOS table into one row per factor/split/horizon with
    explicit IS/OOS IR and their delta.
    """
    rows = []
    keys = ["factor", "split_date", "horizon"]
    for (factor, split_date, horizon), grp in combined.groupby(keys, sort=True):
        is_rows = grp[grp["split"] == "in_sample"]
        oos_rows = grp[grp["split"] == "out_of_sample"]
        if is_rows.empty or oos_rows.empty:
            continue
        r_is = is_rows.iloc[0]
        r_oos = oos_rows.iloc[0]
        rows.append(
            {
                "factor": factor,
                "split_date": split_date,
                "horizon": horizon,
                "is_n_obs": int(r_is["n_obs"]),
                "oos_n_obs": int(r_oos["n_obs"]),
                "is_ic_ir": float(r_is["ic_ir"]),
                "oos_ic_ir": float(r_oos["ic_ir"]),
                "delta_ic_ir": float(r_oos["ic_ir"] - r_is["ic_ir"]),
                "is_ls_sharpe": float(r_is["ls_sharpe"]),
                "oos_ls_sharpe": float(r_oos["ls_sharpe"]),
                "delta_ls_sharpe": float(r_oos["ls_sharpe"] - r_is["ls_sharpe"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2B: costs / attribution / OOS")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--oos-split", default="2020-01-01",
                    help="Single split date. Ignored if --oos-splits is provided.")
    ap.add_argument(
        "--oos-splits",
        nargs="+",
        default=None,
        help="One or more split dates (e.g. 2010-01-01 2015-01-01 2020-01-01).",
    )
    ap.add_argument("--factors", nargs="+", default=list(FACTOR_FILES.keys()),
                    help="Factors for cost grid and OOS validation. Default: all factors.")
    ap.add_argument("--headline-only", action="store_true",
                    help=f"Shortcut for --factors {' '.join(HEADLINE_FACTORS)}")
    ap.add_argument("--skip-costs", action="store_true")
    ap.add_argument("--skip-attribution", action="store_true")
    ap.add_argument("--skip-oos", action="store_true")
    args = ap.parse_args()
    if args.headline_only:
        args.factors = HEADLINE_FACTORS
    split_dates = _unique_preserve_order(args.oos_splits or [args.oos_split])

    out = args.output_dir

    print("[0/3] Loading step1 price panel…")
    step1 = pd.read_parquet(DATA_DIR / "step1_merged_data.parquet")
    prices = build_price_panel(step1)
    print(f"      prices: {prices.shape[0]:,} dates × {prices.shape[1]} tickers")

    headline = load_composites(out, args.factors)

    # ---------------- 1. Cost sensitivity ----------------
    if not args.skip_costs:
        print("\n[1/3] Cost × rebalance sensitivity…")
        cost_dir = out / "cost_sensitivity"
        all_grids = []
        for name, df in headline.items():
            grid = cost_sharpe_grid(df, prices, name)
            all_grids.append(grid)
            csv_path = cost_dir / f"{_slug(name)}_net_sharpe_grid.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            grid.to_csv(csv_path, index=False)
            png_path = plot_cost_sharpe_heatmap(grid, cost_dir / f"{_slug(name)}_heatmap.png")
            print(f"      wrote {_display_path(png_path)}")
            # Also print the 5 bps / monthly row for a quick teaser
            mo = grid[(grid["rebalance"] == "monthly") & (grid["cost_bps"] == 5.0)].iloc[0]
            print(f"      {name:24s} monthly @ 5 bps → net Sharpe {mo['net_sharpe']:5.2f}"
                  f"   ann_turnover {mo['ann_turnover']:.1f}x  net_ann_return {mo['net_ann_return']*100:.1f}%")
        if all_grids:
            all_grid_df = pd.concat(all_grids, ignore_index=True)
            baseline = baseline_slice(all_grid_df, rebalance="monthly", cost_bps=5.0)
            baseline_csv = cost_dir / "cost_summary_monthly_5bps.csv"
            baseline.to_csv(baseline_csv, index=False)
            baseline_png = plot_cost_baseline_bar(
                baseline,
                cost_dir / "cost_summary_monthly_5bps.png",
                rebalance="monthly",
                cost_bps=5.0,
            )
            print(f"      summary -> {_display_path(baseline_csv)}")
            print(f"      summary -> {_display_path(baseline_png)}")

    # ---------------- 2. Multi-factor attribution ----------------
    if not args.skip_attribution:
        print("\n[2/3] FF-style multi-factor attribution on all 12 factors…")
        all_composites = load_composites(out, list(FACTOR_FILES.keys()))
        # Each factor contributes its *daily long-short* return as the
        # regression variable.
        factor_returns = {}
        for name, df in all_composites.items():
            factor_returns[name] = long_short_daily_returns(df, prices)
        table = ff_attribution_table(factor_returns)
        attr_dir = out / "attribution"
        attr_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(attr_dir / "ff_attribution_table.csv", index=False)
        alphas = alpha_summary(table)
        alphas.to_csv(attr_dir / "ff_alpha_summary.csv", index=False)
        print("\n      Per-factor alpha (annualised, sorted):")
        print(alphas.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        plot_attribution_heatmap(table, attr_dir / "attribution_beta_heatmap.png", metric="beta")
        plot_attribution_heatmap(table, attr_dir / "attribution_tstat_heatmap.png",
                                 metric="t_stat", mask_insignificant=None)

    # ---------------- 3. Out-of-sample validation ----------------
    if not args.skip_oos:
        print(f"\n[3/3] In-sample vs out-of-sample (splits = {split_dates})…")
        oos_dir = out / "oos"
        oos_dir.mkdir(parents=True, exist_ok=True)
        oos_tables = []
        for name, df in headline.items():
            print(f"      {name:24s}")
            for split in split_dates:
                tbl = oos_validation(df, prices, name, split_date=split)
                if tbl.empty:
                    print(f"          split {split}: no usable rows")
                    continue
                oos_tables.append(tbl)
                split_slug = _split_slug(split)
                multi = len(split_dates) > 1
                csv_name = (
                    f"{_slug(name)}_oos_comparison_{split_slug}.csv"
                    if multi
                    else f"{_slug(name)}_oos_comparison.csv"
                )
                png_name = (
                    f"{_slug(name)}_oos_bars_{split_slug}.png"
                    if multi
                    else f"{_slug(name)}_oos_bars.png"
                )
                csv_path = oos_dir / csv_name
                png_path = oos_dir / png_name
                tbl.to_csv(csv_path, index=False)
                plot_oos_ic_comparison(tbl, name, png_path, split_date=split_slug)
                print(f"          split {split_slug}:")
                for h in tbl["horizon"].unique():
                    r_is = tbl[(tbl["horizon"] == h) & (tbl["split"] == "in_sample")]
                    r_oos = tbl[(tbl["horizon"] == h) & (tbl["split"] == "out_of_sample")]
                    if r_is.empty or r_oos.empty:
                        continue
                    print(
                        f"              {h}  IS IR {float(r_is.iloc[0]['ic_ir']):6.3f}  ->  "
                        f"OOS IR {float(r_oos.iloc[0]['ic_ir']):6.3f}"
                    )
        if oos_tables:
            combined = pd.concat(oos_tables, ignore_index=True)
            combined.to_csv(oos_dir / "oos_comparison_all.csv", index=False)
            if len(split_dates) > 1:
                combined.to_csv(oos_dir / "oos_comparison_all_splits.csv", index=False)
            summary = _oos_split_summary(combined)
            if not summary.empty:
                summary.to_csv(oos_dir / "oos_split_comparison_summary.csv", index=False)

    print("\nStage 2B complete.")


if __name__ == "__main__":
    main()
