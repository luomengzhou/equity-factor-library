#!/usr/bin/env python3
"""
Full pipeline: load → compute metrics → composite → factor percentile → report.

Usage:
    python scripts/run_pipeline.py                         # all factors
    python scripts/run_pipeline.py --factor Momentum       # single factor
    python scripts/run_pipeline.py --winsorize-bounds 0.05 0.95  # 5%-95% winsorize
    python scripts/run_pipeline.py --report                # generate charts after pipeline
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factor_library.config import DEFAULT_OUTPUT_DIR
from factor_library.data.loader import load_merged_panel
from factor_library.factors.registry import list_factor_types
from factor_library.factors.get_factor import (
    IMPLEMENTED_METRICS,
    _build_factor_composite_with_intermediates,
    _compute_panel_with_metrics,
    _get_metrics_for_factor,
)
from factor_library.data.export import _metric_display_name, _security_name_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Week 6 factor pipeline: load → compute → composite → report"
    )
    parser.add_argument("--factor", type=str, default=None)
    parser.add_argument("--universe", nargs="*", default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument(
        "--rank-method",
        choices=["percentile"],
        default="percentile",
        help="Week 6 standard fixes metric ranking to percentile.",
    )
    parser.add_argument(
        "--composite-method",
        choices=["equal_weight", "inverse_volatility"],
        default="equal_weight",
        help="Default equal_weight (course standard). inverse_volatility weights by 1/std of each rank column.",
    )
    parser.add_argument("--winsorize-bounds", nargs=2, type=float, default=[0.01, 0.99],
                        metavar=("LO", "HI"), help="Winsorize quantile bounds (default: 0.01 0.99)")
    parser.add_argument("--report", action="store_true", help="Generate report after pipeline")
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also export full factor panel as CSV per factor",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-precomputed-step2",
        action="store_true",
        help="Do not load final/app/data/*/step2_factors.parquet; use step1 or raw Excel+DB instead.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds = tuple(args.winsorize_bounds)

    factor_types = [args.factor] if args.factor else list_factor_types()
    universe = args.universe

    for factor_name in factor_types:
        try:
            implemented = _get_metrics_for_factor(factor_name, path=None)
        except (ValueError, NotImplementedError) as e:
            print(f"  Skip {factor_name}: {e}")
            continue

        implemented = [m for m in implemented if m in IMPLEMENTED_METRICS]
        if not implemented:
            print(f"  Skip {factor_name}: no implemented metrics")
            continue

        print(f"\n[{factor_name}] Loading panel...")
        df = load_merged_panel(
            universe=universe,
            start_date=args.start,
            end_date=args.end,
            factor_type=factor_name,
            use_precomputed_step2=not args.no_precomputed_step2,
            metric_keys=implemented,
        )
        if df.empty:
            print(f"  Skip {factor_name}: empty panel")
            continue

        print(f"  Rows: {len(df)}, date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Computing metrics: {implemented}")

        df = _compute_panel_with_metrics(
            df, implemented,
            winsorize=True,
            winsorize_bounds=bounds,
        )

        print(f"  Building composite (rank={args.rank_method}, composite={args.composite_method})...")
        factor_pct, work = _build_factor_composite_with_intermediates(
            df, implemented, args.rank_method, args.composite_method,
        )

        out = df[["fsym_primary_listing_id", "date"]].copy()
        out["factor_value"] = factor_pct
        out = out.dropna(subset=["factor_value"])
        if out.empty:
            print(f"  Skip {factor_name}: no valid factor_value (all NaN after composite)")
            continue

        safe_name = factor_name.replace(" ", "_").replace("/", "_")
        out_path = output_dir / f"{safe_name}_composite.parquet"
        out.to_parquet(out_path, index=False)
        print(f"  Saved: {out_path} ({len(out)} rows)")

        if args.export_csv:
            # Build debug-style wide table: Date, Security, Security_Name, metrics, metric_ranks, Factor_Score, Factor_Rank
            work = work.copy()
            work["date"] = pd.to_datetime(work["date"])
            work["_rank_pct"] = work["_factor_percentile"] * 100
            # Keep only rows with a valid factor percentile (align with parquet output)
            work = work[work["_factor_percentile"].notna()].reset_index(drop=True)

            # Raw metrics and rank columns actually present
            rankable = [m for m in implemented if m != "market_excess_return"]
            raw_cols: list[str] = [m for m in rankable if m in work.columns]
            rank_cols: list[str] = [f"_rank_{m}" for m in rankable if f"_rank_{m}" in work.columns]

            debug = work[["date", "fsym_primary_listing_id"]].copy()
            debug = debug.rename(columns={"fsym_primary_listing_id": "Security"})
            debug["Date"] = debug["date"].dt.strftime("%Y-%m-%d")
            debug = debug.drop(columns=["date"])

            name_map = _security_name_map()
            if not name_map.empty:
                debug = debug.merge(name_map, on="Security", how="left")
            else:
                debug["Security_Name"] = pd.NA

            # Raw metric columns
            for metric in raw_cols:
                debug[_metric_display_name(metric)] = work[metric].values

            # Rank columns
            for metric in [c.replace("_rank_", "") for c in rank_cols]:
                rcol = f"_rank_{metric}"
                debug[f"{_metric_display_name(metric)}_rank"] = work[rcol].values

            score_col = f"{safe_name}_Score"
            rank_col = f"{safe_name}_Rank"
            debug[score_col] = work["_factor_percentile"].values
            debug[rank_col] = work["_rank_pct"].round(0).astype("Int64").values

            raw_display_cols = [
                _metric_display_name(m)
                for m in raw_cols
                if _metric_display_name(m) in debug.columns
            ]
            rank_display_cols = [
                f"{_metric_display_name(m)}_rank"
                for m in [c.replace("_rank_", "") for c in rank_cols]
                if f"{_metric_display_name(m)}_rank" in debug.columns
            ]

            debug = debug[["Date", "Security", "Security_Name"] + raw_display_cols + rank_display_cols + [score_col, rank_col]]

            csv_path = output_dir / f"{safe_name}_composite.csv"
            debug.to_csv(csv_path, index=False, float_format="%.6f")
            print(f"  Saved CSV: {csv_path} ({len(debug)} rows, wide debug format)")

    print("\nPipeline complete.")

    if args.report:
        print("\n--report is deprecated. Use the Stage 2 entry scripts instead:")
        print("    python scripts/run_stage2a.py        # IC decay + tear sheets")
        print("    python scripts/run_stage2b.py        # cost, attribution, OOS")
        print("    python scripts/run_regime.py         # regime-conditional IC")
        print("    python scripts/run_quantiles.py      # quantile monotonicity")
        print("    python scripts/run_concentration.py  # concentration + LOO")
        print("    python scripts/run_vol_targeting.py  # vol-targeted overlay")
        print("    python scripts/run_ensemble.py       # multi-factor ensemble")


if __name__ == "__main__":
    main()
