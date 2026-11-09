#!/usr/bin/env python3
"""
Pilot strategy: short-term reversal within a Value universe.

Implements a minimal, industry-style experiment aligned with the mentor discussion:
- Universe filter: keep the top X% by Value factor_value (cheap stocks).
- Signal: short-term reversal = -past 21-trading-day return.
- Portfolio: equal-weight long top bucket (most negative past return), short bottom bucket.
- Apply weights at t to next-day return to avoid look-ahead.

Outputs under:
  final/app/runtime/output/pilots/value_reversal/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from factor_library.config import (
    DEFAULT_OUTPUT_DIR,
    STEP1_EXTENDED_PARQUET,
    STEP1_MERGED_PARQUET,
)
from factor_library.backtest.long_short_attribution import (
    compute_next_day_returns_from_prices,
)


def _compute_past_return(prices: pd.DataFrame, *, window: int = 21) -> pd.DataFrame:
    p = prices[["fsym_primary_listing_id", "date", "price"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["fsym_primary_listing_id", "date"])
    g = p.groupby("fsym_primary_listing_id", sort=False)["price"]
    p[f"ret_{window}d_past"] = g.pct_change(window).values
    return p[["fsym_primary_listing_id", "date", f"ret_{window}d_past"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Value × short-term reversal pilot.")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory with *_composite.parquet.")
    default_step1 = STEP1_EXTENDED_PARQUET if STEP1_EXTENDED_PARQUET.exists() else STEP1_MERGED_PARQUET
    ap.add_argument("--step1", type=Path, default=default_step1,
                    help="Parquet with prices (defaults to extended panel if present).")
    ap.add_argument("--value-top-quantile", type=float, default=0.3, help="Universe filter: keep top quantile by Value.")
    ap.add_argument("--reversal-window", type=int, default=21, help="Past return window for reversal (default 21d).")
    ap.add_argument("--n-buckets", type=int, default=5)
    ap.add_argument("--min-cross-section", type=int, default=30)
    ap.add_argument("--leg-gross", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.output_dir) / "pilots" / "value_reversal"
    charts_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    value_pq = Path(args.output_dir) / "Value_composite.parquet"
    if not value_pq.is_file():
        raise FileNotFoundError(f"Missing Value composite: {value_pq}")
    value = pd.read_parquet(value_pq)[["fsym_primary_listing_id", "date", "factor_value"]].copy()
    value["date"] = pd.to_datetime(value["date"])
    value = value.rename(columns={"factor_value": "Value"})

    prices = pd.read_parquet(Path(args.step1))
    fwd = compute_next_day_returns_from_prices(prices)
    past = _compute_past_return(prices, window=int(args.reversal_window))
    m = value.merge(past, on=["fsym_primary_listing_id", "date"], how="inner").merge(
        fwd, on=["fsym_primary_listing_id", "date"], how="inner"
    )
    m = m.dropna(subset=["Value", f"ret_{int(args.reversal_window)}d_past", "ret_1d_fwd"]).copy()
    if m.empty:
        raise RuntimeError("Empty merge; check data availability.")

    # Universe: top quantile by Value per date
    q = float(args.value_top_quantile)
    if not (0.0 < q < 1.0):
        raise ValueError("--value-top-quantile must be in (0,1).")
    m["value_rank"] = m.groupby("date", sort=False)["Value"].rank(method="average", pct=True)
    uni = m[m["value_rank"] >= (1.0 - q)].copy()

    # Signal: reversal score = - past return
    past_col = f"ret_{int(args.reversal_window)}d_past"
    uni["reversal_score"] = -uni[past_col]

    # Per-date buckets on reversal_score (within Value universe)
    s = uni["reversal_score"]
    ranks = s.groupby(uni["date"], sort=False).rank(method="average")
    n = s.groupby(uni["date"], sort=False).transform("count")
    uni["n_cs"] = n.astype("Int64")
    nb = int(args.n_buckets)
    uni["bucket"] = (np.floor((ranks - 1) * nb / n) + 1).astype("Int64")

    # Filter thin days
    uni = uni[uni["n_cs"] >= int(args.min_cross_section)].copy()
    if uni.empty:
        raise RuntimeError("No days pass min-cross-section after Value filter.")
    universe_counts = uni.groupby("date", sort=True)["fsym_primary_listing_id"].count()

    top_b = nb
    bot_b = 1
    uni["side"] = np.where(uni["bucket"] == top_b, "long", np.where(uni["bucket"] == bot_b, "short", "mid"))
    legs = uni[uni["side"].isin(["long", "short"])].copy()

    # Equal weights within leg/date; gross per leg
    counts = legs.groupby(["date", "side"], sort=False)["fsym_primary_listing_id"].transform("count").astype(float)
    w = float(args.leg_gross) / counts
    legs["weight"] = np.where(legs["side"] == "long", w, -w)
    legs["contribution"] = legs["weight"] * legs["ret_1d_fwd"]

    # Daily strategy return
    strat = legs.groupby("date", sort=True).agg(
        long_short_ret=("contribution", "sum"),
        n_long=("side", lambda s2: int((s2 == "long").sum())),
        n_short=("side", lambda s2: int((s2 == "short").sum())),
    ).reset_index()
    strat["n_universe"] = strat["date"].map(universe_counts).astype("Int64")
    strat["cumsum"] = strat["long_short_ret"].cumsum()
    strat.to_csv(out_dir / "strategy_daily.csv", index=False)

    # Bucket panel (mean forward return by bucket)
    panel = (
        uni.groupby(["date", "bucket"], sort=True)["ret_1d_fwd"]
        .agg(mean_forward_ret="mean", n="count")
        .reset_index()
    )
    panel.to_csv(out_dir / "bucket_panel.csv", index=False)

    # Compact chart: bucket cumsums + long-short
    try:
        piv = panel.pivot_table(index="date", columns="bucket", values="mean_forward_ret", aggfunc="mean").sort_index()
        cs = piv.cumsum()
        fig, ax = plt.subplots(figsize=(10.5, 4.85))
        buckets_sorted = sorted([c for c in cs.columns if pd.notna(c)])
        blues = plt.cm.Blues(np.linspace(0.42, 0.92, max(1, len(buckets_sorted))))
        for i, b in enumerate(buckets_sorted):
            ax.plot(cs.index.values, cs[b].values, linewidth=1.35, label=f"bucket {int(b)}", color=blues[i])
        ax.axhline(0, color="0.25", linewidth=0.85)
        ax.set_xlabel("date", fontsize=10)
        ax.set_ylabel("cumsum(mean next-day return)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle=(0, (1, 2.5)), alpha=0.5, linewidth=0.85, color="0.5")

        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.plot(strat["date"].values, strat["cumsum"].values, color="#e76f51", linewidth=2.0, label="long–short cumsum")
        ax2.set_ylabel("cumsum(long–short)", fontsize=10, color="#c45c3e")
        ax2.tick_params(axis="y", labelcolor="#8b4518")
        ax2.spines["right"].set_color("0.55")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, frameon=False, ncol=3, loc="upper left", fontsize=8)
        ax.set_title(
            f"Value(top {q:.0%}) × short-term reversal (past {int(args.reversal_window)}d), next-day returns",
            fontsize=10,
            pad=8,
        )
        fig.tight_layout()
        fig.savefig(charts_dir / "portfolio_compact_value_reversal.png", dpi=165, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


if __name__ == "__main__":
    main()

