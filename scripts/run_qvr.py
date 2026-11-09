#!/usr/bin/env python3
"""
Quality × Value × short-term Reversal (QVR) pilot.

Sequential filter strategy inspired by Buffett + AQR:
  1. Universe filter #1: top Q1 % by Quality (Composite) factor_value
     → focus on financially healthy / high-profitability firms
  2. Universe filter #2: within that, keep top Q2 % by Value factor_value
     → among quality names, pick the cheap ones ("quality value")
  3. Signal: short-term reversal score = -past N-day return
     → among quality-value names, long the recent losers (likely rebound),
       short the recent winners (likely mean-revert)
  4. Portfolio: equal-weight long / short buckets, daily rebalance

Compares head-to-head against the simpler Value × Reversal pilot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import empyrical as emp
import seaborn as sns

from factor_library.config import (
    DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    STEP1_EXTENDED_PARQUET,
    STEP1_MERGED_PARQUET,
)
from factor_library.backtest.long_short_attribution import (
    compute_next_day_returns_from_prices,
)
from factor_library.viz.style import apply_style


def _load_factor(output_dir: Path, factor_file: str, rename_to: str) -> pd.DataFrame:
    p = output_dir / factor_file
    if not p.exists():
        raise FileNotFoundError(f"missing composite: {p}")
    df = pd.read_parquet(p)[["fsym_primary_listing_id", "date", "factor_value"]]
    df["date"] = pd.to_datetime(df["date"])
    return df.rename(columns={"factor_value": rename_to})


def _compute_past_return(prices: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    p = prices[["fsym_primary_listing_id", "date", "price"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["fsym_primary_listing_id", "date"])
    p[f"ret_{window}d_past"] = (
        p.groupby("fsym_primary_listing_id", sort=False)["price"]
         .pct_change(window).values
    )
    return p[["fsym_primary_listing_id", "date", f"ret_{window}d_past"]]


def _run_strategy(
    merged: pd.DataFrame,
    label: str,
    n_buckets: int = 5,
    min_cross_section: int = 10,
    leg_gross: float = 0.5,
) -> tuple[pd.Series, dict, pd.DataFrame]:
    """
    Given a merged frame with columns (fsym_primary_listing_id, date,
    reversal_score, ret_1d_fwd), build Q_n - Q_1 daily long/short return.
    Returns the daily series, summary stats dict, and the leg trace.
    """
    m = merged.copy()
    ranks = (m.groupby("date", sort=False)["reversal_score"]
              .rank(method="average"))
    n = m.groupby("date", sort=False)["reversal_score"].transform("count")
    m["n_cs"] = n.astype("Int64")
    m["bucket"] = (np.floor((ranks - 1) * n_buckets / n) + 1).astype("Int64")
    m = m[m["n_cs"] >= min_cross_section].copy()
    if m.empty:
        raise RuntimeError(f"{label}: no days pass min_cross_section after filters")

    # reversal_score = -past_return, so highest score = biggest past losers.
    # For a reversal strategy we should long the highest-score bucket and
    # short the lowest-score bucket.
    m["side"] = np.where(m["bucket"] == n_buckets, "long",
               np.where(m["bucket"] == 1, "short", "mid"))
    legs = m[m["side"].isin(["long", "short"])].copy()
    counts = (legs.groupby(["date", "side"], sort=False)["fsym_primary_listing_id"]
                  .transform("count").astype(float))
    w = leg_gross / counts
    legs["weight"] = np.where(legs["side"] == "long", w, -w)
    legs["contribution"] = legs["weight"] * legs["ret_1d_fwd"]

    strat = legs.groupby("date", sort=True).agg(
        long_short_ret=("contribution", "sum"),
        n_long=("side", lambda s: int((s == "long").sum())),
        n_short=("side", lambda s: int((s == "short").sum())),
        n_universe=("fsym_primary_listing_id", "count"),
    ).reset_index()
    r = strat.set_index("date")["long_short_ret"]

    # log-space max DD to avoid 45-year compound overflow
    log_eq = np.log1p(r).cumsum()
    max_dd = float((np.exp(log_eq - log_eq.cummax()) - 1).min())

    stats = {
        "strategy":    label,
        "n_days":      int(len(r)),
        "avg_universe": float(strat["n_universe"].mean()),
        "ann_return":  float(emp.annual_return(r)),
        "ann_vol":     float(r.std(ddof=1) * np.sqrt(252)),
        "sharpe":      float(emp.sharpe_ratio(r, annualization=252)),
        "sortino":     float(emp.sortino_ratio(r, annualization=252)),
        "max_drawdown": max_dd,
        "calmar":      float(emp.annual_return(r) / abs(max_dd)) if max_dd < 0 else np.nan,
        "win_rate":    float((r > 0).mean()),
        "skew":        float(r.skew()),
        "kurtosis":    float(r.kurtosis()),
    }
    return r, stats, strat


def _display_path(path: Path) -> Path:
    """Return a user-friendly path relative to project root when possible."""
    project_root = Path(__file__).resolve().parents[1]
    try:
        return path.resolve().relative_to(project_root)
    except ValueError:
        return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Quality × Value × short-term Reversal pilot")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory for pilot artefacts (default: <output-dir>/pilots/qvr).",
    )
    ap.add_argument("--quality-top-quantile", type=float, default=0.5,
                    help="Universe filter 1: keep top quantile by Quality (Composite)")
    ap.add_argument("--value-top-quantile", type=float, default=0.5,
                    help="Universe filter 2 (within quality): keep top quantile by Value")
    ap.add_argument("--reversal-window", type=int, default=21,
                    help="Past-return window for the reversal signal (days)")
    ap.add_argument("--n-buckets", type=int, default=5)
    ap.add_argument("--min-cross-section", type=int, default=10)
    args = ap.parse_args()

    out_dir = args.artifact_dir if args.artifact_dir is not None else (args.output_dir / "pilots" / "qvr")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading prices + Quality + Value composites…")
    step1_path = STEP1_EXTENDED_PARQUET if STEP1_EXTENDED_PARQUET.exists() else STEP1_MERGED_PARQUET
    prices = pd.read_parquet(step1_path)
    quality = _load_factor(args.output_dir, "Quality_(Composite)_composite.parquet", "Quality")
    value   = _load_factor(args.output_dir, "Value_composite.parquet", "Value")
    fwd     = compute_next_day_returns_from_prices(prices)
    past    = _compute_past_return(prices, window=args.reversal_window)

    print("[2/4] Applying sequential filters: Quality → Value → reversal signal…")
    m = (past.merge(fwd, on=["fsym_primary_listing_id", "date"], how="inner")
             .merge(quality, on=["fsym_primary_listing_id", "date"], how="inner")
             .merge(value, on=["fsym_primary_listing_id", "date"], how="inner"))
    past_col = f"ret_{args.reversal_window}d_past"
    m = m.dropna(subset=[past_col, "ret_1d_fwd", "Quality", "Value"])

    # Filter 1: top Q1% by Quality each day
    q1 = float(args.quality_top_quantile)
    m["q_rank"] = m.groupby("date", sort=False)["Quality"].rank(pct=True, method="average")
    qvr_universe = m[m["q_rank"] >= (1.0 - q1)].copy()
    print(f"    after Quality top {int(q1*100)}%: avg {qvr_universe.groupby('date').size().mean():.1f} stocks/day")

    # Filter 2: within quality-filtered universe, top Q2% by Value
    q2 = float(args.value_top_quantile)
    qvr_universe["v_rank"] = (qvr_universe.groupby("date", sort=False)["Value"]
                                           .rank(pct=True, method="average"))
    qvr_filtered = qvr_universe[qvr_universe["v_rank"] >= (1.0 - q2)].copy()
    print(f"    after Value top {int(q2*100)}% within quality: avg {qvr_filtered.groupby('date').size().mean():.1f} stocks/day")

    qvr_filtered["reversal_score"] = -qvr_filtered[past_col]

    print("[3/4] Building QVR + Value-only-reversal strategies…")
    # QVR
    qvr_r, qvr_stats, qvr_trace = _run_strategy(
        qvr_filtered[["fsym_primary_listing_id", "date", "reversal_score", "ret_1d_fwd"]],
        label="QVR (Quality×Value×Reversal)",
        n_buckets=args.n_buckets, min_cross_section=args.min_cross_section,
    )
    # Baseline: Value × Reversal (no Quality filter)
    value_only = m[m.groupby("date", sort=False)["Value"].rank(pct=True, method="average") >= (1.0 - q2)].copy()
    value_only["reversal_score"] = -value_only[past_col]
    vr_r, vr_stats, vr_trace = _run_strategy(
        value_only[["fsym_primary_listing_id", "date", "reversal_score", "ret_1d_fwd"]],
        label=f"Value×Reversal (top {int(q2*100)}% value, no quality filter)",
        n_buckets=args.n_buckets, min_cross_section=args.min_cross_section,
    )

    summary = pd.DataFrame([qvr_stats, vr_stats])
    summary.to_csv(out_dir / "qvr_summary.csv", index=False)
    qvr_trace.to_csv(out_dir / "qvr_strategy_daily.csv", index=False)
    vr_trace.to_csv(out_dir / "value_reversal_strategy_daily.csv", index=False)

    print("\n    summary:")
    print(summary.round(4).to_string(index=False))

    print("\n[4/4] Writing comparison chart…")
    apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 6.4),
                                    gridspec_kw={"height_ratios": [3, 1]})
    for name, r, colour in [("QVR", qvr_r, "#2874A6"),
                             ("Value×Reversal (no quality)", vr_r, "#B9770E")]:
        eq = (1 + r).cumprod()
        ax1.plot(eq.index, eq.values, color=colour, lw=1.5, label=name)
    ax1.axhline(1, color="grey", lw=0.7)
    ax1.set_yscale("log")
    ax1.set_title("QVR vs. Value × Reversal — growth of $1 (log, compound, gross)")
    ax1.set_ylabel("Growth of $1")
    ax1.legend(loc="upper left")
    ax1.grid(True, axis="y", which="both")

    # Underwater
    log_eq_qvr = np.log1p(qvr_r).cumsum()
    dd_qvr = np.exp(log_eq_qvr - log_eq_qvr.cummax()) - 1
    ax2.fill_between(dd_qvr.index, dd_qvr.values * 100, 0, color="#2874A6", alpha=0.35, label="QVR")
    log_eq_vr = np.log1p(vr_r).cumsum()
    dd_vr = np.exp(log_eq_vr - log_eq_vr.cummax()) - 1
    ax2.fill_between(dd_vr.index, dd_vr.values * 100, 0, color="#B9770E", alpha=0.35, label="V×R")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, axis="y")

    fig.tight_layout()
    chart_path = out_dir / "qvr_vs_value_reversal.png"
    fig.savefig(chart_path)
    plt.close(fig)
    print(f"    chart -> {_display_path(chart_path)}")


if __name__ == "__main__":
    main()
