#!/usr/bin/env python3
"""
Frame B "top-3 universe-aligned" ensemble + vol-target overlay.

Selection rationale (from slide 9 single-factor Sharpe + economic intuition):
- Size                   (Sharpe +1.10, literature direction)
- Vol (Low Risk) FLIPPED (Sharpe +0.73 in universe direction, literature inverted)
- Investment    FLIPPED  (Sharpe +0.33 in universe direction, literature inverted)

The 4-factor "dir-adjusted" path on slide 10 includes Value and Profitability (FF)
as well; here we keep only the top-3 by single-factor Sharpe + clear economic
intuition (mega-cap growth-tech regime: high-vol + high-growth + concentration
in the smaller-mega-cap survivors).

Outputs:
    runtime/output/ensemble/top3_universe_aligned_composite.parquet
    runtime/output/ensemble/top3_universe_aligned_summary.csv
    runtime/output/ensemble/top3_universe_aligned_comparison.png
    runtime/output/vol_targeting/Top3_universe_aligned_vol_target_stats.csv
    runtime/output/vol_targeting/Top3_universe_aligned_vol_targeting.png
"""
from __future__ import annotations

from pathlib import Path

import empyrical as emp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from factor_library.analysis.ic import build_price_panel
from factor_library.config import DATA_DIR, DEFAULT_OUTPUT_DIR
from factor_library.reports.tearsheets import long_short_daily_returns
from factor_library.viz.style import apply_style


COMPOSITES = {
    "Size": ("Size_composite.parquet", +1),  # literature direction
    "Vol (Low Risk) — flipped": ("Volatility___Low_Risk_composite.parquet", -1),
    "Investment — flipped": ("Investment_composite.parquet", -1),
}


def _max_drawdown(ret: pd.Series) -> float:
    wealth = (1 + ret).cumprod()
    dd = (wealth - wealth.cummax()) / wealth.cummax()
    return float(dd.min())


def _stats(ret: pd.Series, label: str, strategy_type: str = "ensemble") -> dict:
    ret = ret.dropna()
    if len(ret) == 0:
        return {"strategy": label, "n_days": 0}
    return {
        "strategy": label,
        "ann_return": float(emp.annual_return(ret)),
        "ann_vol": float(ret.std(ddof=1) * np.sqrt(252)),
        "sharpe": float(emp.sharpe_ratio(ret, annualization=252)),
        "sortino": float(emp.sortino_ratio(ret, annualization=252)),
        "max_drawdown": _max_drawdown(ret),
        "skew": float(stats.skew(ret)),
        "kurtosis": float(stats.kurtosis(ret) + 3),  # raw kurtosis (normal = 3)
        "n_days": int(len(ret)),
        "strategy_type": strategy_type,
    }


def _vol_target(ret: pd.Series, target_annual_vol: float, window: int = 21,
                max_leverage: float = 4.0) -> pd.Series:
    target_daily_vol = target_annual_vol / np.sqrt(252)
    rolling_vol = ret.rolling(window).std().shift(1)
    scale = (target_daily_vol / rolling_vol).clip(upper=max_leverage)
    return (ret * scale).dropna()


def main():
    out_root = DEFAULT_OUTPUT_DIR
    ens_dir = out_root / "ensemble"
    vt_dir = out_root / "vol_targeting"
    ens_dir.mkdir(parents=True, exist_ok=True)
    vt_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Load price panel + 3 composites…")
    extended = DATA_DIR / "step1_merged_data_extended.parquet"
    step1 = pd.read_parquet(extended if extended.exists() else DATA_DIR / "step1_merged_data.parquet")
    prices = build_price_panel(step1)

    # Build daily L/S for each factor with direction applied
    legs = {}
    for label, (fname, sign) in COMPOSITES.items():
        df = pd.read_parquet(out_root / fname)
        df["date"] = pd.to_datetime(df["date"])
        ls = long_short_daily_returns(df, prices)
        legs[label] = sign * ls
        print(f"    {label:32s}  N={len(ls):>5d}  Sharpe={emp.sharpe_ratio(legs[label], annualization=252):+.3f}")

    leg_df = pd.DataFrame(legs).sort_index()
    # Only count dates where ALL 3 legs have data — otherwise pre-1994 the
    # "ensemble" is really just Vol(LR) (the only price-based factor) with
    # other legs zero-filled, inflating both cumulative growth and the chart.
    leg_df_common = leg_df.dropna()
    common_start = leg_df_common.index.min()
    print(f"\n    Common-availability start: {common_start.date()} "
          f"(N={len(leg_df_common)} days; Vol/Size/Inv all present)")

    # Equal-weight ensemble of the 3 legs (common period only)
    ens = leg_df_common.mean(axis=1)
    print(f"[2/4] Top-3 ensemble (equal-weight)  Sharpe={emp.sharpe_ratio(ens, annualization=252):+.3f}")

    # Persist composite as a "synthetic factor" — wealth path
    wealth = (1 + ens).cumprod()
    out_df = pd.DataFrame({"date": ens.index, "long_short_ret": ens.values, "wealth": wealth.values})
    out_df.to_parquet(ens_dir / "top3_universe_aligned_composite.parquet", index=False)
    print(f"    -> {ens_dir / 'top3_universe_aligned_composite.parquet'}")

    # ---- Summary CSV ----
    rows = []
    rows.append(_stats(ens, "Top-3 (equal-weight) — raw"))
    for tgt in (0.10, 0.15, 0.20):
        ens_vt = _vol_target(ens, target_annual_vol=tgt)
        rows.append(_stats(ens_vt, f"Top-3 (equal-weight) — vol-target {int(tgt*100)}%",
                          strategy_type="vol_target"))

    # Also include single-factor reference rows (universe-aligned)
    for label, leg in legs.items():
        rows.append(_stats(leg, f"single: {label}", strategy_type="single_factor"))

    summary = pd.DataFrame(rows)
    summary_path = ens_dir / "top3_universe_aligned_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[3/4] Summary -> {summary_path}")
    print(summary.round(4).to_string(index=False))

    # ---- Chart 1: ensemble comparison (cumulative growth) ----
    # NOTE: benchmark overlay intentionally NOT included — the 100-stock EW
    # benchmark cumulative wealth is inflated by survivorship and unfair
    # apples-to-apples comparison vs a dollar-neutral L/S strategy. The
    # benchmark Sharpe comparison is acknowledged in slide 06 (regression).
    apply_style()
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    for label, leg in legs.items():
        leg_in = leg.loc[leg.index >= common_start]
        ax.plot((1 + leg_in).cumprod(), lw=1.0, alpha=0.55, label=label)
    ax.plot((1 + ens).cumprod(), lw=2.2, color="#1f7a3a",
            label="Top-3 ensemble (L/S, equal-weight)")
    ax.set_yscale("log")
    ax.set_title(f"Top-3 universe-aligned — cumulative wealth "
                 f"({common_start.year}–{ens.index.max().year}, log-scale)")
    ax.set_ylabel("Cumulative wealth (× initial, log scale)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    chart1 = ens_dir / "top3_universe_aligned_comparison.png"
    fig.savefig(chart1, dpi=110)
    plt.close(fig)
    print(f"    chart -> {chart1}")

    # ---- Chart 2: vol-target overlay at multiple target levels ----
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.plot((1 + ens).cumprod(), lw=1.6, color="#6b6b6b", label="Top-3 raw")
    vt_colors = {0.10: "#1f7a3a", 0.15: "#3b9c5b", 0.20: "#7fc090"}
    for tgt in (0.10, 0.15, 0.20):
        r = _vol_target(ens, target_annual_vol=tgt)
        ax.plot((1 + r).cumprod(), lw=1.8, color=vt_colors[tgt],
                label=f"Top-3 + {int(tgt*100)}% vol-target")
    ax.set_yscale("log")
    ax.set_title(f"Top-3 universe-aligned — raw vs vol-target overlays "
                 f"({common_start.year}–{ens.index.max().year})")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    chart2 = vt_dir / "Top3_universe_aligned_vol_targeting.png"
    fig.savefig(chart2, dpi=110)
    plt.close(fig)
    print(f"    chart -> {chart2}")

    # ---- Vol-target stats CSV (mirrors structure of other vol_target_stats.csv) ----
    vt_rows = []
    for tgt_label, tgt in (("raw", None), ("target 10%", 0.10), ("target 15%", 0.15), ("target 20%", 0.20)):
        r = ens if tgt is None else _vol_target(ens, target_annual_vol=tgt)
        s = _stats(r, tgt_label, strategy_type="ensemble")
        s["strategy_name"] = "Top-3 (Size + Vol-flipped + Inv-flipped)"
        vt_rows.append(s)
    vt_df = pd.DataFrame(vt_rows)[["strategy", "ann_return", "ann_vol", "sharpe", "sortino",
                                    "max_drawdown", "skew", "kurtosis", "strategy_name", "strategy_type"]]
    vt_path = vt_dir / "Top3_universe_aligned_vol_target_stats.csv"
    vt_df.to_csv(vt_path, index=False)
    print(f"\n[4/4] Vol-target stats -> {vt_path}")
    print(vt_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
