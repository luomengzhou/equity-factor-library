#!/usr/bin/env python3
"""
Style × Reversal pilot — seven variants.

Tests whether adding (a) a flag-pattern technical confirmation, (b) a
Quality tilt (replacing Value), and (c) a bull/bear regime gate can
turn the flat Value x Reversal pilot into something useful.

Flag-filter follows the flag-pattern framework: a pullback is treated
as a bounce-ready retracement only if it is shallow (<=61.8% off the
60-day range) and happens from a steep prior advance (>=15% flagpole).
Volume contraction is part of the textbook rule but the panel has no
volume column, so only the price-based conditions are enforced.

Variants
--------
V0  Value × Reversal                              (baseline, existing pilot)
V1  Value × Reversal + flag-filter
V2  Quality × Reversal
V3  Quality × Reversal + flag-filter
V4  Quality × Reversal + bull regime
V5  Q×R + flag-filter + bull regime
V6  Quality × Reversal + bear regime               (Quality's edge is crisis-era)

Daily rebalance, quintile L/S on the reversal score within the filtered
sub-universe. Equal-weight within each leg, 0.5 L1 per leg. Gross of cost.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import empyrical as emp

from factor_library.config import (
    DEFAULT_OUTPUT_DIR,
    STEP1_EXTENDED_PARQUET,
    STEP1_MERGED_PARQUET,
)
from factor_library.backtest.long_short_attribution import (
    compute_next_day_returns_from_prices,
)
from factor_library.viz.style import apply_style


def _display_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[1]
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def _load_factor(output_dir: Path, factor_file: str, rename_to: str) -> pd.DataFrame:
    p = output_dir / factor_file
    if not p.exists():
        raise FileNotFoundError(f"missing composite: {p}")
    df = pd.read_parquet(p)[["fsym_primary_listing_id", "date", "factor_value"]]
    df["date"] = pd.to_datetime(df["date"])
    return df.rename(columns={"factor_value": rename_to})


def _load_regime(output_dir: Path) -> pd.DataFrame:
    p = output_dir / "regime" / "regime_labels.parquet"
    if not p.exists():
        raise FileNotFoundError(f"missing regime labels: {p} — run scripts/run_regime.py first")
    r = pd.read_parquet(p)
    r["date"] = pd.to_datetime(r["date"])
    return r[["date", "bullbear"]]


def _compute_flag_features(
    prices: pd.DataFrame,
    flagpole_window: int = 60,
) -> pd.DataFrame:
    """
    Per (stock, date):
      retrace_pct   = (high_60d - close) / (high_60d - low_60d).    [0,1]
      flagpole_pct  = (high_60d - low_60d) / low_60d.                prior amplitude
      valid_flag    = (retrace_pct <= 0.618) & (flagpole_pct >= 0.15)
    """
    p = prices[["fsym_primary_listing_id", "date", "price"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["fsym_primary_listing_id", "date"])

    g = p.groupby("fsym_primary_listing_id", sort=False)
    high = g["price"].transform(lambda s: s.rolling(flagpole_window, min_periods=flagpole_window).max())
    low  = g["price"].transform(lambda s: s.rolling(flagpole_window, min_periods=flagpole_window).min())
    rng = (high - low).replace(0, np.nan)
    p["retrace_pct"]  = (high - p["price"]) / rng
    p["flagpole_pct"] = (high - low) / low.replace(0, np.nan)
    p["valid_flag"]   = ((p["retrace_pct"] <= 0.618) & (p["flagpole_pct"] >= 0.15)).astype("Int8")
    return p[["fsym_primary_listing_id", "date", "retrace_pct", "flagpole_pct", "valid_flag"]]


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
    m: pd.DataFrame,
    label: str,
    n_buckets: int = 5,
    min_cross_section: int = 10,
    leg_gross: float = 0.5,
) -> tuple[pd.Series, dict]:
    m = m.copy()
    ranks = m.groupby("date", sort=False)["reversal_score"].rank(method="average")
    n = m.groupby("date", sort=False)["reversal_score"].transform("count")
    m["n_cs"] = n.astype("Int64")
    m["bucket"] = (np.floor((ranks - 1) * n_buckets / n) + 1).astype("Int64")
    m = m[m["n_cs"] >= min_cross_section].copy()
    if m.empty:
        return pd.Series(dtype=float), {
            "strategy": label, "n_days": 0, "avg_universe": 0.0,
            "ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
            "sortino": np.nan, "max_drawdown": np.nan, "calmar": np.nan,
            "win_rate": np.nan, "skew": np.nan, "kurtosis": np.nan,
        }

    # reversal_score = -past_return, so highest score = biggest past losers.
    # Reversal trade: long highest-score bucket, short lowest-score bucket.
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
        n_universe=("fsym_primary_listing_id", "count"),
    ).reset_index()
    r = strat.set_index("date")["long_short_ret"]
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
    return r, stats


def main() -> None:
    out_dir = DEFAULT_OUTPUT_DIR / "pilots" / "style_reversal"
    out_dir.mkdir(parents=True, exist_ok=True)

    reversal_window = 21
    style_top_quantile = 0.5

    print("[1/4] Loading panel + Quality + Value composites + regime labels…")
    step1 = STEP1_EXTENDED_PARQUET if STEP1_EXTENDED_PARQUET.exists() else STEP1_MERGED_PARQUET
    prices = pd.read_parquet(step1)
    quality = _load_factor(DEFAULT_OUTPUT_DIR, "Quality_(Composite)_composite.parquet", "Quality")
    value   = _load_factor(DEFAULT_OUTPUT_DIR, "Value_composite.parquet",               "Value")
    regime  = _load_regime(DEFAULT_OUTPUT_DIR)
    fwd     = compute_next_day_returns_from_prices(prices)
    past    = _compute_past_return(prices, window=reversal_window)
    flag    = _compute_flag_features(prices)

    past_col = f"ret_{reversal_window}d_past"
    base = (past.merge(fwd,     on=["fsym_primary_listing_id", "date"], how="inner")
                .merge(quality, on=["fsym_primary_listing_id", "date"], how="inner")
                .merge(value,   on=["fsym_primary_listing_id", "date"], how="inner")
                .merge(flag,    on=["fsym_primary_listing_id", "date"], how="inner")
                .merge(regime,  on=["date"], how="left"))
    base = base.dropna(subset=[past_col, "ret_1d_fwd", "Quality", "Value"])
    base["reversal_score"] = -base[past_col]

    print("[2/4] Building style-filtered sub-universes…")
    def _top_style(df: pd.DataFrame, col: str, q: float) -> pd.DataFrame:
        rank = df.groupby("date", sort=False)[col].rank(pct=True, method="average")
        return df[rank >= (1.0 - q)].copy()

    value_u   = _top_style(base, "Value",   style_top_quantile)
    quality_u = _top_style(base, "Quality", style_top_quantile)
    print(f"    top-{int(style_top_quantile*100)}% Value   universe: avg {value_u.groupby('date').size().mean():.1f}/day")
    print(f"    top-{int(style_top_quantile*100)}% Quality universe: avg {quality_u.groupby('date').size().mean():.1f}/day")

    variants: list[tuple[str, pd.DataFrame]] = [
        ("V0 Value×Reversal",                        value_u),
        ("V1 Value×Reversal + flag-filter",          value_u[value_u["valid_flag"] == 1]),
        ("V2 Quality×Reversal",                      quality_u),
        ("V3 Quality×Reversal + flag-filter",        quality_u[quality_u["valid_flag"] == 1]),
        ("V4 Quality×Reversal + bull regime",        quality_u[quality_u["bullbear"] == "bull"]),
        ("V5 Q×R + flag-filter + bull regime",       quality_u[(quality_u["valid_flag"] == 1) & (quality_u["bullbear"] == "bull")]),
        ("V6 Quality×Reversal + bear regime",        quality_u[quality_u["bullbear"] == "bear"]),
    ]

    print("\n[3/4] Running 7 variants…")
    returns: dict[str, pd.Series] = {}
    stats_rows: list[dict] = []
    for label, frame in variants:
        cols = ["fsym_primary_listing_id", "date", "reversal_score", "ret_1d_fwd"]
        r, s = _run_strategy(frame[cols], label=label)
        returns[label] = r
        stats_rows.append(s)
        print(f"    {label:<50s} n={s['n_days']:>5d}  avg_u={s['avg_universe']:>5.1f}  "
              f"Sharpe={s['sharpe']:>6.2f}  ann={s['ann_return']*100:>6.2f}%  DD={s['max_drawdown']*100:>6.1f}%")

    summary = pd.DataFrame(stats_rows)
    summary.to_csv(out_dir / "style_reversal_summary.csv", index=False)
    for label, r in returns.items():
        if not r.empty:
            slug = label.split()[0].lower()
            r.to_frame("ret").to_csv(out_dir / f"daily_{slug}.csv")

    print("\n[4/4] Writing comparison chart…")
    apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 7),
                                    gridspec_kw={"height_ratios": [3, 1]})
    palette = ["#B9770E", "#D35400", "#2874A6", "#1F618D", "#117A65", "#0E6251", "#7B241C"]
    for (label, r), colour in zip(returns.items(), palette):
        if r.empty:
            continue
        eq = (1 + r).cumprod()
        ax1.plot(eq.index, eq.values, color=colour, lw=1.3, label=label)
    ax1.axhline(1, color="grey", lw=0.7)
    ax1.set_yscale("log")
    ax1.set_title("Value/Quality × Reversal — flag-filter and regime-gate comparison (gross of cost)")
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, axis="y", which="both")

    for (label, r), colour in zip(returns.items(), palette):
        if r.empty:
            continue
        log_eq = np.log1p(r).cumsum()
        dd = np.exp(log_eq - log_eq.cummax()) - 1
        ax2.plot(dd.index, dd.values * 100, color=colour, lw=0.9, alpha=0.75)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, axis="y")

    fig.tight_layout()
    chart = out_dir / "style_reversal_comparison.png"
    fig.savefig(chart)
    plt.close(fig)
    print(f"    chart   -> {_display_path(chart)}")
    print(f"    summary -> {_display_path(out_dir / 'style_reversal_summary.csv')}")


if __name__ == "__main__":
    main()
