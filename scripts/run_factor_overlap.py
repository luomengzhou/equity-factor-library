#!/usr/bin/env python3
"""
Step 3 — cross-factor stock-overlap diagnostic.

For every factor in the ensemble, take the long leg (top-quintile by
percentile-ranked factor_value) on each date, then compute the Jaccard
overlap against every other factor's long leg on the same date. Average
the daily Jaccard over time to summarise how much each pair shares the
same names.

Motivated by the concern that ostensibly
diversified factor ensembles can collapse onto the same underlying
stocks — sector / name concentration the per-factor concentration
analysis (run_concentration.py) does not surface.

Outputs under runtime/output/factor_overlap/:
  pairwise_jaccard.csv          N×N symmetric matrix of mean daily Jaccard
  pairwise_jaccard.png          annotated heatmap
  long_leg_concentration.csv    per-factor: avg long-leg size, top-5 most
                                 frequent ids and how often they appear
  pair_jaccard_timeseries.csv   long-form (factor1, factor2, date, jaccard)
                                 — useful for spotting regime-dependent
                                 collapse e.g. "all factors crowded into
                                 mega-caps after 2020"
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factor_library.config import DEFAULT_OUTPUT_DIR
from factor_library.viz.style import SEQUENTIAL_CMAP, apply_style


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

# Same 11-factor universe as scripts/run_ensemble.py. Low Beta excluded
# because it mirrors Market (joint-regression R²=1).
DEFAULT_FACTORS = [
    "Momentum", "Volatility / Low Risk", "Investment",
    "Value", "Quality", "Quality (Composite)",
    "Yield", "Size", "Profitability (FF)",
    "Market", "Liquidity",
]

ID_COL = "fsym_primary_listing_id"
DATE_COL = "date"


def long_leg_membership(df: pd.DataFrame, factor_name: str,
                        quantile: float) -> pd.DataFrame:
    """Rows belonging to the top `quantile` slice of factor_value per date."""
    f = df[[ID_COL, DATE_COL, "factor_value"]].dropna(subset=["factor_value"]).copy()
    f[DATE_COL] = pd.to_datetime(f[DATE_COL])
    threshold = f.groupby(DATE_COL)["factor_value"].transform(
        lambda s: s.quantile(1.0 - quantile)
    )
    keep = f[f["factor_value"] >= threshold].copy()
    keep["factor"] = factor_name
    return keep[[DATE_COL, ID_COL, "factor"]]


def daily_sets(membership: pd.DataFrame) -> dict[pd.Timestamp, set[str]]:
    return {d: set(g[ID_COL].astype(str))
            for d, g in membership.groupby(DATE_COL, sort=False)}


def pairwise_jaccard(
    sets_a: dict[pd.Timestamp, set[str]],
    sets_b: dict[pd.Timestamp, set[str]],
) -> pd.Series:
    common = sets_a.keys() & sets_b.keys()
    rows = []
    for d in common:
        a = sets_a[d]
        b = sets_b[d]
        union = a | b
        if not union:
            continue
        rows.append((d, len(a & b) / len(union)))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(dict(rows), name="jaccard").sort_index()
    s.index.name = DATE_COL
    return s


def plot_heatmap(jacc: pd.DataFrame, out_path: Path) -> Path:
    apply_style()
    n = len(jacc)
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * n + 2), max(7, 0.85 * n + 1.5)))
    im = ax.imshow(jacc.values, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(jacc.columns, rotation=40, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(jacc.index)
    for i in range(n):
        for j in range(n):
            v = jacc.iloc[i, j]
            if pd.isna(v):
                continue
            color = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Jaccard (long-leg overlap, time-avg)")
    ax.set_title("Cross-factor long-leg stock overlap — top-quintile, time-avg Jaccard")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-factor long-leg stock overlap (Jaccard, time-averaged)"
    )
    ap.add_argument("--factors", nargs="+", default=DEFAULT_FACTORS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--quantile", type=float, default=0.20,
                    help="Top fraction defining the long leg (default 0.20)")
    args = ap.parse_args()

    out = args.output_dir / "factor_overlap"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Load factor composites + extract top-{args.quantile:.0%} long legs…")
    long_legs: dict[str, pd.DataFrame] = {}
    for n in args.factors:
        fn = FACTOR_FILES.get(n)
        if not fn:
            print(f"    [skip] {n}: not in FACTOR_FILES")
            continue
        fp = args.output_dir / fn
        if not fp.exists():
            print(f"    [skip] {n}: {fp.name} missing")
            continue
        df = pd.read_parquet(fp)
        ll = long_leg_membership(df, factor_name=n, quantile=args.quantile)
        long_legs[n] = ll
        avg = ll.groupby(DATE_COL).size().mean()
        print(f"    {n}: avg long-leg size = {avg:.1f}, "
              f"unique ids = {ll[ID_COL].nunique()}, dates = {ll[DATE_COL].nunique()}")

    factors = list(long_legs)

    print(f"\n[2/3] Compute pairwise Jaccard across {len(factors)} factors…")
    sets_by_factor = {f: daily_sets(long_legs[f]) for f in factors}

    jacc = pd.DataFrame(index=factors, columns=factors, dtype=float)
    for f in factors:
        jacc.loc[f, f] = 1.0

    ts_rows = []
    for f1, f2 in combinations(factors, 2):
        s = pairwise_jaccard(sets_by_factor[f1], sets_by_factor[f2])
        if s.empty:
            mean = np.nan
        else:
            mean = float(s.mean())
            for d, v in s.items():
                ts_rows.append({"factor1": f1, "factor2": f2,
                                "date": d, "jaccard": v})
        jacc.loc[f1, f2] = mean
        jacc.loc[f2, f1] = mean

    jacc = jacc.astype(float)
    jacc.to_csv(out / "pairwise_jaccard.csv")
    pd.DataFrame(ts_rows).to_csv(out / "pair_jaccard_timeseries.csv", index=False)
    print(jacc.round(3).to_string())

    # Surface the 5 most overlapping pairs (off-diagonal)
    triu = jacc.where(np.triu(np.ones(jacc.shape, dtype=bool), k=1))
    flat = triu.stack().sort_values(ascending=False)
    print("\n    top-5 most overlapping pairs:")
    for (f1, f2), v in flat.head(5).items():
        print(f"      {f1:24s} ↔ {f2:24s}  Jaccard = {v:.3f}")
    print("\n    bottom-5 least overlapping pairs:")
    for (f1, f2), v in flat.tail(5).items():
        print(f"      {f1:24s} ↔ {f2:24s}  Jaccard = {v:.3f}")

    print(f"\n[3/3] Heatmap + per-factor long-leg concentration table…")
    plot_heatmap(jacc, out / "pairwise_jaccard.png")

    rows = []
    for f in factors:
        ll = long_legs[f]
        n_dates = ll[DATE_COL].nunique()
        freq = (ll.groupby(ID_COL).size().sort_values(ascending=False))
        top5 = freq.head(5)
        rows.append({
            "factor":              f,
            "avg_long_leg_size":   round(ll.groupby(DATE_COL).size().mean(), 1),
            "unique_long_leg_ids": int(ll[ID_COL].nunique()),
            "n_dates":             int(n_dates),
            "top5_ids":            "|".join(top5.index.astype(str)),
            "top5_pct_dates":      "|".join(f"{v / n_dates * 100:.0f}%"
                                            for v in top5.values),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "long_leg_concentration.csv", index=False)
    print(summary.to_string(index=False))

    print(f"\n  outputs -> {out.relative_to(args.output_dir.parent.parent)}")


if __name__ == "__main__":
    main()
