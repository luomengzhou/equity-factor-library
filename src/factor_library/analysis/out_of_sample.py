"""
Out-of-sample (train / test) factor validation.

A hiring PM's first question after seeing a factor's Sharpe is "is it
the same out-of-sample?" We split the date range at ``split_date`` and
report IC, IR, Sharpe, and ann-return on both sides so the degradation
(or stability) is visible at a glance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import empyrical as emp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from factor_library.analysis.ic import (
    compute_ic,
    ic_summary,
    prepare_alphalens_input,
)
from factor_library.reports.tearsheets import long_short_daily_returns
from factor_library.viz.style import apply_style, colour_for


DEFAULT_MAX_LOSS = 0.50


def _split(df: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.to_datetime(split_date)
    return df[df["date"] < d], df[df["date"] >= d]


def oos_validation(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    factor_name: str,
    split_date: str = "2020-01-01",
    horizons: Iterable[int] = (1, 5, 20, 60),
    quantiles: int = 5,
    max_loss: float = DEFAULT_MAX_LOSS,
) -> pd.DataFrame:
    """
    Return a tidy table comparing in-sample vs out-of-sample performance.

    Columns: split, horizon, n_obs, ic_mean, ic_ir, t_stat,
             ls_sharpe, ls_ann_return, ls_max_dd.

    ``split`` is 'in_sample' or 'out_of_sample'; the split_date itself
    belongs to out_of_sample.
    """
    split_label = str(pd.to_datetime(split_date).date())
    is_df, oos_df = _split(factor_df, split_date)
    out = []
    for label, sub in (("in_sample", is_df), ("out_of_sample", oos_df)):
        if sub.empty:
            continue
        merged = prepare_alphalens_input(sub, prices, horizons=tuple(horizons),
                                         quantiles=quantiles, max_loss=max_loss)
        ic_ts = compute_ic(merged)
        summ = ic_summary(ic_ts)
        ls = long_short_daily_returns(sub, prices, quantiles=quantiles)
        ls_sharpe = float(emp.sharpe_ratio(ls, annualization=252))
        ls_ann = float(emp.annual_return(ls))
        ls_mdd = float(emp.max_drawdown((1 + ls).cumprod()))
        for h, row in summ.iterrows():
            out.append({
                "factor":        factor_name,
                "split_date":    split_label,
                "split":         label,
                "horizon":       h,
                "n_obs":         int(row["n_obs"]),
                "ic_mean":       float(row["ic_mean"]),
                "ic_ir":         float(row["ic_ir"]),
                "t_stat":        float(row["t_stat"]),
                "ls_sharpe":     ls_sharpe,
                "ls_ann_return": ls_ann,
                "ls_max_dd":     ls_mdd,
            })
    return pd.DataFrame(out)


def plot_oos_ic_comparison(
    table: pd.DataFrame,
    factor_name: str,
    out_path: Path | str,
    split_date: str | None = None,
) -> Path:
    """
    Side-by-side IC (mean ± IR) comparison at each horizon. In-sample
    and out-of-sample bars per horizon make degradation visually obvious.
    """
    apply_style()
    sub = table[table["factor"] == factor_name].copy()
    colour = colour_for(factor_name)
    horizons = sub["horizon"].unique().tolist()
    splits = ["in_sample", "out_of_sample"]
    width = 0.38
    x = np.arange(len(horizons))
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for i, split in enumerate(splits):
        vals = [sub[(sub["horizon"] == h) & (sub["split"] == split)]["ic_mean"].squeeze()
                for h in horizons]
        irs = [sub[(sub["horizon"] == h) & (sub["split"] == split)]["ic_ir"].squeeze()
               for h in horizons]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      color=colour, alpha=0.85 if split == "in_sample" else 0.45,
                      label=split)
        for b, ir in zip(bars, irs):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"IR {ir:.2f}",
                    ha="center", va="bottom" if b.get_height() >= 0 else "top",
                    fontsize=8, color="#333")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_xlabel("Forward-return horizon")
    ax.set_ylabel("Mean Spearman IC")
    if split_date is None:
        ax.set_title(f"{factor_name} — in-sample vs out-of-sample IC")
    else:
        ax.set_title(f"{factor_name} — in-sample vs out-of-sample IC (split {split_date})")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
