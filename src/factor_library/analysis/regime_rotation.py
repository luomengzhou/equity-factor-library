"""
Regime-driven factor rotation.

Mentor's "next step after we complete all the factors": rotate between
factors based on the current market regime. Builds on the regime
classification in ``regime/detection.py`` and the per-factor long/short
daily returns from ``reports.tearsheets``.

Design (walk-forward, no look-ahead):
    1. For each factor, build the L/S daily return series.
    2. Classify every day into a regime bucket (bull/bear, low/high vol,
       or trend/chop).
    3. On each day t, look at the past ``lookback_days`` of realised
       regime-conditional IR (mean / std) for every factor,
       restricted to days whose regime label matched regime(t-1).
    4. Allocate weight ∝ max(IR, 0) across factors; normalise to sum 1.
    5. Today's rotated-ensemble return is that weighted combination of
       today's factor L/S returns.

Compared against: equal-weight ensemble (flat regime) and the strongest
single factor (Vol / Low Risk).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import empyrical as emp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning, module="alphalens")

from factor_library.reports.tearsheets import long_short_daily_returns
from factor_library.viz.style import apply_style


RegimeDimension = Literal["bullbear", "vol_regime", "trend_chop"]


def build_factor_return_frame(
    factor_composites: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Wide DataFrame: date × factor name, values = daily L/S return."""
    cols = {}
    for name, df in factor_composites.items():
        cols[name] = long_short_daily_returns(df, prices)
    return pd.concat(cols, axis=1).dropna(how="all")


def regime_rotation_returns(
    factor_returns: pd.DataFrame,
    regime_labels: pd.DataFrame,
    dimension: RegimeDimension = "vol_regime",
    lookback_days: int = 252,
    min_regime_days: int = 30,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Walk-forward regime-conditional IR-weighted rotation.

    Parameters
    ----------
    factor_returns : wide DataFrame, index=date, columns=factor names,
                     values=daily L/S return.
    regime_labels  : DataFrame with a 'date' column + one or more regime
                     columns (bullbear / vol_regime / trend_chop).
    dimension      : which regime column to rotate on.

    Returns
    -------
    rotated_return : daily return series of the rotation strategy.
    weights        : per-date per-factor weight matrix actually applied.
    """
    rl = regime_labels.set_index("date")[dimension]
    # shift by one day: today's weight uses yesterday's known regime
    rl_aligned = rl.reindex(factor_returns.index, method="ffill").shift(1)

    weights = pd.DataFrame(
        0.0, index=factor_returns.index, columns=factor_returns.columns
    )

    for i in range(lookback_days, len(factor_returns)):
        current = rl_aligned.iloc[i]
        if pd.isna(current):
            continue
        window_rets = factor_returns.iloc[i - lookback_days: i]
        window_regime = rl_aligned.iloc[i - lookback_days: i]
        mask = window_regime == current
        if mask.sum() < min_regime_days:
            continue
        sub = window_rets.loc[mask.values]
        mean = sub.mean()
        std = sub.std(ddof=1)
        ir = (mean / std).where(std > 0, 0.0)
        w = ir.clip(lower=0.0)
        total = w.sum()
        if total > 0:
            weights.iloc[i] = (w / total).values

    # dot product: today's return applied at today's weight (weight was set
    # from data up to yesterday, so no look-ahead)
    rotated = (factor_returns * weights).sum(axis=1)
    rotated = rotated[(weights.sum(axis=1) > 0).values]
    rotated.name = f"rotation_{dimension}"
    return rotated, weights


def equal_weight_ensemble(factor_returns: pd.DataFrame) -> pd.Series:
    n = len(factor_returns.columns)
    return (factor_returns.sum(axis=1) / n).rename("equal_weight_ensemble")


def _log_max_drawdown(ret: pd.Series) -> float:
    log_eq = np.log1p(ret).cumsum()
    return float((np.exp(log_eq - log_eq.cummax()) - 1).min())


def summarise_strategies(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, r in series_dict.items():
        r = r.dropna()
        if r.empty:
            continue
        ann = float(emp.annual_return(r))
        mdd = _log_max_drawdown(r)
        rows.append({
            "strategy":     name,
            "n_days":       int(len(r)),
            "ann_return":   ann,
            "ann_vol":      float(r.std(ddof=1) * np.sqrt(252)),
            "sharpe":       float(emp.sharpe_ratio(r, annualization=252)),
            "sortino":      float(emp.sortino_ratio(r, annualization=252)),
            "max_drawdown": mdd,
            "calmar":       float(ann / abs(mdd)) if mdd < 0 else np.nan,
        })
    return pd.DataFrame(rows).set_index("strategy")


def plot_rotation_comparison(
    series_dict: dict[str, pd.Series],
    out_path: Path | str,
    title: str,
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    palette = sns.color_palette("Set2", n_colors=max(3, len(series_dict)))
    for (name, r), col in zip(series_dict.items(), palette):
        eq = (1 + r.dropna()).cumprod()
        ax.plot(eq.index, eq.values, color=col, lw=1.6, label=name)
    ax.set_yscale("log")
    ax.axhline(1, color="grey", lw=0.7)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, axis="y", which="both")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_weight_stack(
    weights: pd.DataFrame,
    out_path: Path | str,
    title: str,
    freq: str = "QE",
) -> Path:
    """Stacked-area plot of per-factor weights over time (shows how the
    rotation allocates across factors as regimes change)."""
    apply_style()
    w = weights.loc[weights.sum(axis=1) > 0].resample(freq).mean()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    palette = sns.color_palette("tab20", n_colors=len(w.columns))
    ax.stackplot(w.index, w.T.values, labels=w.columns, colors=palette, alpha=0.85)
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.set_ylabel("Factor weight")
    ax.legend(loc="upper left", fontsize=7, ncols=3, frameon=False)
    ax.grid(True, axis="y")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
