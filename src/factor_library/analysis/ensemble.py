"""
Multi-factor ensemble — combine several factor ranks into a single composite
signal. Three weighting schemes:

* equal          — simple cross-sectional average of ranks
* inverse_vol    — weight each factor inversely to its long/short vol
* ic_weighted    — weight each factor by its trailing 6-month IR (lagged to
                   avoid look-ahead)

The output is a combined DataFrame with the same schema as a single-factor
composite parquet (fsym_primary_listing_id, date, factor_value) so it can
go straight into the existing quantile / tear-sheet / cost pipeline.
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

from factor_library.analysis.ic import compute_ic, prepare_alphalens_input
from factor_library.reports.tearsheets import long_short_daily_returns
from factor_library.viz.style import apply_style


Method = Literal["equal", "inverse_vol", "ic_weighted"]


# ---------------------------------------------------------------------------
# Build the long (id, date) frame with one column per factor
# ---------------------------------------------------------------------------
def stack_factor_values(
    factor_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Return a wide (id, date)-indexed frame with columns = factor names,
    values = factor_value (percentile rank in [0, 1]).
    """
    frames = []
    for name, df in factor_dfs.items():
        t = df[["fsym_primary_listing_id", "date", "factor_value"]].copy()
        t["date"] = pd.to_datetime(t["date"])
        t = t.rename(columns={"factor_value": name})
        frames.append(t.set_index(["fsym_primary_listing_id", "date"]))
    wide = pd.concat(frames, axis=1, join="outer")
    return wide


# ---------------------------------------------------------------------------
# Weighting schemes
# ---------------------------------------------------------------------------
def _equal_weights(factors: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(factors)
    return pd.DataFrame(1.0 / n, index=dates, columns=factors)


def _inverse_vol_weights(
    factor_dfs: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    dates: pd.DatetimeIndex,
    lookback_days: int = 252,
) -> pd.DataFrame:
    """
    For each factor, compute a rolling (lookback) realised vol of its
    long/short Q5-Q1 daily return. Weight = 1 / vol, normalised to sum to 1.
    Weights use only data up to day t-1 (shift to avoid look-ahead).
    """
    ls_returns = {}
    for name, df in factor_dfs.items():
        ls_returns[name] = long_short_daily_returns(df, prices)
    ls_frame = pd.DataFrame(ls_returns).reindex(dates)
    rolling_vol = ls_frame.rolling(lookback_days, min_periods=60).std(ddof=1)
    inv_vol = (1.0 / rolling_vol.replace(0.0, np.nan)).shift(1)  # lag one day
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)  # normalise per row
    return weights


def _cap_weight_row(row: pd.Series, max_weight: float) -> pd.Series:
    """
    Cap a single row of non-negative weights and renormalise to sum to 1.
    """
    w = row.to_numpy(dtype=float, copy=True)
    w = np.where(np.isfinite(w), np.clip(w, 0.0, None), 0.0)
    total = float(w.sum())
    if total <= 0.0:
        return pd.Series(np.zeros_like(w), index=row.index)

    w /= total
    n = len(w)
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError(
            f"max_weight={max_weight:.4f} too small for {n} factors; "
            "must satisfy max_weight * n >= 1."
        )

    for _ in range(n + 2):
        over = w > max_weight + 1e-12
        if not np.any(over):
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight

        under = ~over
        under_sum = float(w[under].sum())
        if under_sum <= 0.0:
            room = np.clip(max_weight - w, 0.0, None)
            room_sum = float(room.sum())
            if room_sum <= 0.0:
                break
            w += room / room_sum * excess
        else:
            w[under] += (w[under] / under_sum) * excess

    total = float(w.sum())
    if total > 0.0:
        w /= total
    return pd.Series(w, index=row.index)


def _apply_weight_cap(weights: pd.DataFrame, max_weight: float) -> pd.DataFrame:
    if max_weight >= 1.0:
        return weights
    return weights.apply(_cap_weight_row, axis=1, max_weight=max_weight)


def _ic_weighted_weights(
    factor_dfs: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    dates: pd.DatetimeIndex,
    lookback_days: int = 126,
    horizon_days: int = 20,
    tau_scale: float = 0.50,
    max_weight: float = 0.30,
    equal_floor: float = 0.30,
) -> pd.DataFrame:
    """
    Weighting with three robustness guards:
      1) IR shrinkage (soft threshold on |IR|)
      2) single-factor cap
      3) equal-weight floor blend

    Base signal uses trailing 6-month IR (ic_mean / ic_std) on the h-day
    forward-return IC series.

    Look-ahead control: IC at date t uses the realised return from t to
    t+h, so the latest IR we can legitimately use to weight the signal
    at day T is IR_{T - h - 1}. We therefore shift by ``horizon_days + 1``
    days rather than the naive 1-day lag.
    """
    if tau_scale < 0.0:
        raise ValueError("tau_scale must be >= 0.")
    if not (0.0 < max_weight <= 1.0):
        raise ValueError("max_weight must be in (0, 1].")
    if not (0.0 <= equal_floor <= 1.0):
        raise ValueError("equal_floor must be in [0, 1].")

    ic_series = {}
    for name, df in factor_dfs.items():
        merged = prepare_alphalens_input(df, prices, horizons=(horizon_days,))
        ic = compute_ic(merged)[f"{horizon_days}D"]
        ic_series[name] = ic
    ic_frame = pd.DataFrame(ic_series).reindex(dates)
    roll_mean = ic_frame.rolling(lookback_days, min_periods=60).mean()
    roll_std = ic_frame.rolling(lookback_days, min_periods=60).std(ddof=1)

    # Shift by horizon+1 to make sure today's weight only uses IC estimates
    # whose forward-return window closed before today.
    ir = (roll_mean / roll_std).shift(horizon_days + 1)
    ir = ir.replace([np.inf, -np.inf], np.nan)

    # Soft-threshold |IR| by date: tau = tau_scale * median(|IR|).
    abs_ir = ir.abs()
    tau = tau_scale * abs_ir.median(axis=1, skipna=True)
    shrunk_mag = abs_ir.sub(tau, axis=0).clip(lower=0.0)
    ir_shrunk = np.sign(ir) * shrunk_mag

    # Long-only factor-combination signal (no negative allocation to factors).
    signal = ir_shrunk.clip(lower=0.0)
    signal_sum = signal.sum(axis=1).replace(0.0, np.nan)
    base_weights = signal.div(signal_sum, axis=0).fillna(0.0)

    capped = _apply_weight_cap(base_weights, max_weight=max_weight)
    equal = _equal_weights(list(signal.columns), signal.index)
    blended = (1.0 - equal_floor) * capped + equal_floor * equal
    blended = blended.div(blended.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

    # Keep warm-up period behaviour: no IR history -> no dynamic weights yet.
    has_ir_history = ir.notna().any(axis=1)
    weights = blended.where(has_ir_history, other=0.0, axis=0)
    return weights


# ---------------------------------------------------------------------------
# Build the combined factor
# ---------------------------------------------------------------------------
def build_ensemble_factor(
    factor_dfs: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    method: Method = "equal",
) -> pd.DataFrame:
    """
    Combine the given factor composites with the chosen scheme and return a
    single-factor DataFrame: (fsym_primary_listing_id, date, factor_value).
    """
    wide = stack_factor_values(factor_dfs)  # MultiIndex (id, date) x factors
    factors = list(factor_dfs.keys())
    dates = wide.index.get_level_values("date").unique().sort_values()

    if method == "equal":
        w = _equal_weights(factors, dates)
    elif method == "inverse_vol":
        w = _inverse_vol_weights(factor_dfs, prices, dates)
    elif method == "ic_weighted":
        w = _ic_weighted_weights(factor_dfs, prices, dates)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    # Weighted sum per (id, date), without imputing missing factor values.
    # For each row we re-normalize on the subset of available factors only.
    dates_idx = wide.index.get_level_values("date")
    w_aligned = w.reindex(dates_idx)
    w_aligned.index = wide.index

    wide_vals = wide.to_numpy(dtype=float)
    base_w = w_aligned.to_numpy(dtype=float)
    available = np.isfinite(wide_vals).astype(float)
    eff_w = base_w * available
    eff_w_sum = np.nansum(eff_w, axis=1)
    denom = np.where(eff_w_sum > 0.0, eff_w_sum, np.nan)
    norm_w = eff_w / denom[:, None]

    combined_vals = np.nansum(wide_vals * norm_w, axis=1)
    combined_vals[~np.isfinite(denom)] = np.nan
    combined = pd.Series(combined_vals, index=wide.index, name="weighted_sum")

    # re-rank cross-sectionally each day so the composite lives on the
    # same [0, 1] percentile scale as the underlying factors
    combined = combined.reset_index()
    combined["factor_value"] = (combined
                                .groupby("date")["weighted_sum"]
                                .rank(pct=True, method="average"))
    combined = combined.dropna(subset=["factor_value"])
    combined = combined[["fsym_primary_listing_id", "date", "factor_value"]]
    return combined


# ---------------------------------------------------------------------------
# Comparison plot + stats
# ---------------------------------------------------------------------------
def _compound_equity(ret: pd.Series) -> pd.Series:
    return (1.0 + ret).cumprod()


def _max_dd_log(ret: pd.Series) -> float:
    log_eq = np.log1p(ret).cumsum()
    dd = np.exp(log_eq - log_eq.cummax()) - 1.0
    return float(dd.min())


def compare_ensembles(
    series_dict: dict[str, pd.Series],
) -> pd.DataFrame:
    """One-row-per-strategy summary: ann_return, sharpe, maxdd, sortino."""
    rows = []
    for name, s in series_dict.items():
        mdd = _max_dd_log(s)
        ann = float(emp.annual_return(s))
        rows.append({
            "strategy":    name,
            "ann_return":  ann,
            "ann_vol":     float(s.std(ddof=1) * np.sqrt(252)),
            "sharpe":      float(emp.sharpe_ratio(s, annualization=252)),
            "sortino":     float(emp.sortino_ratio(s, annualization=252)),
            "max_drawdown": mdd,
            "calmar":      float(ann / abs(mdd)) if mdd < 0 else np.nan,
            "n_days":      int(len(s)),
        })
    return pd.DataFrame(rows).set_index("strategy")


def factor_overlap_matrix(
    factor_dfs: dict[str, pd.DataFrame],
    method: str = "spearman",
) -> tuple[pd.DataFrame, int]:
    """
    Cross-factor overlap proxy:
    correlation matrix of factor_value columns on the strict inner-join
    sample (id, date) covered by every included factor.
    """
    wide = stack_factor_values(factor_dfs).dropna(how="any")
    corr = wide.corr(method=method)
    return corr, int(len(wide))


def plot_factor_overlap_heatmap(
    overlap: pd.DataFrame,
    n_points: int,
    out_path: Path | str,
    method: str = "spearman",
) -> Path:
    """Heatmap for factor overlap matrix."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    sns.heatmap(
        overlap,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": f"{method.capitalize()} correlation"},
        ax=ax,
    )
    ax.set_title(f"Factor overlap ({method}, inner-join on {n_points:,} points)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_ensemble_comparison(
    series_dict: dict[str, pd.Series],
    out_path: Path | str,
    title: str = "Ensemble vs single-factor — growth of $1",
) -> Path:
    """Log-scale growth-of-$1 chart, one line per strategy."""
    apply_style()
    fig, ax = plt.subplots(figsize=(11, 5.4))
    palette = sns.color_palette("Set2", n_colors=max(3, len(series_dict)))
    for (name, s), col in zip(series_dict.items(), palette):
        eq = _compound_equity(s)
        ax.plot(eq.index, eq.values, color=col, lw=1.6, label=name)
    ax.set_yscale("log")
    ax.axhline(1, color="grey", lw=0.7)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1 (log)")
    ax.set_xlabel("")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, axis="y", which="both")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
