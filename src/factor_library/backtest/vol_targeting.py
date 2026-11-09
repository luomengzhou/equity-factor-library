"""
Volatility targeting — the standard buy-side overlay for reducing
drawdown on a positive-expected-return factor without sacrificing
the long-run mean.

Each day scale the book by target_vol / realised_vol so that the
rolling ex-post vol stays close to a constant target. In calm
periods leverage is higher, in panics (2008, 2020 COVID) leverage
auto-deleverages before the drawdown compounds.

Empirically on Vol / Low-Risk this cuts max drawdown roughly in
half while keeping the geometric mean intact (often slightly
higher, because lower drawdown compounds better).
"""
from __future__ import annotations

from pathlib import Path

import empyrical as emp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factor_library.viz.style import apply_style, colour_for


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------
def vol_targeted_returns(
    daily_returns: pd.Series,
    target_vol_annual: float = 0.15,
    vol_lookback: int = 60,
    max_leverage: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Scale each day's return by target_vol / lagged_realised_vol.

    Returns
    -------
    scaled_returns : pd.Series
        Vol-targeted daily return series.
    exposure : pd.Series
        The leverage applied each day. 1.0 = 100% of the original book.
    """
    realised_vol = daily_returns.rolling(vol_lookback, min_periods=20).std() * np.sqrt(252)
    raw_exposure = target_vol_annual / realised_vol
    # shift by one day so today's position uses yesterday's vol estimate
    # (no look-ahead)
    exposure = raw_exposure.shift(1).clip(upper=max_leverage).fillna(0.0)
    scaled = daily_returns * exposure
    scaled.name = "vol_targeted_return"
    exposure.name = "exposure"
    return scaled, exposure


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
def _max_drawdown_log_space(ret: pd.Series) -> float:
    """
    Compute max drawdown using log cumulative returns so a 45-year
    compound equity curve doesn't overflow float64 (empyrical's default
    pathway returns NaN / 0 once (1+r).cumprod() exceeds 1e308).
    """
    log_eq = np.log1p(ret).cumsum()
    running_max = log_eq.cummax()
    drawdown = np.exp(log_eq - running_max) - 1.0  # <= 0
    return float(drawdown.min())


def _compound_stats(ret: pd.Series, label: str) -> dict:
    mdd = _max_drawdown_log_space(ret)
    ann_ret = float(emp.annual_return(ret))
    return {
        "strategy":       label,
        "ann_return":     ann_ret,
        "ann_vol":        float(ret.std(ddof=1) * np.sqrt(252)),
        "sharpe":         float(emp.sharpe_ratio(ret, annualization=252)),
        "sortino":        float(emp.sortino_ratio(ret, annualization=252)),
        "max_drawdown":   mdd,
        "calmar":         float(ann_ret / abs(mdd)) if mdd < 0 else np.nan,
        "skew":           float(ret.skew()),
        "kurtosis":       float(ret.kurtosis()),
    }


def compare_vol_targets(
    daily_returns: pd.Series,
    target_vols: list[float] = (0.10, 0.15, 0.20),
    vol_lookback: int = 60,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    """
    One-row-per-strategy comparison table. First row is the raw factor,
    subsequent rows are vol-targeted versions.
    """
    rows = [_compound_stats(daily_returns, "raw")]
    for tv in target_vols:
        scaled, _ = vol_targeted_returns(
            daily_returns, target_vol_annual=tv,
            vol_lookback=vol_lookback, max_leverage=max_leverage,
        )
        # drop initial NaNs from the lookback window
        scaled = scaled.loc[scaled.ne(0).idxmax():]
        rows.append(_compound_stats(scaled, f"target {int(tv*100)}%"))
    return pd.DataFrame(rows).set_index("strategy")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_vol_targeting_comparison(
    daily_returns: pd.Series,
    factor_name: str,
    out_path: Path | str,
    target_vols: list[float] = (0.10, 0.15, 0.20),
    vol_lookback: int = 60,
    max_leverage: float = 3.0,
) -> Path:
    """
    Two-panel chart:
      top    — growth-of-$1 (log scale) for raw + each vol-target
      bottom — daily exposure (leverage) for the middle target choice
    """
    apply_style()
    base_colour = colour_for(factor_name)

    fig, (ax_eq, ax_lev) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(11, 6.4),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # raw line first
    raw_equity = (1.0 + daily_returns).cumprod()
    ax_eq.plot(raw_equity.index, raw_equity.values,
               color="#555", lw=1.3, label="raw (100% book)")

    # vol-targeted lines
    shades = ["#89B6D1", "#4A8CBA", base_colour]  # light → vivid
    middle_idx = len(target_vols) // 2
    for i, tv in enumerate(target_vols):
        scaled, exposure = vol_targeted_returns(
            daily_returns, target_vol_annual=tv,
            vol_lookback=vol_lookback, max_leverage=max_leverage,
        )
        eq = (1.0 + scaled).cumprod()
        col = shades[i % len(shades)]
        ax_eq.plot(eq.index, eq.values, color=col, lw=1.5,
                   label=f"target vol {int(tv*100)}%")
        if i == middle_idx:
            middle_exposure = exposure

    ax_eq.set_yscale("log")
    ax_eq.set_ylabel("Growth of $1 (log)")
    ax_eq.set_title(f"{factor_name} — vol targeting vs raw (gross)")
    ax_eq.axhline(1, color="grey", lw=0.7)
    ax_eq.legend(loc="upper left", ncols=2, fontsize=9)
    ax_eq.grid(True, axis="y", which="both")

    # exposure panel
    ax_lev.plot(middle_exposure.index, middle_exposure.values,
                color=base_colour, lw=1.0)
    ax_lev.axhline(1.0, color="#888", lw=0.7, linestyle=":")
    ax_lev.set_ylabel(f"Exposure @ {int(target_vols[middle_idx]*100)}%")
    ax_lev.set_ylim(0, max_leverage * 1.05)
    ax_lev.grid(True, axis="y")
    ax_lev.set_xlabel("")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
