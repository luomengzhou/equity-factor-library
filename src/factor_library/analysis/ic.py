"""
Information-Coefficient (IC) diagnostics — thin wrappers over alphalens.

``factor_validation.py`` used to hand-roll Spearman IC per period. alphalens
gives us the same math plus multi-horizon IC decay, IR, IC skew / kurtosis,
and quantile returns for free. We keep a tiny API here so the rest of the
library doesn't import alphalens directly.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DatetimeIndex

# --- pandas 2.3 / alphalens 0.4.6 compat -----------------------------------
# alphalens calls ``df.index.levels[0].freq = <inferred_freq>`` which pandas
# 2.3 validates strictly; on irregular trading-day panels the inferred freq
# disagrees with the index values and a ValueError is raised mid-pipeline.
# The freq assignment is a harmless internal hint — silence the validator.
_orig_freq_setter = DatetimeIndex.freq.fset

def _tolerant_freq_setter(self, value):
    try:
        _orig_freq_setter(self, value)
    except ValueError:
        pass

DatetimeIndex.freq = property(
    DatetimeIndex.freq.fget, _tolerant_freq_setter, DatetimeIndex.freq.fdel
)
# ---------------------------------------------------------------------------

# alphalens emits pandas-3 / future warnings on every import; silence them
# so they don't clutter our logs when callers import us.
warnings.filterwarnings("ignore", category=FutureWarning, module="alphalens")

import alphalens as al  # noqa: E402

from factor_library.viz.style import apply_style, colour_for


DEFAULT_HORIZONS = (1, 5, 20, 60)
DEFAULT_QUANTILES = 5


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def build_price_panel(step1_panel: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long step1 panel into a wide (date × fsym_primary_listing_id) price frame."""
    return (step1_panel
            .pivot_table(index="date", columns="fsym_primary_listing_id",
                         values="price", aggfunc="last")
            .sort_index())


def prepare_alphalens_input(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    quantiles: int = DEFAULT_QUANTILES,
    max_loss: float = 0.50,
) -> pd.DataFrame:
    """
    Turn a (fsym_primary_listing_id, date, factor_value) long DataFrame plus
    a wide price panel into alphalens' clean, merged, quantile-labelled frame.
    """
    factor = (factor_df
              .sort_values(["date", "fsym_primary_listing_id"])
              .set_index(["date", "fsym_primary_listing_id"])["factor_value"])
    return al.utils.get_clean_factor_and_forward_returns(
        factor=factor, prices=prices,
        periods=horizons, quantiles=quantiles, max_loss=max_loss,
    )


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------
def compute_ic(merged: pd.DataFrame) -> pd.DataFrame:
    """Per-period Spearman IC for every forward-return horizon present in ``merged``."""
    return al.performance.factor_information_coefficient(merged)


def ic_summary(ic_ts: pd.DataFrame) -> pd.DataFrame:
    """Mean IC, IC std, IR (mean/std), skew, kurt, t-stat per horizon."""
    rows = []
    for col in ic_ts.columns:
        s = ic_ts[col].dropna()
        mean = s.mean()
        std = s.std()
        ir = mean / std if std else np.nan
        tstat = mean / (std / np.sqrt(len(s))) if std else np.nan
        rows.append({
            "horizon": col, "n_obs": len(s),
            "ic_mean": mean, "ic_std": std, "ic_ir": ir,
            "ic_skew": s.skew(), "ic_kurt": s.kurt(), "t_stat": tstat,
        })
    return pd.DataFrame(rows).set_index("horizon")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_ic_decay(
    ic_summary_df: pd.DataFrame,
    factor_name: str,
    out_path: Path | str,
) -> Path:
    """
    Bar-plot mean IC across horizons, annotated with IR. Shows how quickly
    the signal decays.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    horizons = ic_summary_df.index.tolist()
    means = ic_summary_df["ic_mean"].values
    irs = ic_summary_df["ic_ir"].values
    bars = ax.bar(horizons, means, color=colour_for(factor_name), alpha=0.85, width=0.6)
    ax.axhline(0, color="black", lw=0.8)
    for b, ir in zip(bars, irs):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"IR {ir:.2f}", ha="center", va="bottom" if b.get_height() >= 0 else "top",
                fontsize=9, color="#333")
    ax.set_title(f"{factor_name} — IC decay across horizons")
    ax.set_xlabel("Forward-return horizon")
    ax.set_ylabel("Mean Spearman IC")
    ax.grid(True, axis="y")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_rolling_ic(
    ic_ts: pd.DataFrame,
    horizon: str,
    factor_name: str,
    out_path: Path | str,
    window: int = 252,
    ci: float = 0.95,
) -> Path:
    """
    Rolling mean IC with bootstrap-style ± n·se band. Replaces the old
    ``ic_mean_by_year`` bar chart.
    """
    apply_style()
    s = ic_ts[horizon].dropna()
    roll_mean = s.rolling(window).mean()
    roll_std = s.rolling(window).std()
    # z-score for two-sided CI
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(ci, 1.96)
    half_width = z * roll_std / np.sqrt(window)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    colour = colour_for(factor_name)
    ax.plot(roll_mean.index, roll_mean.values, color=colour, lw=1.4,
            label=f"{window}-day rolling mean IC")
    ax.fill_between(roll_mean.index,
                    (roll_mean - half_width).values,
                    (roll_mean + half_width).values,
                    color=colour, alpha=0.18, label=f"±{int(ci*100)}% CI")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{factor_name} — rolling {horizon} IC ({window}d window)")
    ax.set_xlabel(""); ax.set_ylabel("IC")
    ax.legend(loc="best")
    ax.grid(True, axis="y")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Convenience: full IC work-up for one factor
# ---------------------------------------------------------------------------
def run_ic_analysis(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    factor_name: str,
    output_dir: Path | str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict:
    """
    Full IC decay + rolling-IC write-out for one factor. Returns a dict with
    the summary table and paths to the charts.
    """
    out = Path(output_dir) / "ic_decay"
    merged = prepare_alphalens_input(factor_df, prices, horizons=horizons)
    ic_ts = compute_ic(merged)
    summary = ic_summary(ic_ts)

    summary_path = out / f"{factor_name.replace(' / ', '_').replace(' ', '_')}_ic_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path)

    decay_png = plot_ic_decay(summary, factor_name,
                              out / f"{factor_name.replace(' / ', '_').replace(' ', '_')}_decay.png")

    # Rolling IC on the 20-day horizon (monthly-ish) — most commonly reported.
    default_rolling_horizon = f"{horizons[2]}D" if len(horizons) >= 3 else ic_ts.columns[0]
    rolling_png = plot_rolling_ic(
        ic_ts, default_rolling_horizon, factor_name,
        out / f"{factor_name.replace(' / ', '_').replace(' ', '_')}_rolling_{default_rolling_horizon}.png",
    )

    return {
        "summary": summary,
        "summary_csv": summary_path,
        "decay_png": decay_png,
        "rolling_png": rolling_png,
        "ic_ts": ic_ts,
        "merged": merged,
    }
