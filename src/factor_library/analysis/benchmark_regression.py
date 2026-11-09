"""
Benchmark regression — how much of the 100-stock equal-weight benchmark's
daily return is explained by the library's 12 factors?

Mentor's explicit Stage-2 priority: "do a regression where you basically
have like all the factors and hopefully all the factors together are gonna
be able to explain like a very large portion of the benchmark market returns."

Two views, because a "factor return" has two common meanings:

  1. **Long/short (L/S)** — top quintile minus bottom quintile, dollar
     neutral by construction. This is the academic standard (Fama-French
     SMB / HML etc.) and what the rest of the library uses. Because L/S
     books have market beta ≈ 0, their R² against a market-beta-driven
     benchmark is mechanically low — typical of this kind of regression.
  2. **Long-only top quintile (LO)** — the top-quintile portfolio return
     with full market exposure. Useful for answering "how much of the
     benchmark does a portfolio tilted toward this factor explain."

Both are reported so the reader can choose the interpretation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from factor_library.viz.style import DIVERGING_CMAP, apply_style, colour_for


# ---------------------------------------------------------------------------
# Benchmark construction
# ---------------------------------------------------------------------------
def build_equal_weight_benchmark(step1_panel: pd.DataFrame) -> pd.Series:
    """
    Daily equal-weight return of every active name in the panel.
    No survivorship filter beyond what the panel already supplies.
    """
    wide = (step1_panel
            .pivot_table(index="date", columns="fsym_primary_listing_id",
                         values="price", aggfunc="last")
            .sort_index())
    daily_returns = wide.pct_change()
    benchmark = daily_returns.mean(axis=1, skipna=True)
    benchmark.name = "benchmark_return"
    return benchmark.dropna()


# ---------------------------------------------------------------------------
# Per-factor OLS — benchmark ~ α + β · factor
# ---------------------------------------------------------------------------
def regress_benchmark_on_factor(
    benchmark: pd.Series,
    factor_return: pd.Series,
    factor_name: str,
) -> dict:
    df = pd.concat({"bench": benchmark, "factor": factor_return}, axis=1).dropna()
    if len(df) < 50:
        return {"factor": factor_name, "n_obs": int(len(df)),
                "alpha_daily": np.nan, "alpha_ann_bps": np.nan,
                "beta": np.nan, "beta_t": np.nan, "beta_p": np.nan,
                "r_squared": np.nan}
    X = sm.add_constant(df["factor"])
    res = sm.OLS(df["bench"], X).fit()
    return {
        "factor":         factor_name,
        "n_obs":          int(res.nobs),
        "alpha_daily":    float(res.params["const"]),
        "alpha_ann_bps":  float(res.params["const"] * 252 * 1e4),
        "beta":           float(res.params["factor"]),
        "beta_t":         float(res.tvalues["factor"]),
        "beta_p":         float(res.pvalues["factor"]),
        "r_squared":      float(res.rsquared),
    }


# ---------------------------------------------------------------------------
# Joint OLS — benchmark ~ α + Σ β_i · factor_i
# ---------------------------------------------------------------------------
def regress_benchmark_on_all(
    benchmark: pd.Series,
    factor_returns: dict[str, pd.Series],
) -> tuple[dict, pd.DataFrame]:
    df = pd.concat(factor_returns, axis=1, join="inner")
    df["__bench__"] = benchmark
    df = df.dropna()
    if len(df) < 100:
        raise RuntimeError(
            f"joint regression has too few common observations ({len(df)})"
        )
    X = sm.add_constant(df.drop(columns="__bench__"))
    res = sm.OLS(df["__bench__"], X).fit()
    summary = {
        "alpha_daily":   float(res.params["const"]),
        "alpha_ann_bps": float(res.params["const"] * 252 * 1e4),
        "alpha_t":       float(res.tvalues["const"]),
        "r_squared":     float(res.rsquared),
        "adj_r_squared": float(res.rsquared_adj),
        "n_obs":         int(res.nobs),
        "n_factors":     int(len(factor_returns)),
    }
    betas = pd.DataFrame({
        "factor":  list(res.params.index[1:]),
        "beta":    [float(v) for v in res.params.values[1:]],
        "t_stat":  [float(v) for v in res.tvalues.values[1:]],
        "p_value": [float(v) for v in res.pvalues.values[1:]],
    }).sort_values("beta", key=np.abs, ascending=False).reset_index(drop=True)
    return summary, betas


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_per_factor_r2(per_factor_rows: pd.DataFrame, out_path: Path | str) -> Path:
    apply_style()
    df = per_factor_rows.sort_values("r_squared", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(df) + 1.2))
    colours = [colour_for(f) for f in df["factor"]]
    ax.barh(df["factor"], df["r_squared"] * 100, color=colours, alpha=0.85)
    for i, (r, beta, t) in enumerate(zip(df["r_squared"], df["beta"], df["beta_t"])):
        ax.text(r * 100 + 0.5, i, f"β={beta:+.2f}  t={t:+.1f}",
                va="center", fontsize=9, color="#333")
    ax.set_xlabel(r"$R^2$ (% of benchmark return explained by this factor alone)")
    ax.set_title("Per-factor regression: benchmark ~ α + β · factor")
    ax.grid(True, axis="x")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def long_only_top_quintile_returns(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    quantiles: int = 5,
) -> pd.Series:
    """
    Daily equal-weight return of the top-quintile (highest-factor-value)
    portfolio. NOT dollar neutral — has full market exposure. Mirrors
    ``long_short_daily_returns`` from reports.tearsheets but keeps only
    the long leg.
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="alphalens")
    import alphalens as al

    factor = (factor_df
              .sort_values(["date", "fsym_primary_listing_id"])
              .set_index(["date", "fsym_primary_listing_id"])["factor_value"])
    merged = al.utils.get_clean_factor_and_forward_returns(
        factor=factor, prices=prices,
        periods=(1,), quantiles=quantiles, max_loss=0.50,
    )
    dates = merged.index.get_level_values(0)
    by_q = (merged
            .groupby([dates, "factor_quantile"], observed=True)["1D"]
            .mean().unstack())
    top = by_q[quantiles].dropna()
    top.name = "top_quintile_return"
    return top


def partial_r2_per_factor(
    benchmark: pd.Series,
    factor_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Leave-one-out feature importance for the joint regression.

    For each factor f, refit the joint model without f and measure the
    drop in R² (``delta_r2 = full_r2 − loo_r2``). This is the unique
    variance in the benchmark that factor f explains after controlling
    for all other factors — a more honest "importance" metric than the
    raw coefficient, which is polluted by collinearity.

    Also reports ``share_of_r2`` (each factor's delta as a fraction of
    the sum of deltas) so the bars can be read as "% of joint R²
    uniquely attributable to this factor".

    Same sample as ``regress_benchmark_on_all`` (inner join across all
    factors + benchmark), so the full-model R² matches exactly.
    """
    df = pd.concat(factor_returns, axis=1, join="inner")
    df["__bench__"] = benchmark
    df = df.dropna()
    if len(df) < 100:
        raise RuntimeError(
            f"partial-R² regression has too few common observations ({len(df)})"
        )
    y = df["__bench__"]
    X_full = sm.add_constant(df.drop(columns="__bench__"))
    full_r2 = sm.OLS(y, X_full).fit().rsquared

    rows = []
    for name in factor_returns:
        X_loo = sm.add_constant(df.drop(columns=["__bench__", name]))
        loo_r2 = sm.OLS(y, X_loo).fit().rsquared
        rows.append({
            "factor":   name,
            "full_r2":  float(full_r2),
            "loo_r2":   float(loo_r2),
            "delta_r2": float(full_r2 - loo_r2),
        })
    out = pd.DataFrame(rows)
    total = out["delta_r2"].clip(lower=0).sum()
    out["share_of_r2"] = out["delta_r2"].clip(lower=0) / total if total > 0 else np.nan
    return out.sort_values("delta_r2", ascending=False).reset_index(drop=True)


def plot_partial_r2(partial: pd.DataFrame, joint_summary: dict, out_path: Path | str) -> Path:
    """
    Horizontal bar of leave-one-out ΔR² per factor. Answers the question
    "if I drop this factor, how much of the joint R² do I lose?" —
    collinearity-aware feature importance.
    """
    apply_style()
    df = partial.sort_values("delta_r2", ascending=True).copy()
    colours = [colour_for(f.replace("_", " ").replace("  ", " / ")) for f in df["factor"]]

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(df) + 1.6))
    ax.barh(df["factor"], df["delta_r2"] * 100, color=colours, alpha=0.85)
    for i, (d, s) in enumerate(zip(df["delta_r2"], df["share_of_r2"])):
        label = f"ΔR²={d*100:+.2f} pp  ({s*100:.1f}%)" if pd.notna(s) else f"ΔR²={d*100:+.2f} pp"
        ax.text(d * 100 + 0.02, i, label, va="center", fontsize=9, color="#333")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(r"Leave-one-out $\Delta R^2$ (percentage points — drop in joint $R^2$ if factor is removed)")
    ax.set_title(
        f"Feature importance: partial $R^2$ per factor   "
        f"(full joint $R^2$ = {joint_summary['r_squared']*100:.1f}%, "
        f"n = {joint_summary['n_obs']:,})"
    )
    ax.grid(True, axis="x")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_joint_betas(betas: pd.DataFrame, joint_summary: dict, out_path: Path | str) -> Path:
    apply_style()
    df = betas.sort_values("beta").copy()
    colours = [colour_for(f.replace("_", " ").replace("  ", " / ")) for f in df["factor"]]
    # Mask insignificant
    alpha_hatch = [("" if p < 0.05 else "//") for p in df["p_value"]]

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(df) + 1.6))
    bars = ax.barh(df["factor"], df["beta"], color=colours, alpha=0.85)
    for b, h in zip(bars, alpha_hatch):
        b.set_hatch(h)
    for i, (b, t, p) in enumerate(zip(df["beta"], df["t_stat"], df["p_value"])):
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(b + (0.003 if b >= 0 else -0.003), i,
                f"t={t:+.1f}{sig}",
                va="center", ha="left" if b >= 0 else "right",
                fontsize=9, color="#333")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(r"Joint regression $\beta$ (hatched = $p \geq 0.05$)")
    ax.set_title(
        f"Joint regression: benchmark ~ α + Σ βᵢ · factorᵢ   "
        f"($R^2$ = {joint_summary['r_squared']*100:.1f}%, "
        f"adj = {joint_summary['adj_r_squared']*100:.1f}%, "
        f"n = {joint_summary['n_obs']:,})"
    )
    ax.grid(True, axis="x")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
