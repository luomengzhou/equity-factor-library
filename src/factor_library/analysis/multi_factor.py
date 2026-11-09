"""
Multi-factor regression attribution — how much of each factor's long/short
return is explained by the others?

For each target factor we regress its daily long-short return onto the
other factors' long-short returns (plus a constant). The intercept
(*alpha*) is the residual edge that survives after controlling for
common risk premia. Mirrors the classic Fama–French style attribution
but uses the project's own 12 factors as the RHS instead of Mkt-RF /
SMB / HML / MOM / RMW / CMA.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from factor_library.viz.style import DIVERGING_CMAP, apply_style


def ff_attribution_table(
    factor_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    For every factor in ``factor_returns``, regress it on every **other**
    factor (no look-through / include-self). Returns a long-form table with
    one row per (target, predictor) pair plus one row per target reporting
    alpha, R², N, annualised-alpha.

    ``factor_returns`` maps factor_name → daily long-short return series.
    """
    rows = []
    returns = pd.concat(factor_returns, axis=1, join="inner").dropna()
    factors = list(factor_returns.keys())

    for target in factors:
        y = returns[target]
        X_cols = [c for c in factors if c != target]
        X = sm.add_constant(returns[X_cols])
        res = sm.OLS(y, X, missing="drop").fit()
        alpha = res.params["const"]
        alpha_t = res.tvalues["const"]
        r2 = res.rsquared
        n = int(res.nobs)
        ann_alpha = alpha * 252
        rows.append({
            "target":    target,
            "predictor": "ALPHA",
            "beta":      alpha,
            "t_stat":    alpha_t,
            "p_value":   res.pvalues["const"],
            "r2":        r2,
            "n_obs":     n,
            "ann_alpha": ann_alpha,
        })
        for pred in X_cols:
            rows.append({
                "target":    target,
                "predictor": pred,
                "beta":      res.params[pred],
                "t_stat":    res.tvalues[pred],
                "p_value":   res.pvalues[pred],
                "r2":        r2,
                "n_obs":     n,
                "ann_alpha": ann_alpha,
            })
    return pd.DataFrame(rows)


def plot_attribution_heatmap(
    table: pd.DataFrame,
    out_path: Path | str,
    metric: str = "beta",
    mask_insignificant: float | None = 0.05,
) -> Path:
    """
    Heatmap of target × predictor betas (or t-stats). Optionally mask cells
    whose p-value is above ``mask_insignificant``.

    ``metric``: "beta" or "t_stat".
    """
    apply_style()
    piv_metric = table.pivot(index="target", columns="predictor", values=metric)
    # Reorder: ALPHA first, then the factor predictors alphabetically.
    cols = ["ALPHA"] + sorted(c for c in piv_metric.columns if c != "ALPHA")
    piv_metric = piv_metric[cols]
    piv_metric = piv_metric.loc[cols[1:]]  # rows in same order, minus ALPHA (not a target)
    annot = piv_metric.round(2)

    if mask_insignificant is not None:
        piv_p = table.pivot(index="target", columns="predictor", values="p_value")
        piv_p = piv_p[cols].loc[cols[1:]]
        mask = piv_p > mask_insignificant
    else:
        mask = None

    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(cols)), 0.55 * len(cols) + 0.8))
    sns.heatmap(
        piv_metric, cmap=DIVERGING_CMAP, center=0,
        annot=annot, fmt="", mask=mask,
        linewidths=0.4, cbar_kws={"label": metric},
        ax=ax,
    )
    ax.set_title(f"Factor attribution — {metric} (OLS, masked where p > {mask_insignificant})"
                 if mask_insignificant else f"Factor attribution — {metric}")
    ax.set_xlabel("Predictor")
    ax.set_ylabel("Target factor")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def alpha_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Compact per-target alpha table: sorted by ann_alpha descending."""
    s = table[table["predictor"] == "ALPHA"][
        ["target", "beta", "t_stat", "p_value", "ann_alpha", "r2", "n_obs"]
    ].rename(columns={"beta": "alpha_daily"})
    return s.sort_values("ann_alpha", ascending=False).reset_index(drop=True)
