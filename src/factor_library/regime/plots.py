"""
Heatmap renderers for regime-conditional IC/IR tables.

Takes the long-form output of ``compute_ic_summary_vs_forward_return_grouped``
(one row per (factor, group)) and produces a (factor × group) heatmap of IR
or ic_mean with the library-standard style.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from factor_library.viz.style import DIVERGING_CMAP, apply_style


def plot_regime_ir_heatmap(
    grouped_ic: pd.DataFrame,
    out_path: Path | str,
    *,
    metric: str = "ir",
    title: str | None = None,
    regime_order: list[str] | None = None,
) -> Path:
    """
    Render a factor × regime heatmap of ``metric`` (default: IR).

    ``grouped_ic`` columns: factor, group (the regime label), ic_mean,
    ic_std, ir, n_days, ret_col.
    """
    apply_style()
    df = grouped_ic.copy()
    # Prettier factor labels — restore display names
    df["factor"] = df["factor"].str.replace("___", " / ").str.replace("_", " ")
    piv = df.pivot(index="factor", columns="group", values=metric).sort_index()
    if regime_order is not None:
        ordered = [c for c in regime_order if c in piv.columns]
        piv = piv[ordered]

    fig, ax = plt.subplots(
        figsize=(max(5, 0.9 * piv.shape[1] + 2.5), 0.42 * piv.shape[0] + 1.5)
    )
    sns.heatmap(
        piv, cmap=DIVERGING_CMAP, center=0, annot=True, fmt=".2f",
        linewidths=0.4, cbar_kws={"label": metric.upper()}, ax=ax,
    )
    ax.set_title(title or f"Factor {metric.upper()} by regime")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
