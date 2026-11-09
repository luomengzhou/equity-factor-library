from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LongShortConfig:
    """
    A simple, report-friendly long-short definition.

    - On each date t, form long = top bucket, short = bottom bucket by factor value.
    - Use equal weights within each leg, dollar-neutral: +0.5 long, -0.5 short.
    - Apply weights at t to next-day return (t->t+1) to avoid look-ahead.
    """

    n_buckets: int = 5
    min_cross_section: int = 30
    leg_gross: float = 0.5
    vol_lookback: int = 60


def compute_past_daily_returns_from_prices(
    prices: pd.DataFrame,
    *,
    id_col: str = "fsym_primary_listing_id",
    date_col: str = "date",
    price_col: str = "price",
    out_col: str = "ret_1d",
) -> pd.DataFrame:
    """Past 1-day returns aligned at date t (return from t-1 to t)."""
    p = prices[[id_col, date_col, price_col]].copy()
    p[date_col] = pd.to_datetime(p[date_col])
    p = p.sort_values([id_col, date_col])
    g = p.groupby(id_col, sort=False)[price_col]
    p[out_col] = g.pct_change().values
    return p[[id_col, date_col, out_col]]


def compute_rolling_volatility(
    daily_ret: pd.DataFrame,
    *,
    window: int = 60,
    min_periods: int | None = None,
    id_col: str = "fsym_primary_listing_id",
    date_col: str = "date",
    ret_col: str = "ret_1d",
    out_col: str = "sigma_60d",
) -> pd.DataFrame:
    """
    Rolling volatility estimate aligned at date t using past returns only.

    This is a simple ex-ante risk proxy for risk-contribution diagnostics.
    """
    if min_periods is None:
        min_periods = max(20, int(window // 2))
    d = daily_ret[[id_col, date_col, ret_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values([id_col, date_col])
    g = d.groupby(id_col, sort=False)[ret_col]
    d[out_col] = g.rolling(int(window), min_periods=int(min_periods)).std().reset_index(level=0, drop=True).values
    return d[[id_col, date_col, out_col]]


def compute_next_day_returns_from_prices(
    prices: pd.DataFrame,
    *,
    id_col: str = "fsym_primary_listing_id",
    date_col: str = "date",
    price_col: str = "price",
    out_col: str = "ret_1d_fwd",
) -> pd.DataFrame:
    """Forward 1-day returns aligned at date t (return from t to t+1)."""
    p = prices[[id_col, date_col, price_col]].copy()
    p[date_col] = pd.to_datetime(p[date_col])
    p = p.sort_values([id_col, date_col])
    g = p.groupby(id_col, sort=False)[price_col]
    # pct_change gives r_t = P_t/P_{t-1}-1; shift(-1) aligns r_{t+1} onto date t
    p[out_col] = g.pct_change().shift(-1).values
    return p[[id_col, date_col, out_col]]


def _bucket_by_rank(
    factor_df: pd.DataFrame,
    *,
    factor_col: str,
    cfg: LongShortConfig,
    id_col: str,
    date_col: str,
) -> pd.DataFrame:
    """Return id/date rows with factor value and bucket 1..N."""
    f = factor_df[[id_col, date_col, factor_col]].copy()
    f[date_col] = pd.to_datetime(f[date_col])
    f = f.dropna(subset=[factor_col])
    if f.empty:
        return f.assign(bucket=pd.Series(dtype="Int64"))

    s = f[factor_col]
    ranks = s.groupby(f[date_col], sort=False).rank(method="average")
    n = s.groupby(f[date_col], sort=False).transform("count")
    buckets = np.floor((ranks - 1) * int(cfg.n_buckets) / n) + 1
    f["bucket"] = buckets.astype("Int64")
    f["n_cs"] = n.astype("Int64")
    return f


def compute_long_short_weights(
    bucketed: pd.DataFrame,
    *,
    cfg: LongShortConfig,
    exclude_ids: Iterable[str] | None = None,
    id_col: str = "fsym_primary_listing_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Create per-(id,date) weights for an equal-weight long-short.

    Exclusion is implemented as a hard drop before forming equal weights, i.e.
    weights are re-normalized within each leg on each date.
    """
    b = bucketed[[id_col, date_col, "bucket", "n_cs"]].copy()
    b[date_col] = pd.to_datetime(b[date_col])
    if exclude_ids is not None:
        ex = set(str(x) for x in exclude_ids)
        if ex:
            b = b[~b[id_col].astype(str).isin(ex)].copy()

    # cross-section filter
    b = b[b["n_cs"] >= int(cfg.min_cross_section)].copy()
    if b.empty:
        return b.assign(weight=pd.Series(dtype=float))

    top = int(cfg.n_buckets)
    bot = 1
    b["side"] = np.where(b["bucket"] == top, "long", np.where(b["bucket"] == bot, "short", "mid"))
    b = b[b["side"].isin(["long", "short"])].copy()
    if b.empty:
        return b.assign(weight=pd.Series(dtype=float))

    # equal weights within side/date; gross per leg = cfg.leg_gross
    counts = b.groupby([date_col, "side"], sort=False)[id_col].transform("count").astype(float)
    w = float(cfg.leg_gross) / counts
    b["weight"] = np.where(b["side"] == "long", w, -w)
    return b[[id_col, date_col, "side", "bucket", "weight"]]


def compute_long_short_returns_and_contributions(
    factor_df: pd.DataFrame,
    *,
    factor_col: str,
    forward_1d_ret: pd.DataFrame,
    cfg: LongShortConfig = LongShortConfig(),
    exclude_ids: Iterable[str] | None = None,
    id_col: str = "fsym_primary_listing_id",
    date_col: str = "date",
    ret_col: str = "ret_1d_fwd",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute strategy daily return and per-stock contribution time series.

    Returns
    -------
    strat : DataFrame
        columns: date, factor, long_short_ret, n_long, n_short
    contrib : DataFrame
        columns: date, id, factor, side, weight, ret_1d_fwd, contribution
    """
    # Build bucket membership from factor
    bucketed = _bucket_by_rank(factor_df, factor_col=factor_col, cfg=cfg, id_col=id_col, date_col=date_col)
    w = compute_long_short_weights(bucketed, cfg=cfg, exclude_ids=exclude_ids, id_col=id_col, date_col=date_col)
    if w.empty:
        strat = pd.DataFrame(columns=[date_col, "factor", "long_short_ret", "n_long", "n_short"])
        contrib = pd.DataFrame(columns=[date_col, id_col, "factor", "side", "weight", ret_col, "contribution"])
        return strat, contrib

    r = forward_1d_ret[[id_col, date_col, ret_col]].copy()
    r[date_col] = pd.to_datetime(r[date_col])
    m = w.merge(r, on=[id_col, date_col], how="left").dropna(subset=[ret_col])
    m["contribution"] = m["weight"] * m[ret_col]
    m["factor"] = factor_col

    # daily aggregation
    daily = m.groupby(date_col, sort=True).agg(
        long_short_ret=("contribution", "sum"),
        n_long=("side", lambda s: int((s == "long").sum())),
        n_short=("side", lambda s: int((s == "short").sum())),
    ).reset_index()
    daily["factor"] = factor_col

    return daily[[date_col, "factor", "long_short_ret", "n_long", "n_short"]], m[
        [date_col, id_col, "factor", "side", "weight", ret_col, "contribution"]
    ]


def summarize_concentration(contrib: pd.DataFrame, *, id_col: str = "fsym_primary_listing_id") -> pd.DataFrame:
    """
    Per-date concentration diagnostics based on contribution magnitudes.

    Outputs:
    - hhi_abs: sum(p_i^2) where p_i = |contrib_i| / sum_j |contrib_j|
    - eff_n: 1 / hhi_abs
    - top5_share_abs: sum of top-5 |contrib| over sum |contrib|
    """
    if contrib.empty:
        return pd.DataFrame(columns=["date", "hhi_abs", "eff_n", "top5_share_abs", "abs_pnl"])

    d = contrib.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["abs_c"] = d["contribution"].abs()

    rows = []
    for dt, g in d.groupby("date", sort=True):
        abs_sum = float(g["abs_c"].sum())
        if not np.isfinite(abs_sum) or abs_sum <= 0:
            continue
        p = (g["abs_c"] / abs_sum).to_numpy(dtype=float)
        hhi = float(np.sum(p * p))
        eff_n = float(1.0 / hhi) if hhi > 0 else float("nan")
        top5 = float(np.sort(p)[-5:].sum()) if len(p) >= 5 else float(np.sum(p))
        rows.append(
            {
                "date": dt,
                "hhi_abs": hhi,
                "eff_n": eff_n,
                "top5_share_abs": top5,
                "abs_pnl": abs_sum,
            }
        )
    return pd.DataFrame(rows)


def leave_one_out_summary(
    factor_df: pd.DataFrame,
    *,
    factor_col: str,
    forward_1d_ret: pd.DataFrame,
    cfg: LongShortConfig = LongShortConfig(),
    id_col: str = "fsym_primary_listing_id",
    date_col: str = "date",
    ret_col: str = "ret_1d_fwd",
) -> pd.DataFrame:
    """
    LOO diagnostic: rerun the strategy excluding each id (renormalized within legs).

    Returns a table with the delta in mean daily return and IR vs baseline.
    """
    base_strat, _ = compute_long_short_returns_and_contributions(
        factor_df,
        factor_col=factor_col,
        forward_1d_ret=forward_1d_ret,
        cfg=cfg,
        exclude_ids=None,
        id_col=id_col,
        date_col=date_col,
        ret_col=ret_col,
    )
    if base_strat.empty:
        return pd.DataFrame(columns=[id_col, "mean_base", "ir_base", "mean_loo", "ir_loo", "delta_mean", "delta_ir"])

    base = base_strat["long_short_ret"].dropna().to_numpy(dtype=float)
    mean_base = float(np.mean(base))
    std_base = float(np.std(base, ddof=1)) if len(base) > 1 else float("nan")
    ir_base = mean_base / std_base if std_base and np.isfinite(std_base) and std_base > 0 else float("nan")

    ids = (
        factor_df[id_col]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
    )
    rows: list[dict[str, object]] = []
    for sid in ids:
        strat_loo, _ = compute_long_short_returns_and_contributions(
            factor_df,
            factor_col=factor_col,
            forward_1d_ret=forward_1d_ret,
            cfg=cfg,
            exclude_ids=[sid],
            id_col=id_col,
            date_col=date_col,
            ret_col=ret_col,
        )
        r = strat_loo["long_short_ret"].dropna().to_numpy(dtype=float)
        if len(r) < 5:
            continue
        mean_loo = float(np.mean(r))
        std_loo = float(np.std(r, ddof=1)) if len(r) > 1 else float("nan")
        ir_loo = mean_loo / std_loo if std_loo and np.isfinite(std_loo) and std_loo > 0 else float("nan")
        rows.append(
            {
                id_col: sid,
                "mean_base": mean_base,
                "ir_base": ir_base,
                "mean_loo": mean_loo,
                "ir_loo": ir_loo,
                "delta_mean": mean_loo - mean_base,
                "delta_ir": (ir_loo - ir_base) if (np.isfinite(ir_loo) and np.isfinite(ir_base)) else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("delta_mean", ascending=True).reset_index(drop=True)
    return out


def compute_ex_ante_constraint_diagnostics(
    weights: pd.DataFrame,
    *,
    sigma: pd.DataFrame | None = None,
    weight_cap: float | None = None,
    rc_cap: float | None = None,
    id_col: str = "fsym_primary_listing_id",
    date_col: str = "date",
    weight_col: str = "weight",
    sigma_col: str = "sigma_60d",
) -> pd.DataFrame:
    """
    Per-date, ex-ante diagnostics that can translate LOO findings into trading constraints.

    - weight concentration: max |w|, effective N on weights, top-5 weight share
    - risk-contribution proxy: RC_i ∝ |w_i| * sigma_i, then normalize across names per day
      (this is a simple proxy, not a full covariance model)
    - cap triggers (optional): share of days where max|w| exceeds weight_cap; where max RC share exceeds rc_cap
    """
    if weights.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "n_names",
                "max_abs_weight",
                "eff_n_weights",
                "top5_weight_share",
                "rc_hhi",
                "rc_eff_n",
                "rc_top5_share",
                "max_rc_share",
                "breach_weight_cap",
                "breach_rc_cap",
            ]
        )

    w = weights[[id_col, date_col, weight_col]].copy()
    w[date_col] = pd.to_datetime(w[date_col])
    w["abs_w"] = w[weight_col].abs()
    if sigma is not None and not sigma.empty:
        s = sigma[[id_col, date_col, sigma_col]].copy()
        s[date_col] = pd.to_datetime(s[date_col])
        w = w.merge(s, on=[id_col, date_col], how="left")
    else:
        w[sigma_col] = np.nan

    rows: list[dict[str, object]] = []
    for dt, g in w.groupby(date_col, sort=True):
        g = g.dropna(subset=["abs_w"])
        if g.empty:
            continue
        abs_sum = float(g["abs_w"].sum())
        if not np.isfinite(abs_sum) or abs_sum <= 0:
            continue
        p_w = (g["abs_w"] / abs_sum).to_numpy(dtype=float)
        hhi_w = float(np.sum(p_w * p_w))
        eff_n_w = float(1.0 / hhi_w) if hhi_w > 0 else float("nan")
        top5_w = float(np.sort(p_w)[-5:].sum()) if len(p_w) >= 5 else float(np.sum(p_w))
        max_abs_w = float(g["abs_w"].max())

        # RC proxy
        rc = None
        max_rc_share = float("nan")
        rc_hhi = float("nan")
        rc_eff_n = float("nan")
        rc_top5 = float("nan")
        if sigma_col in g.columns and g[sigma_col].notna().any():
            sub = g.dropna(subset=[sigma_col]).copy()
            if not sub.empty:
                sub["rc_proxy"] = sub["abs_w"] * sub[sigma_col].abs()
                rc_sum = float(sub["rc_proxy"].sum())
                if np.isfinite(rc_sum) and rc_sum > 0:
                    rc = (sub["rc_proxy"] / rc_sum).to_numpy(dtype=float)
                    rc_hhi = float(np.sum(rc * rc))
                    rc_eff_n = float(1.0 / rc_hhi) if rc_hhi > 0 else float("nan")
                    rc_top5 = float(np.sort(rc)[-5:].sum()) if len(rc) >= 5 else float(np.sum(rc))
                    max_rc_share = float(np.max(rc)) if len(rc) else float("nan")

        breach_w = bool(weight_cap is not None and np.isfinite(max_abs_w) and max_abs_w > float(weight_cap))
        breach_rc = bool(rc_cap is not None and np.isfinite(max_rc_share) and max_rc_share > float(rc_cap))

        rows.append(
            {
                "date": dt,
                "n_names": int(len(g)),
                "max_abs_weight": max_abs_w,
                "eff_n_weights": eff_n_w,
                "top5_weight_share": top5_w,
                "rc_hhi": rc_hhi,
                "rc_eff_n": rc_eff_n,
                "rc_top5_share": rc_top5,
                "max_rc_share": max_rc_share,
                "breach_weight_cap": breach_w,
                "breach_rc_cap": breach_rc,
            }
        )

    return pd.DataFrame(rows)

