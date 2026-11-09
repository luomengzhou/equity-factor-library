"""Smoke tests for the point-in-time merge and the pure factor-construction helpers.

These run without vendor data: they build tiny synthetic panels in memory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_library.config import FUNDAMENTAL_REPORT_LAG_DAYS
from factor_library.data.cache import _make_cache_key
from factor_library.data.loader import _merge_metric_asof
from factor_library.factors.calculators.base import apply_rank, build_composite, winsorize_global
from factor_library.factors.calculators.momentum import calculate_momentum_matrix


def _daily_panel(n_days: int = 400, ids=("A-R", "B-R")) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    frames = []
    for i, pid in enumerate(ids):
        frames.append(pd.DataFrame({
            "fsym_primary_listing_id": pid,
            "date": dates,
            "price": 100.0 + i + np.arange(n_days) * 0.1,
        }))
    return pd.concat(frames, ignore_index=True)


def test_fundamental_asof_merge_applies_reporting_lag():
    panel = _daily_panel()
    report_date = pd.Timestamp("2020-03-31")
    fund = pd.DataFrame({
        "fsym_primary_listing_id": ["A-R"],
        "date": [report_date],
        "ff_eps": [5.0],
    })
    out = _merge_metric_asof(panel, fund, "date", ["ff_eps"])
    a = out[out["fsym_primary_listing_id"] == "A-R"].set_index("date")["ff_eps"]
    first_visible = a.first_valid_index()
    # The quarter-end number must not be visible on the quarter-end date itself...
    assert pd.isna(a.loc[report_date])
    # ...but only once the reporting lag has elapsed.
    assert first_visible >= report_date + pd.Timedelta(days=FUNDAMENTAL_REPORT_LAG_DAYS)
    assert first_visible <= report_date + pd.Timedelta(days=FUNDAMENTAL_REPORT_LAG_DAYS + 4)
    # B has no fundamentals -> all NaN, no cross-contamination between ids.
    assert out.loc[out["fsym_primary_listing_id"] == "B-R", "ff_eps"].isna().all()


def test_market_data_asof_merge_has_no_lag():
    panel = _daily_panel(n_days=50)
    vol = panel[["fsym_primary_listing_id", "date"]].copy()
    vol["volume"] = 1_000.0
    out = _merge_metric_asof(panel, vol, "date", ["volume"], lag_days=0)
    assert out["volume"].notna().all()


def test_momentum_uses_only_past_prices():
    panel = _daily_panel(n_days=300, ids=("A-R",))
    mom = calculate_momentum_matrix(panel.copy())
    # Perturb the LAST price only: no earlier row's momentum may change.
    bumped = panel.copy()
    bumped.loc[bumped.index[-1], "price"] *= 2
    mom2 = calculate_momentum_matrix(bumped)
    for col in ("mom_12m_1m", "mom_6m_1m", "mom_1m_reversal", "mom_52w_high"):
        pd.testing.assert_series_equal(mom[col].iloc[:-1], mom2[col].iloc[:-1], check_names=False)
    # 12-1 momentum is NaN until 273 trading days of history exist.
    assert mom["mom_12m_1m"].iloc[:273].isna().all()
    assert mom["mom_12m_1m"].iloc[273:].notna().all()


def test_rank_and_composite_are_cross_sectional_per_date():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3),
        "m1": [1.0, 2.0, 3.0, 30.0, 20.0, 10.0],
        "m2": [3.0, 2.0, 1.0, np.nan, 1.0, 2.0],
    })
    df["_r1"] = apply_rank(df, "m1")
    df["_r2"] = apply_rank(df, "m2")
    assert df["_r1"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0, 1.0, 2 / 3, 1 / 3])
    comp = build_composite(df, ["_r1", "_r2"], min_count=2)
    assert comp.iloc[0] == pytest.approx((1 / 3 + 1.0) / 2)
    assert np.isnan(comp.iloc[3])  # only one metric available -> below min_count


def test_winsorize_clips_tails_only():
    s = pd.Series(np.r_[np.linspace(0, 1, 99), 1000.0])
    w = winsorize_global(s, 0.01, 0.99)
    assert w.max() < 1000.0
    assert (w.iloc[1:-1] == s.iloc[1:-1]).all()


def test_cache_key_depends_on_winsorize_bounds():
    args = (["A-R"], "Momentum", None, None, "percentile", "equal_weight", True)
    k1 = _make_cache_key(*args, winsorize_bounds=(0.01, 0.99))
    k2 = _make_cache_key(*args, winsorize_bounds=(0.05, 0.95))
    assert k1 != k2
