"""
Unified Data Loader — final self-contained panel + DB metric enrichment.

Core responsibilities:
1) Build base daily panel (price + coverage + assets + shares)
2) Optionally enrich panel with required DB metrics per factor metric_keys
3) All file inputs come from ``final/data/`` only (no other Week* folders).

Optional DB access uses ``columbia_factor_library`` — internal data-access helpers,
not distributed; see README Data section. Every import is guarded and the loader
falls back to the local panel / yfinance when they are missing.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from factor_library.config import (
    DATA_DIR,
    ENRICH_COVERAGE_THRESHOLD,
    FUNDAMENTAL_REPORT_LAG_DAYS,
    PART1_FILE,
    PART2_FILE,
    PRICE_TABLE,
    STEP1_EXTENDED_PARQUET,
    STEP1_MERGED_PARQUET,
)


# metric_key -> db table keys (table_fsym_mapping.METRIC_TABLES)
METRIC_KEY_TO_TABLE_KEYS: dict[str, list[str]] = {
    "market_cap": ["mktcap"],
    # Value
    "earnings_yield": ["eps_ltm"],
    "book_to_price": ["bps_q"],
    "sales_yield": ["sales_ps_ltm"],
    "cf_yield": ["fcfe_ltm"],
    "ev_ebitda": ["ev_q", "ebitda_ltm"],
    "dividend_yield": ["div_paid_ltm"],
    "shareholder_yield": ["div_paid_ltm", "buyback"],
    "peg": ["eps_ltm"],
    # Investment
    "asset_growth_rate": ["asset_growth"],
    # Quality
    "roe": ["roe"],
    "roa": ["roa"],
    "gross_profitability": [
        "gross_profitability_sales",
        "gross_profitability_cogs",
        "gross_profitability_assets",
    ],
    "profit_margin": ["profit_margin_sales", "profit_margin_net_inc"],
    "leverage": ["leverage_eq", "leverage_debt"],
    "earnings_variability": ["earnings_variability"],
    "accruals": ["accruals_net_inc", "accruals_cf", "accruals_assets"],
    # Profitability (FF)
    "rmw_proxy": [
        "rmw_proxy_shldrs_eq",
        "rmw_proxy_ebit_oper_ps",
        "rmw_proxy_com_shs_out_secs",
    ],
    # Yield
    "dividend_growth": ["dividend_growth"],
    "payout_ratio": ["payout_ratio_eps", "payout_ratio_dps"],
    # QMJ proxy needs all quality legs
    "qmj_proxy": [
        "roe", "roa",
        "gross_profitability_sales", "gross_profitability_cogs", "gross_profitability_assets",
        "profit_margin_sales", "profit_margin_net_inc",
        "leverage_eq", "leverage_debt",
        "earnings_variability",
        "accruals_net_inc", "accruals_cf", "accruals_assets",
    ],
}

# table key -> value columns needed in panel
TABLE_KEY_VALUE_COLS: dict[str, list[str]] = {
    "mktcap": ["full_mv"],
    "eps_ltm": ["ff_eps"],
    "bps_q": ["ff_bps"],
    "sales_ps_ltm": ["ff_sales_ps"],
    "fcfe_ltm": ["ff_free_cf_fcfe"],
    "ev_q": ["ff_entrpr_value", "ff_entrpr_val"],
    "ebitda_ltm": ["ff_ebitda_oper"],
    "div_paid_ltm": ["ff_div_cf", "ff_dps"],
    "shares": ["ff_com_shs_out_secs"],
    "buyback": ["ff_stk_purch_cf", "ff_stk_sale_cf"],
    "asset_growth": ["ff_assets"],
    "roe": ["ff_roe", "fe_roe"],
    "roa": ["ff_roa"],
    "gross_profitability_sales": ["ff_sales"],
    "gross_profitability_cogs": ["ff_cogs"],
    "gross_profitability_assets": ["ff_assets"],
    "profit_margin_sales": ["ff_sales"],
    "profit_margin_net_inc": ["ff_net_inc"],
    "leverage_eq": ["ff_shldrs_eq"],
    "leverage_debt": ["ff_debt"],
    "earnings_variability": ["ff_eps"],
    "accruals_net_inc": ["ff_net_inc"],
    "accruals_cf": ["ff_free_ps_cf"],
    "accruals_assets": ["ff_assets"],
    "rmw_proxy_shldrs_eq": ["ff_shldrs_eq"],
    "rmw_proxy_ebit_oper_ps": ["ff_ebit_oper_ps"],
    "rmw_proxy_com_shs_out_secs": ["ff_com_shs_out_secs"],
    "dividend_growth": ["ff_dps"],
    "payout_ratio_eps": ["ff_eps"],
    "payout_ratio_dps": ["ff_dps"],
}


def _candidate_files() -> list[Path]:
    """Candidate Excel files under ``final/data/`` only."""
    out = []
    for p in (PART1_FILE, PART2_FILE):
        if p.exists() and p not in out:
            out.append(p)
    return out


def _find_sheet_name(xls: pd.ExcelFile, keywords: list[str]) -> str | None:
    """Find sheet by exact or substring match."""
    lowered = {s.lower(): s for s in xls.sheet_names}
    for k in keywords:
        if k.lower() in lowered:
            return lowered[k.lower()]
    for s in xls.sheet_names:
        if any(k.lower() in s.lower() for k in keywords):
            return s
    return None


def _read_prices_from_file(file_path: Path) -> tuple[pd.DataFrame, str] | None:
    """Load price data from Excel file."""
    if not file_path.exists():
        return None
    xls = pd.ExcelFile(file_path)
    sheet = _find_sheet_name(xls, ["prices", "price"])
    if sheet is None:
        return None
    df = pd.read_excel(file_path, sheet_name=sheet)
    if df.empty or df.shape[1] < 2:
        return None
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    id_columns = [c for c in df.columns if c != "date"]
    if not id_columns:
        return None
    df_long = pd.melt(
        df,
        id_vars=["date"],
        value_vars=id_columns,
        var_name="fsym_id",
        value_name="price",
    )
    df_long["price"] = pd.to_numeric(df_long["price"], errors="coerce")
    df_long = df_long.dropna(subset=["price"])
    return df_long, f"{file_path.name}:{sheet}"


def _read_coverage_from_file(file_path: Path) -> tuple[pd.DataFrame, str] | None:
    """Load symbol coverage mapping from Excel file."""
    if not file_path.exists():
        return None
    xls = pd.ExcelFile(file_path)
    sheet = _find_sheet_name(xls, ["symbol_coverage", "coverage", "mapping"])
    if sheet is None:
        return None
    df = pd.read_excel(file_path, sheet_name=sheet)
    if not {"fsym_id", "fsym_primary_listing_id"}.issubset(set(df.columns)):
        return None
    optional_cols = [
        "proper_name",
        "ticker",
        "symbol",
        "security_name",
        "company_name",
    ]
    keep_cols = ["fsym_id", "fsym_primary_listing_id"] + [c for c in optional_cols if c in df.columns]
    df = df[keep_cols].dropna(subset=["fsym_id", "fsym_primary_listing_id"])
    df = df.drop_duplicates(subset=["fsym_id", "fsym_primary_listing_id"], keep="first")
    return df, f"{file_path.name}:{sheet}"


def load_prices() -> pd.DataFrame:
    """Load price data from candidate files."""
    for file_path in _candidate_files():
        result = _read_prices_from_file(file_path)
        if result is not None:
            return result[0]
    raise FileNotFoundError(
        "No usable prices sheet found. Place metrics_part1.xlsx / metrics_part2.xlsx "
        f"under {DATA_DIR}."
    )


def load_symbol_coverage() -> pd.DataFrame:
    """Load symbol coverage mapping.

    Unions the curated 101-name Excel coverage with a DB-snapshot-derived
    full-US coverage parquet (if present), so volume/metric enrichment can
    resolve -R -> -S for the expanded universes (S&P 500, point-in-time).
    """
    frames: list[pd.DataFrame] = []
    for file_path in _candidate_files():
        result = _read_coverage_from_file(file_path)
        if result is not None:
            frames.append(result[0])
            break

    snapshot = DATA_DIR / "db_snapshot" / "symbol_coverage_full.parquet"
    if snapshot.exists():
        snap = pd.read_parquet(snapshot)
        if {"fsym_id", "fsym_primary_listing_id"}.issubset(snap.columns):
            frames.append(snap)

    if not frames:
        return pd.DataFrame(
            columns=[
                "fsym_id",
                "fsym_primary_listing_id",
                "proper_name",
                "ticker",
                "symbol",
                "security_name",
                "company_name",
            ]
        )

    out = pd.concat(frames, ignore_index=True)
    # Prefer the curated Excel rows on conflict (kept first by concat order).
    out = out.dropna(subset=["fsym_id", "fsym_primary_listing_id"]).drop_duplicates(
        subset=["fsym_primary_listing_id"], keep="first"
    )
    return out.reset_index(drop=True)


def _date_col(df: pd.DataFrame) -> str | None:
    """Pick best date column in raw metric table."""
    for c in ("date", "as_of_date", "pricedate"):
        if c in df.columns:
            return c
    return None


def _load_metric_table_with_fallback_ids(
    table_key: str,
    primary_ids: Sequence[str],
) -> pd.DataFrame:
    """
    Load metric table with ID fallback:
    1) Query using primary IDs (-R)
    2) If empty, map to fsym_id (-S) via symbol_coverage and query again
    """
    try:
        # Internal data-access helpers, not distributed; see README Data section.
        from columbia_factor_library.factor_data_access import get_metric_table
    except ImportError:
        return pd.DataFrame()

    primary_ids = list(dict.fromkeys([str(x) for x in primary_ids]))
    cov = load_symbol_coverage()
    if cov.empty:
        cov = pd.DataFrame(columns=["fsym_id", "fsym_primary_listing_id"])

    # Try primary IDs first
    df = get_metric_table(table_key, primary_ids)
    if df is not None and len(df) > 0:
        df = df.copy()
        df["fsym_primary_listing_id"] = df["fsym_id"]
        return df

    # Fallback to -S IDs when available
    if not cov.empty:
        r_to_s = (
            cov.drop_duplicates(subset=["fsym_primary_listing_id"])
            .set_index("fsym_primary_listing_id")["fsym_id"]
            .to_dict()
        )
        s_ids = [r_to_s.get(r) for r in primary_ids if r_to_s.get(r)]
        s_ids = list(dict.fromkeys(s_ids))
        if s_ids:
            df2 = get_metric_table(table_key, s_ids)
            if df2 is not None and len(df2) > 0:
                df2 = df2.copy()
                df2 = df2.merge(
                    cov[["fsym_id", "fsym_primary_listing_id"]].drop_duplicates(),
                    on="fsym_id",
                    how="left",
                )
                return df2

    return pd.DataFrame()


def _merge_metric_asof(
    panel: pd.DataFrame,
    metric_df: pd.DataFrame,
    metric_date_col: str,
    value_cols: list[str],
    *,
    lag_days: int | None = None,
    keep_asof_date_as: str | None = None,
) -> pd.DataFrame:
    """As-of merge metric columns into daily panel by company.

    Parameters
    ----------
    lag_days : days added to ``metric_date_col`` before the backward as-of join.
        ``None`` (default) uses ``FUNDAMENTAL_REPORT_LAG_DAYS`` — the approximate
        filing lag for fundamentals. Pass ``0`` for market data (volume, OHLC)
        that is observable on its own date.
    keep_asof_date_as : if given, the (lagged) as-of date that matched each
        panel row is kept under this column name (e.g. ``asset_date``), so
        report-sequence calculations downstream can group by it.
    """
    if lag_days is None:
        lag_days = int(FUNDAMENTAL_REPORT_LAG_DAYS or 0)
    if panel.empty:
        return panel
    out = panel.copy()
    if metric_df.empty:
        for c in value_cols:
            if c not in out.columns:
                out[c] = np.nan
        return out

    out["date"] = pd.to_datetime(out["date"], utc=False).astype("datetime64[ns]")
    metric_df = metric_df.copy()
    metric_df[metric_date_col] = pd.to_datetime(metric_df[metric_date_col], errors="coerce").astype("datetime64[ns]")
    if lag_days:
        metric_df[metric_date_col] = metric_df[metric_date_col] + pd.Timedelta(days=int(lag_days))
    metric_df = metric_df.dropna(subset=[metric_date_col, "fsym_primary_listing_id"])

    present_value_cols = [c for c in value_cols if c in metric_df.columns]
    if not present_value_cols:
        for c in value_cols:
            if c not in out.columns:
                out[c] = np.nan
        return out.sort_values(["fsym_primary_listing_id", "date"]).reset_index(drop=True)

    # Collapse to one row per (id, asof-date); keep latest on ties.
    metric_df = metric_df[
        ["fsym_primary_listing_id", metric_date_col] + present_value_cols
    ].drop_duplicates(
        subset=["fsym_primary_listing_id", metric_date_col], keep="last"
    )

    # Vectorized as-of merge: one grouped merge_asof over all ids at once
    # (replaces an O(n) python loop that was pathological for large universes).
    # merge_asof requires both sides *globally* sorted on the 'on' key (not
    # just within each by-group), otherwise pandas raises "left keys must be
    # sorted". Sort by date first; the id/date order is restored on return.
    out = out.sort_values(["date", "fsym_primary_listing_id"]).reset_index(drop=True)
    right = (
        metric_df.rename(columns={metric_date_col: "__asof_date"})
        .sort_values(["__asof_date", "fsym_primary_listing_id"])
        .reset_index(drop=True)
    )
    # Avoid x/y suffixes when a value column already exists in the panel:
    # stage incoming values under __new_ then combine_first to backfill NaNs.
    right_renames = {c: f"__new_{c}" for c in present_value_cols if c in out.columns}
    if right_renames:
        right = right.rename(columns=right_renames)

    merged = pd.merge_asof(
        out,
        right,
        left_on="date",
        right_on="__asof_date",
        by="fsym_primary_listing_id",
        direction="backward",
    )
    for c in present_value_cols:
        nc = f"__new_{c}"
        if nc in merged.columns:
            # Prefer freshly merged DB values; keep existing only where new is NaN.
            merged[c] = merged[nc].combine_first(merged[c]) if c in merged.columns else merged[nc]
            merged = merged.drop(columns=[nc], errors="ignore")
    if keep_asof_date_as is not None and "__asof_date" in merged.columns:
        merged[keep_asof_date_as] = merged["__asof_date"]
    merged = merged.drop(columns=["__asof_date"], errors="ignore")
    for c in value_cols:
        if c not in merged.columns:
            merged[c] = np.nan

    return merged.sort_values(["fsym_primary_listing_id", "date"]).reset_index(drop=True)


def _required_table_keys(metric_keys: Sequence[str]) -> list[str]:
    """Metric keys -> required db table keys."""
    required: set[str] = set()
    for mk in metric_keys:
        required.update(METRIC_KEY_TO_TABLE_KEYS.get(mk, []))
    # shares and asset_growth are common requirements for many formulas
    if any(m in metric_keys for m in ("market_cap", "dividend_yield", "shareholder_yield", "turnover")):
        required.add("shares")
    if "asset_growth_rate" in metric_keys:
        required.add("asset_growth")
    return sorted(required)


def _empty_volume_frame() -> pd.DataFrame:
    """Canonical empty frame for volume + OHLC liquidity inputs."""
    return pd.DataFrame(
        columns=[
            "fsym_primary_listing_id",
            "volume_date",
            "volume",
            "db_price",
            "high_price",
            "low_price",
        ]
    )


def _load_volume_from_yfinance(
    primary_ids: Sequence[str],
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Fallback liquidity panel from yfinance when DB market prices are unavailable.
    """
    mapping_path = DATA_DIR / "fsym_to_yf_ticker.json"
    if not mapping_path.exists():
        return _empty_volume_frame()
    try:
        fsym_to_ticker = json.loads(mapping_path.read_text())
    except Exception:
        return _empty_volume_frame()

    selected = {
        str(pid): fsym_to_ticker.get(str(pid))
        for pid in list(dict.fromkeys([str(x) for x in primary_ids]))
    }
    selected = {pid: t for pid, t in selected.items() if t}
    if not selected:
        return _empty_volume_frame()

    start_ts = pd.to_datetime(start_date) if start_date is not None else pd.Timestamp("1980-01-01")
    end_ts = pd.to_datetime(end_date) if end_date is not None else pd.Timestamp.today()
    tickers = list(dict.fromkeys(selected.values()))
    try:
        import yfinance as yf

        raw = yf.download(
            tickers,
            start=start_ts.strftime("%Y-%m-%d"),
            end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            group_by="ticker",
        )
    except Exception as e:
        warnings.warn(f"yfinance volume fallback failed: {e}", UserWarning, stacklevel=2)
        return _empty_volume_frame()
    if raw is None or raw.empty:
        return _empty_volume_frame()

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        available = set(raw.columns.get_level_values(0))
        for pid, ticker in selected.items():
            if ticker not in available:
                continue
            sub = raw[ticker].copy()
            if "Volume" not in sub.columns:
                continue
            sub = sub.rename(
                columns={
                    "Volume": "volume",
                    "Close": "db_price",
                    "High": "high_price",
                    "Low": "low_price",
                }
            )
            sub = sub.reset_index()
            date_col = "Date" if "Date" in sub.columns else sub.columns[0]
            sub = sub.rename(columns={date_col: "volume_date"})
            sub["fsym_primary_listing_id"] = pid
            frames.append(sub)
    else:
        ticker = tickers[0]
        sub = raw.copy()
        if "Volume" in sub.columns:
            sub = sub.rename(
                columns={
                    "Volume": "volume",
                    "Close": "db_price",
                    "High": "high_price",
                    "Low": "low_price",
                }
            )
            sub = sub.reset_index()
            date_col = "Date" if "Date" in sub.columns else sub.columns[0]
            sub = sub.rename(columns={date_col: "volume_date"})
            pid = next((k for k, v in selected.items() if v == ticker), None)
            if pid is not None:
                sub["fsym_primary_listing_id"] = pid
                frames.append(sub)

    if not frames:
        return _empty_volume_frame()

    vol = pd.concat(frames, ignore_index=True)
    vol["volume_date"] = pd.to_datetime(vol["volume_date"], errors="coerce")
    for col in ("volume", "db_price", "high_price", "low_price"):
        if col not in vol.columns:
            vol[col] = np.nan
        vol[col] = pd.to_numeric(vol[col], errors="coerce")
    vol = vol.dropna(subset=["fsym_primary_listing_id", "volume_date"])
    if vol.empty:
        return _empty_volume_frame()
    return vol[
        [
            "fsym_primary_listing_id",
            "volume_date",
            "volume",
            "db_price",
            "high_price",
            "low_price",
        ]
    ].sort_values(["fsym_primary_listing_id", "volume_date"]).reset_index(drop=True)


def load_volume(
    primary_ids: Sequence[str],
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Load daily volume from the DB price table (``PRICE_TABLE``), aggregated by fsym_id/date.
    Returns columns: fsym_primary_listing_id, volume_date, volume, db_price, high_price, low_price
    """
    cov = load_symbol_coverage()
    if cov.empty:
        return _load_volume_from_yfinance(
            primary_ids,
            start_date=start_date,
            end_date=end_date,
        )

    r_to_s = (
        cov.drop_duplicates(subset=["fsym_primary_listing_id"])
        .set_index("fsym_primary_listing_id")["fsym_id"]
        .to_dict()
    )
    s_ids = [r_to_s.get(str(r)) for r in primary_ids if r_to_s.get(str(r))]
    s_ids = list(dict.fromkeys(s_ids))
    if not s_ids:
        return _load_volume_from_yfinance(
            primary_ids,
            start_date=start_date,
            end_date=end_date,
        )

    try:
        # Internal data-access helpers, not distributed; see README Data section.
        from columbia_factor_library.db1_query_example import get_engine_for_db1_clone
    except ImportError:
        return _load_volume_from_yfinance(
            primary_ids,
            start_date=start_date,
            end_date=end_date,
        )

    try:
        from sqlalchemy import text

        # Generic schema: one row per (fsym_id, price_date) with volume and OHLC.
        sql = text(
            f"""
            SELECT
                fsym_id,
                price_date AS volume_date,
                SUM(COALESCE(volume, 0)) AS volume,
                AVG(price) AS db_price,
                MAX(high_price) AS high_price,
                MIN(low_price) AS low_price
            FROM {PRICE_TABLE}
            WHERE fsym_id IN :fsym_ids
            GROUP BY fsym_id, price_date
            """
        )
        with get_engine_for_db1_clone().connect() as conn:
            rows = conn.execute(sql, {"fsym_ids": tuple(s_ids)}).fetchall()
            cols = ["fsym_id", "volume_date", "volume", "db_price", "high_price", "low_price"]
        vol = pd.DataFrame(rows, columns=cols)
    except Exception as e:
        warnings.warn(
            f"DB load volume failed ({e}); falling back to yfinance volume/OHLC.",
            UserWarning,
            stacklevel=2,
        )
        return _load_volume_from_yfinance(
            primary_ids,
            start_date=start_date,
            end_date=end_date,
        )
    if vol.empty:
        return _load_volume_from_yfinance(
            primary_ids,
            start_date=start_date,
            end_date=end_date,
        )

    vol["volume_date"] = pd.to_datetime(vol["volume_date"])
    vol["volume"] = pd.to_numeric(vol["volume"], errors="coerce")
    vol["db_price"] = pd.to_numeric(vol["db_price"], errors="coerce")
    vol["high_price"] = pd.to_numeric(vol["high_price"], errors="coerce")
    vol["low_price"] = pd.to_numeric(vol["low_price"], errors="coerce")

    vol = vol.merge(
        cov[["fsym_id", "fsym_primary_listing_id"]].drop_duplicates(),
        on="fsym_id",
        how="left",
    )
    return vol[
        [
            "fsym_primary_listing_id",
            "volume_date",
            "volume",
            "db_price",
            "high_price",
            "low_price",
        ]
    ].dropna(subset=["fsym_primary_listing_id"])


def enrich_panel_with_db_metrics(
    panel: pd.DataFrame,
    metric_keys: Sequence[str],
) -> pd.DataFrame:
    """Load and merge required DB metrics based on requested metric_keys."""
    if panel.empty or not metric_keys:
        return panel

    out = panel.copy()
    primary_ids = out["fsym_primary_listing_id"].dropna().astype(str).unique().tolist()

    # Short-circuit: if the panel already carries the ff_* columns these
    # metric_keys need (e.g. a pre-materialized base parquet), skip the live
    # DB table loop entirely — fundamental factors then compute fully offline
    # and are immune to mid-run credential/VPN drops. Liquidity volume (below)
    # is still loaded live, as it is not part of the ff_* expected set.
    table_keys = _required_table_keys(metric_keys)

    def _column_coverage(col: str) -> float:
        if col not in out.columns:
            return 0.0
        return float(out[col].notna().mean())

    def _table_key_satisfied(tk: str) -> bool:
        cols = TABLE_KEY_VALUE_COLS.get(tk, [])
        threshold = ENRICH_COVERAGE_THRESHOLD
        # Skip mktcap pull only when vendor full_mv is already well-populated.
        if tk == "mktcap":
            return _column_coverage("full_mv") >= threshold
        if not cols:
            return False
        # Value cols within a table key are aliases — any one with enough coverage.
        return any(_column_coverage(c) >= threshold for c in cols)

    if table_keys and all(_table_key_satisfied(tk) for tk in table_keys):
        table_keys = []

    for tk in table_keys:
        try:
            raw = _load_metric_table_with_fallback_ids(tk, primary_ids)
        except Exception as e:
            warnings.warn(f"DB load failed for {tk}: {e}", UserWarning, stacklevel=2)
            continue
        if raw.empty:
            continue
        dcol = _date_col(raw)
        if dcol is None:
            continue
        need_cols = TABLE_KEY_VALUE_COLS.get(tk, [])
        present_cols = [c for c in need_cols if c in raw.columns]
        if not present_cols:
            continue
        keep = ["fsym_primary_listing_id", dcol] + present_cols
        raw = raw[keep].copy()
        # Market value (full_mv) is a daily market observable; fundamentals
        # (ff_*) get the reporting lag.
        out = _merge_metric_asof(
            out, raw, dcol, present_cols,
            lag_days=0 if tk == "mktcap" else None,
        )

    # Normalize aliases after merge
    if "ff_entrpr_value" not in out.columns and "ff_entrpr_val" in out.columns:
        out["ff_entrpr_value"] = out["ff_entrpr_val"]

    if "market_cap" not in out.columns:
        mc = pd.Series(np.nan, index=out.index, dtype=float)
        if "full_mv" in out.columns:
            mc = out["full_mv"].copy()
        if "price" in out.columns and "ff_com_shs_out_secs" in out.columns:
            # Fill any NaN full_mv rows with price × shares (handles new-name expansions)
            computed = out["price"] * out["ff_com_shs_out_secs"]
            mc = mc.fillna(computed)
        out["market_cap"] = mc

    # Liquidity requires daily volume / intraday range from DB market prices.
    # Skip the live (slow, cred-fragile) pull when a 'volume' column is already
    # materialized in the panel (e.g. step1_pit3000_enriched.parquet).
    liq_keys = {"adv_dollar", "turnover", "amihud_illiquidity", "bid_ask_spread", "pastor_stambaugh"}
    volume_already_present = _column_coverage("volume") >= ENRICH_COVERAGE_THRESHOLD
    if (liq_keys & set(metric_keys)) and not volume_already_present:
        try:
            vol = load_volume(
                primary_ids,
                start_date=out["date"].min(),
                end_date=out["date"].max(),
            )
            if not vol.empty:
                # Volume / OHLC are market data observable on their own date:
                # no reporting lag (the fundamental lag would make liquidity
                # metrics 90 days stale).
                out = _merge_metric_asof(
                    out,
                    vol.rename(columns={"volume_date": "date"}),
                    "date",
                    ["volume", "db_price", "high_price", "low_price"],
                    lag_days=0,
                )
        except Exception as e:
            warnings.warn(f"DB load volume failed: {e}", UserWarning, stacklevel=2)

    return out


def load_assets(primary_ids: Sequence[str]) -> pd.DataFrame:
    """Load asset data from DB with local fallback."""
    try:
        raw = _load_metric_table_with_fallback_ids("asset_growth", primary_ids)
        if not raw.empty and {"fsym_primary_listing_id", "date", "ff_assets"}.issubset(raw.columns):
            out = raw[["fsym_primary_listing_id", "date", "ff_assets"]].copy()
            out = out.rename(columns={"date": "asset_date"})
            out["asset_date"] = pd.to_datetime(out["asset_date"])
            return out.sort_values(["fsym_primary_listing_id", "asset_date"])
    except Exception as e:
        warnings.warn(f"DB load_assets failed ({e}). Trying local Week 6/data/.", UserWarning, stacklevel=2)

    for name in ["assets.parquet", "step1_merged_data.parquet"]:
        p = DATA_DIR / name
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "date" not in df.columns or "ff_assets" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        id_col = "fsym_primary_listing_id" if "fsym_primary_listing_id" in df.columns else "fsym_id"
        sub = df[[id_col, "date", "ff_assets"]].dropna(subset=["ff_assets"]).copy()
        sub = sub.rename(columns={id_col: "fsym_primary_listing_id", "date": "asset_date"})
        sub = sub[sub["fsym_primary_listing_id"].isin(primary_ids)]
        return sub.sort_values(["fsym_primary_listing_id", "asset_date"])
    return pd.DataFrame(columns=["fsym_primary_listing_id", "asset_date", "ff_assets"])


def load_shares(primary_ids: Sequence[str]) -> pd.DataFrame:
    """Load shares outstanding from DB with local fallback."""
    try:
        raw = _load_metric_table_with_fallback_ids("shares", primary_ids)
        if not raw.empty and {"fsym_primary_listing_id", "date", "ff_com_shs_out_secs"}.issubset(raw.columns):
            out = raw[["fsym_primary_listing_id", "date", "ff_com_shs_out_secs"]].copy()
            out = out.rename(columns={"date": "share_date"})
            out["share_date"] = pd.to_datetime(out["share_date"])
            return out.sort_values(["fsym_primary_listing_id", "share_date"])
    except Exception as e:
        warnings.warn(f"DB load_shares failed ({e}). Trying local Week 6/data/.", UserWarning, stacklevel=2)

    for name in ["shares.parquet", "step1_merged_data.parquet"]:
        p = DATA_DIR / name
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "date" not in df.columns or "ff_com_shs_out_secs" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        id_col = "fsym_primary_listing_id" if "fsym_primary_listing_id" in df.columns else "fsym_id"
        sub = df[[id_col, "date", "ff_com_shs_out_secs"]].dropna(subset=["ff_com_shs_out_secs"]).copy()
        sub = sub.rename(columns={id_col: "fsym_primary_listing_id", "date": "share_date"})
        sub = sub[sub["fsym_primary_listing_id"].isin(primary_ids)]
        return sub.sort_values(["fsym_primary_listing_id", "share_date"])
    return pd.DataFrame(columns=["fsym_primary_listing_id", "share_date", "ff_com_shs_out_secs"])


def forward_fill_price_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy helper kept for compatibility; Week 6 standard does not forward fill."""
    return df.sort_values(
        ["fsym_primary_listing_id", "date"]
        if "fsym_primary_listing_id" in df.columns
        else ["fsym_id", "date"]
    ).reset_index(drop=True)


def _merge_asof_panel(
    df: pd.DataFrame,
    df_assets: pd.DataFrame,
    df_shares: pd.DataFrame,
) -> pd.DataFrame:
    """Merge assets and shares via merge_asof (backward) by company."""
    out = df.copy()
    out = _merge_metric_asof(out, df_assets, "asset_date", ["ff_assets"], keep_asof_date_as="asset_date")
    out = _merge_metric_asof(out, df_shares, "share_date", ["ff_com_shs_out_secs"], keep_asof_date_as="share_date")
    if "asset_date" not in out.columns:
        out["asset_date"] = pd.NaT
    if "share_date" not in out.columns:
        out["share_date"] = pd.NaT
    return out.sort_values(["fsym_primary_listing_id", "date"]).reset_index(drop=True)


def _step2_paths(factor_type: str) -> list[Path]:
    """Pre-computed step2 parquet paths under ``final/data/`` only."""
    ft = factor_type.strip().lower()
    name_map = {
        "momentum": "momentum/step2_factors.parquet",
        "volatility": "volatility/step2_factors.parquet",
        "volatility / low risk": "volatility/step2_factors.parquet",
        "investment": "step2_factors.parquet",
        "size": "step2_factors.parquet",
    }
    rel = name_map.get(ft)
    if not rel:
        return []
    p = DATA_DIR / rel
    return [p] if p.exists() else []


def load_precomputed_step2(factor_type: str) -> pd.DataFrame | None:
    """Load pre-computed step2 output from ``final/app/data/`` only."""
    for p in _step2_paths(factor_type):
        try:
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            continue
    return None


def _expected_ff_columns_for_metric_keys(metric_keys: Sequence[str]) -> set[str]:
    """DB value columns (ff_*) needed for the requested metric keys."""
    cols: set[str] = set()
    for mk in metric_keys:
        for tk in METRIC_KEY_TO_TABLE_KEYS.get(mk, []):
            cols.update(TABLE_KEY_VALUE_COLS.get(tk, []))
    return cols


def warn_panel_health_after_enrich(
    df: pd.DataFrame,
    metric_keys: Sequence[str] | None,
) -> None:
    """
    After DB enrichment: warn if expected columns are missing or all-NaN.
    Does not raise — callers may still run price-only factors.
    """
    if df.empty or not metric_keys:
        return
    expected = _expected_ff_columns_for_metric_keys(metric_keys)
    for col in sorted(expected):
        if col not in df.columns:
            warnings.warn(
                f"DB enrich: expected column '{col}' not present — "
                f"metrics that depend on it may be all NaN.",
                UserWarning,
                stacklevel=2,
            )
            continue
        if df[col].notna().sum() == 0:
            warnings.warn(
                f"DB enrich: column '{col}' is all NaN — check DB connectivity, table mapping, "
                f"and universe IDs.",
                UserWarning,
                stacklevel=2,
            )


def apply_membership_mask(
    panel: pd.DataFrame,
    membership_parquet: Path | str,
) -> pd.DataFrame:
    """Keep rows where the security was in the index as of the last rebalance on or before ``date``."""
    if panel.empty:
        return panel
    mem_path = Path(membership_parquet)
    if not mem_path.exists():
        raise FileNotFoundError(f"membership_parquet not found: {mem_path.resolve()}")

    mem = pd.read_parquet(mem_path)
    date_col = "effective_date" if "effective_date" in mem.columns else "date"
    if not {"fsym_primary_listing_id", "in_top3000"}.issubset(mem.columns):
        raise ValueError(
            f"{mem_path.name} must contain fsym_primary_listing_id and in_top3000"
        )
    active = mem.loc[mem["in_top3000"].astype(bool)].copy()
    active[date_col] = pd.to_datetime(active[date_col]).astype("datetime64[ns]")
    active = active.rename(columns={date_col: "rebalance_date"})
    active = active[["rebalance_date", "fsym_primary_listing_id"]].drop_duplicates(
        subset=["fsym_primary_listing_id", "rebalance_date"], keep="last"
    )

    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    active_by_id = {
        pid: grp.sort_values("rebalance_date").reset_index(drop=True)
        for pid, grp in active.groupby("fsym_primary_listing_id", sort=False)
    }

    kept: list[pd.DataFrame] = []
    for pid, grp in out.groupby("fsym_primary_listing_id", sort=False):
        reb = active_by_id.get(pid)
        if reb is None or reb.empty:
            continue
        left = grp.sort_values("date").reset_index(drop=True)
        tagged = pd.merge_asof(
            left,
            reb[["rebalance_date"]].assign(_in_index=1),
            left_on="date",
            right_on="rebalance_date",
            direction="backward",
        )
        in_index = tagged.loc[tagged["_in_index"] == 1].drop(
            columns=["rebalance_date", "_in_index"], errors="ignore"
        )
        if not in_index.empty:
            kept.append(in_index)

    if not kept:
        return out.iloc[0:0].reset_index(drop=True)
    return pd.concat(kept, ignore_index=True)


def load_merged_panel(
    universe: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    factor_type: str | None = None,
    use_precomputed_step2: bool = True,
    metric_keys: Sequence[str] | None = None,
    base_parquet: "Path | None" = None,
    membership_parquet: "Path | None" = None,
) -> pd.DataFrame:
    """
    Load merged panel for factor calculation.

    Parameters
    ----------
    universe : list of fsym_primary_listing_id (or fsym_id). If -S IDs, maps to -R.
    start_date, end_date : optional date filter
    factor_type : factor type being computed
    use_precomputed_step2 : load pre-computed step2 parquet from ``final/data/`` if present
    metric_keys : requested metric keys; used to load required DB metric tables
    base_parquet : if provided, load this parquet as the base panel instead of frozen paths
    membership_parquet : optional monthly index membership; filters rows to in-index dates
    """
    df = None

    if base_parquet is not None:
        bp = Path(base_parquet)
        if not bp.exists():
            raise FileNotFoundError(f"base_parquet not found: {bp.resolve()}")

    # Priority 1: pre-computed step2 under final/data/ (skipped when using a custom panel)
    if base_parquet is None and use_precomputed_step2 and factor_type is not None:
        df = load_precomputed_step2(factor_type)

    # Priority 2-override: caller-supplied base panel (e.g. step1_sp500.parquet)
    if df is None and base_parquet is not None:
        df = pd.read_parquet(base_parquet)
        df["date"] = pd.to_datetime(df["date"])

    # Priority 2a: yfinance-extended step1 panel (includes post-frozen dates).
    if df is None and STEP1_EXTENDED_PARQUET.exists():
        df = pd.read_parquet(STEP1_EXTENDED_PARQUET)
        df["date"] = pd.to_datetime(df["date"])

    # Priority 2b: frozen step1 panel (original vendor CSV/Excel snapshot).
    if df is None and STEP1_MERGED_PARQUET.exists():
        df = pd.read_parquet(STEP1_MERGED_PARQUET)
        df["date"] = pd.to_datetime(df["date"])

    # Priority 3: build from raw Excel prices + DB assets/shares
    if df is None:
        prices = load_prices()
        coverage = load_symbol_coverage()

        if not coverage.empty:
            df = prices.merge(coverage, on="fsym_id", how="left")
            df["fsym_primary_listing_id"] = df["fsym_primary_listing_id"].fillna(df["fsym_id"])
        else:
            df = prices.copy()
            df["fsym_primary_listing_id"] = df["fsym_id"]

        df = df.sort_values(["fsym_primary_listing_id", "date"]).reset_index(drop=True)

        primary_ids = df["fsym_primary_listing_id"].dropna().unique().tolist()
        try:
            df_assets = load_assets(primary_ids)
            df_shares = load_shares(primary_ids)
            df = _merge_asof_panel(df, df_assets, df_shares)
        except Exception:
            df["ff_assets"] = np.nan
            df["asset_date"] = pd.NaT
            df["ff_com_shs_out_secs"] = np.nan
            df["share_date"] = pd.NaT

    # Universe filter first (reduce downstream DB merge volume)
    if universe is not None:
        primary_set = set(resolve_universe_to_primary(universe))
        df = df[df["fsym_primary_listing_id"].isin(primary_set)]

    if membership_parquet is not None:
        df = apply_membership_mask(df, membership_parquet)

    # Enrich with required DB metrics for requested factor metrics
    if metric_keys:
        df = enrich_panel_with_db_metrics(df, metric_keys)
        warn_panel_health_after_enrich(df, metric_keys)

    # Date filter last (preserve asof context first)
    if start_date:
        df = df[pd.to_datetime(df["date"]) >= pd.to_datetime(start_date)]
    if end_date:
        df = df[pd.to_datetime(df["date"]) <= pd.to_datetime(end_date)]

    if df.empty:
        warnings.warn(
            "load_merged_panel: panel is empty after load/enrich/universe/date filters. "
            "Check universe, date range, and DB connectivity.",
            UserWarning,
            stacklevel=2,
        )

    return df.sort_values(["fsym_primary_listing_id", "date"]).reset_index(drop=True)


# Deprecated alias (older notebooks)
load_week5_step2 = load_precomputed_step2


def resolve_universe_to_primary(universe: Sequence[str]) -> list[str]:
    """Map -S IDs to -R (fsym_primary_listing_id) where possible."""
    universe = [str(u).strip() for u in universe]
    r_ids = [u for u in universe if u.endswith("-R")]
    s_ids = [u for u in universe if u.endswith("-S")]
    if not s_ids:
        return universe
    cov = load_symbol_coverage()
    if cov.empty:
        return universe
    s_to_r = cov.set_index("fsym_id")["fsym_primary_listing_id"].to_dict()
    result = list(r_ids)
    for s in s_ids:
        result.append(s_to_r.get(s, s))
    return list(dict.fromkeys(result))
