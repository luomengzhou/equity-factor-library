#!/usr/bin/env python3
"""
Refresh the price panel by pulling every ticker from yfinance and appending
the new daily rows to the existing ``step1_merged_data.parquet``.

Non-destructive: the original file is never overwritten. A new
``step1_merged_data_extended.parquet`` is written side-by-side. The loader
prefers the extended file when present.

Fundamentals (ff_assets, ff_com_shs_out_secs) on the new rows are
forward-filled from the most recent known value per ticker — fundamentals
are quarterly, so 2-3 months of forward fill is a sensible approximation
until a full DB refresh is run.

Usage:
    python scripts/refresh_prices_yfinance.py                 # pull all 101
    python scripts/refresh_prices_yfinance.py --pilot         # 5 tickers only
    python scripts/refresh_prices_yfinance.py --start 2026-02-03 --end 2026-04-19
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
import yfinance as yf  # noqa: E402

from factor_library.config import DATA_DIR


PILOT_TICKERS = ["MH33D6-R", "P8R3C2-R", "QLGSL2-R", "KV0J41-R", "M1FW60-R"]


def load_mapping() -> dict[str, str]:
    p = DATA_DIR / "fsym_to_yf_ticker.json"
    if not p.exists():
        raise FileNotFoundError(f"missing mapping file {p}; run the mapping cell first")
    return json.loads(p.read_text())


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Return a wide DataFrame of close prices (split/div adjusted) keyed by date×ticker."""
    print(f"    yfinance: {len(tickers)} tickers, {start} -> {end}")
    df = yf.download(tickers, start=start, end=end,
                     auto_adjust=True, progress=False, group_by="ticker")
    # Normalise to a single-level column frame of Close
    if isinstance(df.columns, pd.MultiIndex):
        close = df.xs("Close", axis=1, level=1)
    else:
        close = df[["Close"]].rename(columns={"Close": tickers[0]})
    return close.dropna(how="all")


def build_new_rows(
    close_wide: pd.DataFrame,
    fsym_to_ticker: dict[str, str],
    step1_existing: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert yfinance wide close → long panel with the same schema as
    step1_merged_data.parquet (date, fsym_id, price, fsym_primary_listing_id,
    asset_date, ff_assets, share_date, ff_com_shs_out_secs).

    Fundamentals are forward-filled from the last known value in the
    existing panel per fsym_primary_listing_id.
    """
    rows = []
    last_fund = (step1_existing
                 .sort_values(["fsym_primary_listing_id", "date"])
                 .groupby("fsym_primary_listing_id")
                 .tail(1)
                 .set_index("fsym_primary_listing_id"))

    # we need fsym_id too — take last known per fsym_primary_listing_id
    fsym_id_lookup = (step1_existing
                      .drop_duplicates("fsym_primary_listing_id", keep="last")
                      .set_index("fsym_primary_listing_id")["fsym_id"]
                      .to_dict())

    for fsym, ticker in fsym_to_ticker.items():
        if ticker not in close_wide.columns:
            continue
        series = close_wide[ticker].dropna()
        if series.empty:
            continue
        for date, px in series.items():
            last = last_fund.loc[fsym] if fsym in last_fund.index else None
            rows.append({
                "date":                   pd.Timestamp(date).normalize(),
                "fsym_id":                fsym_id_lookup.get(fsym, fsym),
                "price":                  float(px),
                "fsym_primary_listing_id": fsym,
                "asset_date":             last["asset_date"] if last is not None else pd.NaT,
                "ff_assets":              last["ff_assets"] if last is not None else float("nan"),
                "share_date":             last["share_date"] if last is not None else pd.NaT,
                "ff_com_shs_out_secs":    last["ff_com_shs_out_secs"] if last is not None else float("nan"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh price panel via yfinance")
    ap.add_argument("--pilot", action="store_true",
                    help="Only pull 5 pilot tickers (AAPL/MSFT/META/JNJ/UNP)")
    ap.add_argument("--start", default=None,
                    help="YYYY-MM-DD; default = day after last date in step1")
    ap.add_argument("--end", default=None,
                    help="YYYY-MM-DD; default = today")
    ap.add_argument("--output-name", default="step1_merged_data_extended.parquet",
                    help="Output parquet filename under final/app/data/")
    args = ap.parse_args()

    print("[1/4] Loading existing step1 panel + mapping…")
    step1 = pd.read_parquet(DATA_DIR / "step1_merged_data.parquet")
    step1["date"] = pd.to_datetime(step1["date"])
    mapping = load_mapping()
    if args.pilot:
        mapping = {k: mapping[k] for k in PILOT_TICKERS}
    print(f"    step1 last date: {step1['date'].max().date()}")
    print(f"    mapping: {len(mapping)} tickers")

    start = args.start or (step1["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"\n[2/4] Pulling yfinance close prices {start} -> {end}…")
    close = download_prices(list(mapping.values()), start=start, end=end)
    print(f"    got {close.shape[0]} dates × {close.shape[1]} tickers")

    print("\n[3/4] Building new rows with forward-filled fundamentals…")
    new_rows = build_new_rows(close, mapping, step1)
    new_rows = new_rows[new_rows["date"] > step1["date"].max()]
    print(f"    {len(new_rows):,} new rows after {step1['date'].max().date()}")

    if new_rows.empty:
        print("\n    No new rows to append. Source already up to date.")
        return

    # Concatenate + dedup (defensive) + sort
    extended = pd.concat([step1, new_rows], ignore_index=True)
    extended = (extended
                .drop_duplicates(subset=["fsym_primary_listing_id", "date"], keep="last")
                .sort_values(["fsym_primary_listing_id", "date"])
                .reset_index(drop=True))

    out_path = DATA_DIR / args.output_name
    extended.to_parquet(out_path, index=False)
    print(f"\n[4/4] Wrote {out_path}")
    print(f"    total rows: {len(extended):,}  (was {len(step1):,}, +{len(new_rows):,})")
    print(f"    date range: {extended['date'].min().date()} -> {extended['date'].max().date()}")


if __name__ == "__main__":
    main()
