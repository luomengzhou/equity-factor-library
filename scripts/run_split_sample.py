#!/usr/bin/env python3
"""
Pre/post-2000 split-sample check for the Size finding.

This is a deck-defense diagnostic for the natural pushback:
"Is the Size Sharpe just a full-sample artifact, and what happens after 2000?"

It evaluates:
  - Size single-factor Q5-Q1 long/short returns
  - Equal-weight ensemble Q5-Q1 long/short returns
  - Vol-target overlays on each series
"""
from __future__ import annotations

import argparse
from pathlib import Path

import empyrical as emp
import numpy as np
import pandas as pd

from factor_library.analysis.ic import build_price_panel
from factor_library.backtest.vol_targeting import vol_targeted_returns
from factor_library.config import DATA_DIR, DEFAULT_OUTPUT_DIR
from factor_library.reports.tearsheets import long_short_daily_returns


STRATEGY_FILES = {
    "Size": DEFAULT_OUTPUT_DIR / "Size_composite.parquet",
    "Equal ensemble": DEFAULT_OUTPUT_DIR / "ensemble" / "ensemble_equal_composite.parquet",
}


def _load_prices() -> pd.DataFrame:
    extended = DATA_DIR / "step1_merged_data_extended.parquet"
    source = extended if extended.exists() else DATA_DIR / "step1_merged_data.parquet"
    step1 = pd.read_parquet(source)
    return build_price_panel(step1)


def _max_drawdown_log_space(ret: pd.Series) -> float:
    log_eq = np.log1p(ret).cumsum()
    drawdown = np.exp(log_eq - log_eq.cummax()) - 1.0
    return float(drawdown.min())


def _stats(ret: pd.Series, label: str) -> dict:
    ret = ret.dropna()
    if len(ret) == 0:
        return {
            "variant": label,
            "n_days": 0,
            "start_date": pd.NaT,
            "end_date": pd.NaT,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
        }

    mdd = _max_drawdown_log_space(ret)
    ann_return = float(emp.annual_return(ret))
    return {
        "variant": label,
        "n_days": int(len(ret)),
        "start_date": ret.index.min().date(),
        "end_date": ret.index.max().date(),
        "ann_return": ann_return,
        "ann_vol": float(ret.std(ddof=1) * np.sqrt(252)),
        "sharpe": float(emp.sharpe_ratio(ret, annualization=252)),
        "sortino": float(emp.sortino_ratio(ret, annualization=252)),
        "max_drawdown": mdd,
        "calmar": float(ann_return / abs(mdd)) if mdd < 0 else np.nan,
    }


def _vol_target_stats(
    ret: pd.Series,
    target_vols: tuple[float, ...],
    vol_lookback: int,
    max_leverage: float,
) -> list[dict]:
    rows = [_stats(ret, "raw")]
    for target_vol in target_vols:
        scaled, _ = vol_targeted_returns(
            ret,
            target_vol_annual=target_vol,
            vol_lookback=vol_lookback,
            max_leverage=max_leverage,
        )
        nonzero = scaled.ne(0)
        scaled = scaled.loc[nonzero.idxmax():] if nonzero.any() else scaled.iloc[0:0]
        rows.append(_stats(scaled, f"target {int(target_vol * 100)}%"))
    return rows


def _split_returns(ret: pd.Series, split_date: str) -> dict[str, pd.Series]:
    split = pd.to_datetime(split_date)
    return {
        "all_available": ret,
        "pre_2000": ret[ret.index < split],
        "post_2000": ret[ret.index >= split],
    }


def _sample_note(strategy: str, sample: str, n_days: int, start_date: object) -> str:
    if strategy == "Size" and sample == "pre_2000":
        return "Size composite begins in late 1999; pre-2000 sample is not meaningful."
    if n_days < 252:
        return "Less than one trading year; annualized stats are unstable."
    if pd.notna(start_date) and sample == "all_available":
        start_year = pd.to_datetime(start_date).year
        if start_year > 1980:
            return (
                f"Actual available sample starts {start_date}; "
                "do not label as 1980-2026."
            )
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre/post-2000 split-sample check for Size, equal ensemble, and vol-target overlays."
    )
    ap.add_argument("--split-date", default="2000-01-01")
    ap.add_argument("--target-vols", nargs="+", type=float, default=[0.10, 0.15, 0.20])
    ap.add_argument("--vol-lookback", type=int, default=60)
    ap.add_argument("--max-leverage", type=float, default=3.0)
    ap.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "split_sample",
        help="Directory for split-sample CSV outputs.",
    )
    args = ap.parse_args()

    prices = _load_prices()
    rows: list[dict] = []

    for strategy_name, path in STRATEGY_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing composite for {strategy_name}: {path}")

        factor_df = pd.read_parquet(path)
        factor_df["date"] = pd.to_datetime(factor_df["date"])
        ret = long_short_daily_returns(factor_df, prices)

        for sample_name, sample_ret in _split_returns(ret, args.split_date).items():
            for row in _vol_target_stats(
                sample_ret,
                target_vols=tuple(args.target_vols),
                vol_lookback=args.vol_lookback,
                max_leverage=args.max_leverage,
            ):
                row.update(
                    {
                        "strategy": strategy_name,
                        "sample": sample_name,
                        "split_date": args.split_date,
                        "note": _sample_note(
                            strategy_name,
                            sample_name,
                            int(row["n_days"]),
                            row["start_date"],
                        ),
                    }
                )
                rows.append(row)

    summary = pd.DataFrame(rows)
    columns = [
        "strategy",
        "sample",
        "variant",
        "split_date",
        "n_days",
        "start_date",
        "end_date",
        "ann_return",
        "ann_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "note",
    ]
    summary = summary[columns].sort_values(["strategy", "sample", "variant"])

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.artifact_dir / "pre_post_2000_summary.csv"
    summary.to_csv(summary_path, index=False)

    deck_table = (
        summary[summary["variant"].isin(["raw", "target 15%"])]
        .pivot_table(
            index=["strategy", "variant"],
            columns="sample",
            values="sharpe",
            aggfunc="first",
        )
        .reset_index()
    )
    deck_path = args.artifact_dir / "pre_post_2000_deck_table.csv"
    deck_table.to_csv(deck_path, index=False)

    print(f"summary -> {summary_path}")
    print(f"deck table -> {deck_path}")
    print()
    print(deck_table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
