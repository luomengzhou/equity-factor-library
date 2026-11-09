#!/usr/bin/env python3
"""
Variant strategy optimization.
Runs multiple strategy versions and logs parameters/results.
Sector rotation is not implementable with local data (no sector labels);
Value + Momentum + Low-Beta combinations are used as the practical substitute.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve repo-local paths (portable across machines).
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from factor_library.backtest.long_short_attribution import (
    LongShortConfig,
    compute_long_short_returns_and_contributions,
    compute_next_day_returns_from_prices,
)

OUTPUT_DIR = ROOT / 'runtime' / 'output'
LOG_DIR = ROOT / 'experiments' / 'results'
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / 'optimization_log.txt'
RESULT_CSV = LOG_DIR / 'variant_scan_results.csv'
BEST_JSON = LOG_DIR / 'variant_best_params.json'


def log(msg: str) -> None:
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


@dataclass
class VersionResult:
    version: str
    note: str
    value_quantile: float | None = None
    reversal_window: int | None = None
    n_buckets: int = 5
    value_weight: float | None = None
    momentum_weight: float | None = None
    lowbeta_weight: float | None = None
    total_ret: float | None = None
    annual_ret: float | None = None
    daily_mean: float | None = None
    daily_std: float | None = None
    sharpe: float | None = None
    max_dd: float | None = None
    win_rate: float | None = None
    n_days: int | None = None


def calculate_metrics(strat: pd.DataFrame) -> dict:
    ret = strat['long_short_ret'].astype(float).to_numpy()
    cum = strat['cumsum'].astype(float).to_numpy()
    daily_std = float(np.std(ret))
    daily_mean = float(np.mean(ret))
    sharpe = float(daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
    return {
        'total_ret': float(cum[-1]),
        'annual_ret': float(cum[-1] / len(strat) * 252),
        'daily_mean': daily_mean,
        'daily_std': daily_std,
        'sharpe': sharpe,
        'max_dd': float((cum - np.maximum.accumulate(cum)).min()),
        'win_rate': float((ret > 0).mean()),
        'n_days': int(len(strat)),
    }


def load_base_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = pd.read_parquet(ROOT / 'data' / 'step1_merged_data.parquet')
    prices['date'] = pd.to_datetime(prices['date'])

    value = pd.read_parquet(OUTPUT_DIR / 'Value_composite.parquet')
    value['date'] = pd.to_datetime(value['date'])

    momentum = pd.read_parquet(OUTPUT_DIR / 'Momentum_composite.parquet')
    momentum['date'] = pd.to_datetime(momentum['date'])

    low_beta_path = OUTPUT_DIR / 'Low_Beta___BAB_composite.parquet'
    lowbeta = pd.read_parquet(low_beta_path)
    lowbeta['date'] = pd.to_datetime(lowbeta['date'])

    return prices, value, momentum, lowbeta


def build_factor_df(*, value: pd.DataFrame | None = None, momentum: pd.DataFrame | None = None, lowbeta: pd.DataFrame | None = None, score_name: str = 'score') -> pd.DataFrame:
    frames = []
    if value is not None:
        frames.append(value[['fsym_primary_listing_id', 'date', 'factor_value']].rename(columns={'factor_value': 'value'}))
    if momentum is not None:
        frames.append(momentum[['fsym_primary_listing_id', 'date', 'factor_value']].rename(columns={'factor_value': 'momentum'}))
    if lowbeta is not None:
        frames.append(lowbeta[['fsym_primary_listing_id', 'date', 'factor_value']].rename(columns={'factor_value': 'lowbeta'}))

    df = frames[0]
    for frame in frames[1:]:
        df = df.merge(frame, on=['fsym_primary_listing_id', 'date'], how='inner')

    cols = [c for c in ['value', 'momentum', 'lowbeta'] if c in df.columns]
    df = df.dropna(subset=cols).copy()
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    if 'momentum' in df.columns:
        df['momentum'] = pd.to_numeric(df['momentum'], errors='coerce')
    if 'lowbeta' in df.columns:
        df['lowbeta'] = pd.to_numeric(df['lowbeta'], errors='coerce')
    return df


def run_strategy(price_df: pd.DataFrame, factor_df: pd.DataFrame, factor_col: str, *, n_buckets: int = 5, min_cross_section: int = 30, leg_gross: float = 0.5) -> pd.DataFrame | None:
    fwd = compute_next_day_returns_from_prices(price_df)
    cfg = LongShortConfig(n_buckets=n_buckets, min_cross_section=min_cross_section, leg_gross=leg_gross)
    strat, _ = compute_long_short_returns_and_contributions(
        factor_df,
        factor_col=factor_col,
        forward_1d_ret=fwd,
        cfg=cfg,
    )
    if strat.empty:
        return None
    strat = strat.copy()
    strat['cumsum'] = strat['long_short_ret'].cumsum()
    return strat


def apply_value_universe(df: pd.DataFrame, quantile: float) -> pd.DataFrame:
    out = df.copy()
    out['value_rank'] = out.groupby('date', sort=False)['value'].rank(method='average', pct=True)
    return out[out['value_rank'] >= (1.0 - quantile)].copy()


def optimize_value_reversal(price_df: pd.DataFrame, value_df: pd.DataFrame) -> list[VersionResult]:
    log('\n[Version 1] Baseline and value-reversal scan')
    results: list[VersionResult] = []

    # baseline from current implementation
    baseline_cfgs = [
        (0.30, 21, 5),
        (0.50, 10, 7),
    ]
    for idx, (q, window, buckets) in enumerate(baseline_cfgs, start=1):
        merged = build_factor_df(value=value_df)
        merged = apply_value_universe(merged, q)
        # use price-based past return via temporary merge
        prices = price_df[['fsym_primary_listing_id', 'date', 'price']].copy()
        prices['date'] = pd.to_datetime(prices['date'])
        past = prices.sort_values(['fsym_primary_listing_id', 'date'])
        past[f'ret_{window}d_past'] = past.groupby('fsym_primary_listing_id', sort=False)['price'].pct_change(window).values
        merged = merged.merge(past[['fsym_primary_listing_id', 'date', f'ret_{window}d_past']], on=['fsym_primary_listing_id', 'date'], how='inner')
        merged['score'] = -merged[f'ret_{window}d_past']
        strat = run_strategy(price_df, merged[['fsym_primary_listing_id', 'date', 'score']], 'score', n_buckets=buckets, min_cross_section=30)
        if strat is not None:
            metrics = calculate_metrics(strat)
            results.append(VersionResult(
                version=f'v1_{idx}',
                note=f'value_q={q:.0%}, reversal={window}d, buckets={buckets}',
                value_quantile=q,
                reversal_window=window,
                n_buckets=buckets,
                **metrics,
            ))
    return results


def optimize_combo(price_df: pd.DataFrame, value_df: pd.DataFrame, momentum_df: pd.DataFrame, lowbeta_df: pd.DataFrame) -> list[VersionResult]:
    log('\n[Version 2] Value + Momentum combinations')
    results: list[VersionResult] = []

    # Create merged panel
    merged_vm = build_factor_df(value=value_df, momentum=momentum_df)
    merged_vml = build_factor_df(value=value_df, momentum=momentum_df, lowbeta=lowbeta_df)

    # Use the strongest value universe from scan as default universe filter
    for q in [0.3, 0.5]:
        uni_vm = apply_value_universe(merged_vm, q)
        uni_vml = apply_value_universe(merged_vml, q)

        for wv in [0.25, 0.5, 0.75]:
            wm = 1.0 - wv
            tmp = uni_vm.copy()
            tmp['score'] = wv * tmp['value'] + wm * tmp['momentum']
            strat = run_strategy(price_df, tmp[['fsym_primary_listing_id', 'date', 'score']], 'score', n_buckets=7, min_cross_section=30)
            if strat is not None:
                metrics = calculate_metrics(strat)
                results.append(VersionResult(
                    version=f'combo_vm_q{int(q*100)}_wv{wv:.2f}',
                    note='value + momentum inside value universe',
                    value_quantile=q,
                    n_buckets=7,
                    value_weight=wv,
                    momentum_weight=wm,
                    **metrics,
                ))

        for wv in [0.25, 0.5]:
            wm = 0.35
            wb = 1.0 - wv - wm
            if wb <= 0:
                continue
            tmp = uni_vml.copy()
            tmp['score'] = wv * tmp['value'] + wm * tmp['momentum'] + wb * tmp['lowbeta']
            strat = run_strategy(price_df, tmp[['fsym_primary_listing_id', 'date', 'score']], 'score', n_buckets=7, min_cross_section=30)
            if strat is not None:
                metrics = calculate_metrics(strat)
                results.append(VersionResult(
                    version=f'combo_vml_q{int(q*100)}_wv{wv:.2f}_wm{wm:.2f}',
                    note='value + momentum + lowbeta inside value universe',
                    value_quantile=q,
                    n_buckets=7,
                    value_weight=wv,
                    momentum_weight=wm,
                    lowbeta_weight=wb,
                    **metrics,
                ))

    return results


def main() -> None:
    LOG_FILE.write_text('', encoding='utf-8')
    log('================================================================================')
    log('[STEP 3] Combination and variant strategy optimization')
    log('================================================================================')

    prices, value, momentum, lowbeta = load_base_data()
    log(f'✓ Data loaded: prices={len(prices)}, value={len(value)}, momentum={len(momentum)}, lowbeta={len(lowbeta)}')

    industry_cols = [c for c in prices.columns if any(k in c.lower() for k in ['industry', 'sector', 'gics', 'sic'])]
    if not industry_cols:
        log('⚠ No sector/industry fields found in local data; sector rotation not implementable. Using low-beta defensive substitute instead.')

    versions: list[VersionResult] = []
    versions.extend(optimize_value_reversal(prices, value))
    versions.extend(optimize_combo(prices, value, momentum, lowbeta))

    if not versions:
        log('✗ No viable versions produced')
        return

    df = pd.DataFrame([asdict(v) for v in versions])
    df = df.sort_values(['sharpe', 'annual_ret'], ascending=False)
    df.to_csv(RESULT_CSV, index=False)

    best = df.iloc[0].to_dict()
    with open(BEST_JSON, 'w', encoding='utf-8') as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    log('\n[Top 10 versions]')
    cols = ['version', 'note', 'value_quantile', 'reversal_window', 'value_weight', 'momentum_weight', 'lowbeta_weight', 'total_ret', 'annual_ret', 'sharpe', 'max_dd']
    log(df.head(10)[[c for c in cols if c in df.columns]].to_string(index=False))
    log(f'\n✓ Results saved: {RESULT_CSV}')
    log(f'✓ Best version saved: {BEST_JSON}')
    log('\n[Best version]')
    for key in ['version', 'note', 'value_quantile', 'reversal_window', 'value_weight', 'momentum_weight', 'lowbeta_weight', 'total_ret', 'annual_ret', 'sharpe', 'max_dd', 'win_rate']:
        if key in best:
            log(f'  {key}: {best[key]}')


if __name__ == '__main__':
    main()
