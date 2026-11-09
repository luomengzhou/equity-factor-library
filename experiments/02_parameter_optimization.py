#!/usr/bin/env python3
"""
Strategy Parameter Optimization Script
Systematic parameter sweep for the Value × Reversal strategy.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Resolve repo-local paths (portable across machines).
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUTPUT_DIR = ROOT / 'runtime' / 'output'
OPTIMIZATION_DIR = ROOT / 'experiments' / 'results'
OPTIMIZATION_DIR.mkdir(parents=True, exist_ok=True)

from factor_library.backtest.long_short_attribution import compute_next_day_returns_from_prices

RESULTS_LOG = OPTIMIZATION_DIR / 'optimization_log.txt'

def log(msg):
    print(msg)
    with open(RESULTS_LOG, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

def compute_past_return(prices, window=21):
    """Compute past N-day return for each stock."""
    p = prices[["fsym_primary_listing_id", "date", "price"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["fsym_primary_listing_id", "date"])
    g = p.groupby("fsym_primary_listing_id", sort=False)["price"]
    p[f"ret_{window}d_past"] = g.pct_change(window).values
    return p[["fsym_primary_listing_id", "date", f"ret_{window}d_past"]]

def backtest_strategy(value, prices, value_quantile=0.3, reversal_window=21, n_buckets=5, min_cross_section=30):
    """Run backtest and return daily strategy returns."""
    try:
        fwd = compute_next_day_returns_from_prices(prices)
        past = compute_past_return(prices, window=reversal_window)

        m = value.merge(past, on=["fsym_primary_listing_id", "date"], how="inner").merge(
            fwd, on=["fsym_primary_listing_id", "date"], how="inner"
        )
        m = m.dropna(subset=["factor_value", f"ret_{reversal_window}d_past", "ret_1d_fwd"])
        
        if m.empty:
            return None
        
        # Universe filter
        m["value_rank"] = m.groupby("date", sort=False)["factor_value"].rank(method="average", pct=True)
        uni = m[m["value_rank"] >= (1.0 - value_quantile)].copy()
        
        past_col = f"ret_{reversal_window}d_past"
        uni["reversal_score"] = -uni[past_col]
        
        # Bucketing
        s = uni["reversal_score"]
        ranks = s.groupby(uni["date"], sort=False).rank(method="average")
        n = s.groupby(uni["date"], sort=False).transform("count")
        uni["n_cs"] = n.astype("Int64")
        uni["bucket"] = (np.floor((ranks - 1) * n_buckets / n) + 1).astype("Int64")
        
        # Filter thin days
        uni = uni[uni["n_cs"] >= min_cross_section].copy()
        
        if uni.empty:
            return None
        
        # Long/short assignment
        top_b = n_buckets
        bot_b = 1
        uni["side"] = np.where(uni["bucket"] == top_b, "long", 
                              np.where(uni["bucket"] == bot_b, "short", "mid"))
        legs = uni[uni["side"].isin(["long", "short"])].copy()
        
        # Equal weight positioning
        counts = legs.groupby(["date", "side"], sort=False)["fsym_primary_listing_id"].transform("count").astype(float)
        w = 0.5 / counts
        legs["weight"] = np.where(legs["side"] == "long", w, -w)
        legs["contribution"] = legs["weight"] * legs["ret_1d_fwd"]
        
        # Daily returns
        strat = legs.groupby("date", sort=True).agg(
            long_short_ret=("contribution", "sum"),
            n_long=("side", lambda s2: int((s2 == "long").sum())),
            n_short=("side", lambda s2: int((s2 == "short").sum())),
        ).reset_index()
        
        strat["cumsum"] = strat["long_short_ret"].cumsum()
        return strat
        
    except Exception as e:
        print(f"Backtest error: {e}")
        return None

def calculate_metrics(strat_df):
    """Compute key performance metrics for a strategy."""
    if strat_df is None or len(strat_df) < 100:
        return None
    
    ret = strat_df['long_short_ret'].values
    cum = strat_df['cumsum'].values
    
    metrics = {
        'total_ret': cum[-1],
        'annual_ret': cum[-1] / len(strat_df) * 252,
        'daily_mean': np.mean(ret),
        'daily_std': np.std(ret),
        'sharpe': np.mean(ret) / np.std(ret) * np.sqrt(252) if np.std(ret) > 0 else 0,
        'max_dd': (cum - np.maximum.accumulate(cum)).min(),
        'win_rate': (ret > 0).sum() / len(ret),
        'n_days': len(strat_df)
    }
    return metrics

# Load data
log("\n" + "="*80)
log("[STEP 2] Parameter sweep optimization")
log("="*80)

log("\nLoading data...")
prices = pd.read_parquet(ROOT / 'data' / 'step1_merged_data.parquet')
value = pd.read_parquet(OUTPUT_DIR / 'Value_composite.parquet')
value = value.rename(columns={'factor_value': 'factor_value'})
value['date'] = pd.to_datetime(value['date'])

log(f"✓ Data loaded: {len(prices)} price rows, {len(value)} value factor rows")

log("\nStarting parameter sweep...")
log("\n[Parameter grid]")
log("  Value quantile: [10%, 15%, 20%, 30%, 40%, 50%]")
log("  Reversal window: [5d, 10d, 21d, 42d]")
log("  Buckets: [3, 5, 7]")

results = []
best_result = {'sharpe': -np.inf}

for value_q in [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
    for rev_window in [5, 10, 21, 42]:
        for n_buck in [3, 5, 7]:
            log(f"\n  Testing: Value_q={value_q:.0%}, rev_window={rev_window}d, n_buckets={n_buck}")
            
            strat = backtest_strategy(value, prices, 
                                    value_quantile=value_q,
                                    reversal_window=rev_window,
                                    n_buckets=n_buck)
            
            if strat is not None:
                metrics = calculate_metrics(strat)
                if metrics is not None:
                    result = {
                        'value_quantile': value_q,
                        'reversal_window': rev_window,
                        'n_buckets': n_buck,
                        **metrics
                    }
                    results.append(result)
                    
                    log(f"    → Total: {metrics['total_ret']:.2%}, Annual: {metrics['annual_ret']:.2%}, Sharpe: {metrics['sharpe']:.2f}")
                    
                    if metrics['sharpe'] > best_result['sharpe']:
                        best_result = result

log("\n\n" + "="*80)
log("[Sweep results summary]")
log("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('sharpe', ascending=False)

log("\n[Top 10 parameter combinations by Sharpe]")
log(results_df.head(10)[['value_quantile', 'reversal_window', 'n_buckets', 'total_ret', 'annual_ret', 'sharpe']].to_string())

results_df.to_csv(OPTIMIZATION_DIR / 'parameter_scan_v1.csv', index=False)
log(f"\n✓ Sweep results saved: {OPTIMIZATION_DIR / 'parameter_scan_v1.csv'}")

log(f"\n[Best parameters]")
best_params = {
    'value_quantile': best_result['value_quantile'],
    'reversal_window': best_result['reversal_window'],
    'n_buckets': best_result['n_buckets']
}
log(f"  Value quantile: {best_params['value_quantile']:.0%}")
log(f"  Reversal window: {best_params['reversal_window']}d")
log(f"  Buckets: {best_params['n_buckets']}")
log(f"  → Sharpe: {best_result['sharpe']:.2f}")
log(f"  → Annual Return: {best_result['annual_ret']:.2%}")
log(f"  → Total Return: {best_result['total_ret']:.2%}")

with open(OPTIMIZATION_DIR / 'best_params_v1.json', 'w') as f:
    json.dump(best_params, f)

log("\nOptimization complete ✓")
