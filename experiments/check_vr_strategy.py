#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
vr = pd.read_csv(ROOT / 'runtime' / 'output' / 'pilots' / 'value_reversal' / 'strategy_daily.csv')
vr['date'] = pd.to_datetime(vr['date'])

print('Value x Short-term Reversal strategy summary')
print('='*70)
print(f'Date range: {vr["date"].min().date()} - {vr["date"].max().date()}')
print(f'Trading days: {len(vr)}')
print(f'\n[Key performance metrics]')
print(f'  Cumulative return: {vr["cumsum"].iloc[-1]:.2%}')
print(f'  Annualised return: {vr["cumsum"].iloc[-1] / len(vr) * 252:.2%}')
print(f'  Mean daily return: {vr["long_short_ret"].mean():.4%}')
print(f'  Daily return std:  {vr["long_short_ret"].std():.4%}')
print(f'  Max drawdown:      {(vr["cumsum"] - vr["cumsum"].cummax()).min():.2%}')
print(f'  Win rate:          {(vr["long_short_ret"] > 0).sum() / len(vr):.2%}')

# Period returns
print(f'\n[Sub-period returns]')
for years in [1, 3, 5]:
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365*years)
    subset = vr[vr['date'] >= cutoff]
    if len(subset) > 0:
        ret = subset['cumsum'].iloc[-1] - subset['cumsum'].iloc[0]
        print(f'  Past {years}y: {ret:.2%}')

print(f'\n[Portfolio size]')
print(f'  Avg longs:    {vr["n_long"].mean():.1f}')
print(f'  Avg shorts:   {vr["n_short"].mean():.1f}')
print(f'  Avg universe: {vr["n_universe"].mean():.1f}')

print(f'\nLast 5 trading days:')
print(vr[['date', 'long_short_ret', 'cumsum', 'n_long', 'n_short']].tail(5).to_string(index=False))
