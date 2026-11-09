#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
mom = pd.read_csv(ROOT / 'runtime' / 'output' / 'attribution' / 'Momentum' / 'strategy_daily.csv')
mom['date'] = pd.to_datetime(mom['date'])

print('\nMomentum factor standalone performance')
print('='*70)
print(f'Date range: {mom["date"].min().date()} - {mom["date"].max().date()}')
print(f'Trading days: {len(mom)}')
print(f'\n[Key performance metrics]')
print(f'  Cumulative return: {mom["cumsum"].iloc[-1]:.2%}')
print(f'  Annualised return: {mom["cumsum"].iloc[-1] / len(mom) * 252:.2%}')
print(f'  Mean daily return: {mom["long_short_ret"].mean():.4%}')
print(f'  Daily return std:  {mom["long_short_ret"].std():.4%}')
print(f'  Max drawdown:      {(mom["cumsum"] - mom["cumsum"].cummax()).min():.2%}')
print(f'  Win rate:          {(mom["long_short_ret"] > 0).sum() / len(mom):.2%}')

print(f'\n[Sub-period returns]')
for years in [1, 3, 5]:
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365*years)
    subset = mom[mom['date'] >= cutoff]
    if len(subset) > 0:
        ret = subset['cumsum'].iloc[-1] - subset['cumsum'].iloc[0]
        print(f'  Past {years}y: {ret:.2%}')

print(f'\nLast 5 trading days:')
print(mom[['date', 'long_short_ret', 'cumsum', 'n_long', 'n_short']].tail(5).to_string(index=False))
