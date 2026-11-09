# Experiments — Apr 17 ad-hoc scratch

One-off scripts used to explore factor parameters and validate pilot strategies before the broader refactor. Kept here for reproducibility of `results/best_params_v1.json`.

## Scripts

| Script | Purpose |
|---|---|
| `01_validate_factors.py` | Integrity sweep over every factor parquet: coverage, non-null rate, percentile health |
| `02_parameter_optimization.py` | Parameter scan for the value-reversal pilot (6 × 4 × 3 = 72 combos) |
| `03_variant_optimization.py` | Multi-variant strategy comparison: value+momentum, value+momentum+low-beta |
| `check_momentum.py` | Quick spot-check of the Momentum composite |
| `check_vr_strategy.py` | Quick spot-check of the value-reversal strategy |

## Results

- `results/best_params_v1.json` — tuned parameters selected from `02_parameter_optimization.py`
- `results/parameter_scan_v1.csv` — full sweep output (all 72 combos, ranked by Sharpe)
- `results/quick_variant_scan.csv` — variant comparison from `03_variant_optimization.py`

## Status

These scripts now use repo-relative paths (portable across machines) but still contain duplicated ad-hoc back-test logic. In the next refactor pass, their reusable logic should be extracted into `src/factor_library/backtest/` (see project root README / plan).

Not imported by the package. Run standalone:

```bash
python experiments/02_parameter_optimization.py
```
