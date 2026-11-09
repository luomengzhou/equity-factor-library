# Equity Factor Library

A research library for building and stress-testing equity factors, written for my Columbia MSBA capstone with an index-provider partner. It turns raw price and fundamental panels into 12 cross-sectional factor composites. It then puts each factor through the same validation stack: IC decay, quantile monotonicity, transaction costs, out-of-sample splits, regime conditioning, concentration checks, and multi-factor ensembles.

> **What is in this repo:** the library and pipeline code (~11k lines of Python, all written by me).
> **What is not:** the vendor data panels, the partner's internal database helpers, and the result tables built from that data. The capstone results were produced on licensed data that I cannot redistribute, so this repo shows the design and the methodology. I'm happy to walk through the findings in conversation.

## Factors

12 composites. Each one is built from several underlying metrics: rank each metric cross-sectionally by date, then combine.

| Group | Factors | Example metrics |
|---|---|---|
| Price-based | Market, Momentum, Volatility / Low Risk, Low Beta / BAB, Liquidity | 12-1 momentum, 52-week-high, idiosyncratic vol, downside vol, Amihud illiquidity |
| Fundamental | Value, Yield, Size, Investment, Quality, Quality (Composite), Profitability (FF) | E/P, B/P, EV/EBITDA, shareholder yield, asset growth, ROE, gross profitability, accruals, Piotroski F-score |

`Quality (Composite)` is a residual sleeve. A QMJ-style raw score is orthogonalised cross-sectionally against the plain Quality factor on each date, so it carries only incremental information.

## Architecture

```
                 ┌──────────────── data/ ────────────────┐
 price + funda   │ loader.py   point-in-time as-of merge │
 panels ───────► │             (90-day reporting lag),   │
                 │             universe membership mask  │
                 │ cache.py    versioned parquet cache   │
                 └───────────────────┬───────────────────┘
                                     ▼
                 ┌──────────────── factors/ ─────────────┐
                 │ registry.py   factor → metric mapping  │
                 │ calculators/  one module per family    │
                 │ base.py       winsorise · rank · combine│
                 │ get_factor()  single public entry point│
                 └───────────────────┬───────────────────┘
                                     ▼
  ┌─────────── analysis/ ──────────┐   ┌─────────── backtest/ ───────────┐
  │ ic.py           IC decay,      │   │ quantiles.py    Q1…Q5 fans      │
  │                 rolling IC     │   │ costs.py        turnover × bps  │
  │ out_of_sample   split-date OOS │   │                 rebalance grid  │
  │ multi_factor    FF attribution │   │ vol_targeting   risk overlay    │
  │ ensemble        EW / inv-vol / │   │ long_short_attr concentration,  │
  │                 IC-weighted    │   │                 leave-one-out   │
  │ benchmark_reg   L/S vs long-   │   │ signal_weighted EW vs signal    │
  │                 only framing   │   │                 weighting       │
  └────────────────────────────────┘   └─────────────────────────────────┘
                 regime/  (vol / bull-bear / size buckets → conditional IR)
                 reports/ (quantstats tear sheets)   viz/ (shared style)
```

```python
from factor_library.factors.get_factor import get_factor

mom = get_factor(universe, "Momentum",
                 start_date="2000-01-01",
                 composite_method="equal_weight",   # or "inverse_volatility"
                 winsorize_bounds=(0.01, 0.99))
```

## Validation pipeline

`scripts/` has one CLI per stage. `scripts/run_storyline.py` runs them in order.

| Stage | Script | Question it answers |
|---|---|---|
| Build | `run_pipeline.py` | Compute all 12 composites (cached, reproducible) |
| Signal | `run_stage2a.py` | Does the factor predict returns, and over what horizon? (IC decay, rolling IC, tear sheets) |
| Robustness | `run_stage2b.py` | Does it survive costs, factor attribution, and an out-of-sample split? |
| Monotonicity | `run_quantiles.py` | Do Q1→Q5 returns line up in order, or does only one tail work? |
| Concentration | `run_concentration.py` | Is the factor riding on a handful of names? (top-5 share, effective N, leave-one-out) |
| Regime | `run_regime.py`, `run_regime_rotation.py` | Does it work only in some market states, and is rotation worth it? |
| Portfolio | `run_ensemble.py`, `run_factor_overlap.py` | How should factors be combined? Signal correlation vs. long-leg overlap (Jaccard) |
| Risk | `run_vol_targeting.py` | What does a vol-target overlay do to drawdown and tails? |
| Framing | `run_benchmark_regression.py`, `run_beta_four_objects.py` | Is the "alpha" really alpha once you compare long/short to long-only and decompose beta? |
| Robustness | `run_split_sample.py`, `experiments/` | Pre/post-2000 stability; parameter and variant sweeps |

## Install and run

```bash
pip install -e .                      # installs the `factor_library` package from src/
pytest                                # synthetic-data smoke tests (no vendor data needed)
python scripts/run_pipeline.py        # needs the data panel under data/ (see Known limitations)
python scripts/run_storyline.py --dry-run
```

The `tests/` folder checks the parts that do not need data: the point-in-time
as-of merge (reporting lag applied, no cross-id contamination), that momentum
uses only past prices, cross-sectional ranking / compositing, winsorisation,
and the factor-cache key.

## Design choices worth noting

- **Point-in-time fundamentals.** Fundamentals are merged with a configurable reporting lag (`FUNDAMENTAL_REPORT_LAG_DAYS`, default 90) before the as-of join, so no factor uses a number before it would have been public.
- **Universe handling.** A monthly membership mask lets the same code run on a fixed large-cap panel or a point-in-time top-N universe. This makes survivorship bias something you can measure instead of ignore.
- **After-cost by default.** Long/short books carry explicit turnover. Sharpe is reported over a grid of cost (bps per side) × rebalance frequency (daily / weekly / monthly).
- **Versioned cache.** Factor outputs are cached under a revision key that changes when a factor definition changes, so stale results can't silently leak into later stages.
- **Negative results kept.** For example, the IC-weighted ensemble was tested and did not beat equal weighting. The pipeline keeps such variants so the comparison stays reproducible.

## Known limitations

- Winsorisation bounds for derived signals are global (full-sample) quantiles. This is a mild look-ahead; rolling bounds would be stricter.
- The 90-day reporting lag is an approximation, not vendor "available-date" fields. It is applied when fundamentals are joined onto the daily panel by `data/loader.py`. When a pre-merged panel that already carries the `ff_*` columns is loaded, the loader trusts that panel's alignment and does not re-lag it, so the point-in-time guarantee is only as good as the panel construction.
- Some fundamental signals are computed on the daily (as-of filled) panel rather than on the quarterly report sequence (`earnings_variability` rolling std, the Piotroski leverage-improvement check). Between report dates these are step functions, so they carry less information than their names suggest.
- The code expects the panel schema of the original dataset. Running it on public data needs a small adapter for `data/loader.py` (`scripts/refresh_prices_yfinance.py` shows the price side).

## Data

The loader looks for its inputs under `data/` (override with `FACTOR_LIBRARY_APP_DIR`):

- `metrics_part1.xlsx` / `metrics_part2.xlsx` — raw price panel plus a symbol-coverage sheet (vendor export, not distributed).
- `step1_merged_data.parquet` — the frozen merged daily panel; `step1_merged_data_extended.parquet` is the same panel extended with yfinance rows by `scripts/refresh_prices_yfinance.py`.
- `DTB3.xlsx` — 3-month T-bill rate for the risk-free leg.

Fundamentals use FactSet-style `ff_*` field names (e.g. `ff_assets`, `ff_com_shs_out_secs`). Volume and point-in-time fundamentals were pulled through the partner's internal data-access helpers, which are not part of this repo; the loader imports them inside `try/except ImportError` and falls back to the local panel or yfinance when they are absent. The daily price table name is read from `FACTOR_LIB_PRICE_TABLE` (default `prices`) so the SQL in `data/loader.py` has no hard-coded internal schema.

## Stack

`pandas` · `numpy` · `scipy` · `statsmodels` · `alphalens-reloaded` · `empyrical-reloaded` · `pyfolio-reloaded` · `quantstats` · `matplotlib` / `seaborn`

## License

MIT
