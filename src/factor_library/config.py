"""
Factor library — configuration.

Paths are anchored to the project root so that `import factor_library`
users don't need to `cd` before running.
"""
from __future__ import annotations

import os
from pathlib import Path

# src/factor_library/config.py -> parents[0]=factor_library, [1]=src, [2]=project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Override via env var if the repo is relocated.
APP_DIR = Path(os.environ.get("FACTOR_LIBRARY_APP_DIR", PROJECT_ROOT))

DATA_DIR = APP_DIR / "data"

# Excel + bundled parquet inputs
FACTOR_LIBRARY_PATH = DATA_DIR / "preliminary_factor_library.xlsx"
# Raw vendor export of the price panel (not distributed); generic file names.
PART1_FILE = DATA_DIR / "metrics_part1.xlsx"
PART2_FILE = DATA_DIR / "metrics_part2.xlsx"
RF_DTB3_XLSX_PATH = DATA_DIR / "DTB3.xlsx"

# Frozen vendor panel (original CSV/Excel → parquet, ends 2026-02-02).
# Daily price table used by the optional DB volume loader; internal name is
# supplied via env rather than hard-coded.
PRICE_TABLE = os.environ.get("FACTOR_LIB_PRICE_TABLE", "prices")
STEP1_MERGED_PARQUET   = DATA_DIR / "step1_merged_data.parquet"
# Extended panel: original + yfinance-pulled rows for dates after the
# frozen file's last date. Written by scripts/refresh_prices_yfinance.py.
# Loader prefers this when present; falls back to STEP1_MERGED_PARQUET.
STEP1_EXTENDED_PARQUET = DATA_DIR / "step1_merged_data_extended.parquet"

# Runtime artefacts (regenerable; whole runtime/ tree is gitignored)
RUNTIME_DIR = APP_DIR / "runtime"
DEFAULT_CACHE_DIR = RUNTIME_DIR / "cache"
DEFAULT_OUTPUT_DIR = RUNTIME_DIR / "output"
DEFAULT_EXPORT_DIR = DEFAULT_OUTPUT_DIR / "export"
DEFAULT_EXPERIMENTS_DIR = RUNTIME_DIR / "experiments"
CACHE_SUBDIR = "factor_cache"

# Bump when factor *definition* changes (e.g. METRIC_DIRECTION) so cached parquet
# files are treated as stale.
FACTOR_CACHE_REVISION = "20260601_pit_membership_cache_base_parquet"

# Minimum non-null share within the current panel before skipping a live DB pull.
ENRICH_COVERAGE_THRESHOLD = 0.5

# Days added to fundamental ``date`` before as-of merge (approximate filing lag).
FUNDAMENTAL_REPORT_LAG_DAYS = 90

# pit3000 monthly membership artefacts
UNIVERSE_PIT3000_JSON = DATA_DIR / "universe_pit3000.json"
UNIVERSE_PIT3000_MEMBERSHIP_PARQUET = DATA_DIR / "universe_pit3000_membership.parquet"

# Factor library sheet
FACTOR_METRICS_SHEET = "Factor_Metrics"
FACTOR_TYPE_COL = "Factor Type"
METRIC_FORMULA_COL = "Metric / Formula"
