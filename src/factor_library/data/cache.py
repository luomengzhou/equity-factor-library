"""
Factor result cache — key generation, read, write.

The cache key is a hash of (universe, factor_name, date range, rank_method,
composite_method, winsorize, base_parquet, use_precomputed_step2,
``FACTOR_CACHE_REVISION``). Bump the revision in
``factor_library.config`` when a factor definition changes so stale parquets
are not served.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from factor_library.config import (
    CACHE_SUBDIR,
    DEFAULT_CACHE_DIR,
    FACTOR_CACHE_REVISION,
)


def _make_cache_key(
    universe: list[str],
    factor_name: str,
    start_date: str | None,
    end_date: str | None,
    rank_method: str,
    composite_method: str,
    winsorize: bool,
    *,
    base_parquet: Path | str | None = None,
    use_precomputed_step2: bool = True,
    membership_parquet: Path | str | None = None,
    winsorize_bounds: tuple[float, float] | None = None,
) -> str:
    base_key = None
    if base_parquet is not None:
        base_key = str(Path(base_parquet).resolve())
    mem_key = None
    if membership_parquet is not None:
        mem_key = str(Path(membership_parquet).resolve())
    payload = {
        "universe": sorted(str(u) for u in universe),
        "factor_name": factor_name.strip(),
        "start_date": start_date,
        "end_date": end_date,
        "rank_method": rank_method,
        "composite_method": composite_method,
        "winsorize": winsorize,
        "winsorize_bounds": (
            [float(winsorize_bounds[0]), float(winsorize_bounds[1])]
            if winsorize and winsorize_bounds is not None else None
        ),
        "base_parquet": base_key,
        "use_precomputed_step2": use_precomputed_step2,
        "membership_parquet": mem_key,
        "factor_cache_revision": FACTOR_CACHE_REVISION,
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in factor_name.strip())
    return f"{safe}_{h}"


def get_cache_path(cache_dir: Path | str | None, cache_key: str) -> Path:
    base = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    sub = base / CACHE_SUBDIR
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{cache_key}.parquet"


def read_from_cache(cache_path: Path) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    try:
        return pd.read_parquet(cache_path)
    except Exception:
        return None


def write_to_cache(df: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)


def get_cached_factor(
    universe: list[str],
    factor_name: str,
    start_date: str | None,
    end_date: str | None,
    rank_method: str,
    composite_method: str,
    winsorize: bool,
    cache_dir: Path | str | None,
    *,
    base_parquet: Path | str | None = None,
    use_precomputed_step2: bool = True,
    membership_parquet: Path | str | None = None,
    winsorize_bounds: tuple[float, float] | None = None,
) -> pd.DataFrame | None:
    path = get_cache_path(cache_dir, _make_cache_key(
        universe, factor_name, start_date, end_date,
        rank_method, composite_method, winsorize,
        base_parquet=base_parquet,
        use_precomputed_step2=use_precomputed_step2,
        membership_parquet=membership_parquet,
        winsorize_bounds=winsorize_bounds,
    ))
    return read_from_cache(path)


def save_factor_to_cache(
    df: pd.DataFrame,
    universe: list[str],
    factor_name: str,
    start_date: str | None,
    end_date: str | None,
    rank_method: str,
    composite_method: str,
    winsorize: bool,
    cache_dir: Path | str | None,
    *,
    base_parquet: Path | str | None = None,
    use_precomputed_step2: bool = True,
    membership_parquet: Path | str | None = None,
    winsorize_bounds: tuple[float, float] | None = None,
) -> Path:
    path = get_cache_path(cache_dir, _make_cache_key(
        universe, factor_name, start_date, end_date,
        rank_method, composite_method, winsorize,
        base_parquet=base_parquet,
        use_precomputed_step2=use_precomputed_step2,
        membership_parquet=membership_parquet,
        winsorize_bounds=winsorize_bounds,
    ))
    write_to_cache(df, path)
    return path
