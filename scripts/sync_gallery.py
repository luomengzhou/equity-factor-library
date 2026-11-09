#!/usr/bin/env python3
"""
Publish regenerable runtime outputs into docs/gallery.

`runtime/output` is the source of truth but is gitignored. `docs/gallery`
is the browsable snapshot used by docs/index.html and presentation links.
Run this after rerunning pipeline scripts so the published CSV/PNG/HTML/TXT
artifacts do not drift from the latest runtime numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from factor_library.config import DEFAULT_OUTPUT_DIR, PROJECT_ROOT


DEFAULT_SUFFIXES = (".csv", ".png", ".html", ".txt")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_artifacts(source_dir: Path, suffixes: tuple[str, ...]) -> list[Path]:
    suffix_set = {s.lower() for s in suffixes}
    return sorted(
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in suffix_set
    )


def _copy_if_changed(source: Path, target: Path, dry_run: bool) -> str:
    existed = target.exists()
    if target.exists() and _sha256(source) == _sha256(target):
        return "unchanged"

    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return "updated" if existed else "created"


def _prune_stale(
    target_dir: Path,
    valid_relative_paths: set[Path],
    suffixes: tuple[str, ...],
    dry_run: bool,
) -> int:
    stale = [
        p for p in _iter_artifacts(target_dir, suffixes)
        if p.relative_to(target_dir) not in valid_relative_paths
    ]
    if dry_run:
        return len(stale)

    for path in stale:
        path.unlink()

    # Remove empty directories left behind by stale artifact cleanup.
    for path in sorted(target_dir.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    return len(stale)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sync latest runtime/output display artifacts into docs/gallery."
    )
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--target-dir", type=Path, default=PROJECT_ROOT / "docs" / "gallery")
    ap.add_argument(
        "--suffixes",
        nargs="+",
        default=list(DEFAULT_SUFFIXES),
        help="File suffixes to publish from runtime output.",
    )
    ap.add_argument(
        "--prune-stale",
        action="store_true",
        help=(
            "Delete matching-suffix files in docs/gallery that are not present "
            "under runtime/output. Off by default to preserve gallery-only pilots."
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    source_dir = args.source_dir.resolve()
    target_dir = args.target_dir.resolve()
    suffixes = tuple(s if s.startswith(".") else f".{s}" for s in args.suffixes)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    artifacts = _iter_artifacts(source_dir, suffixes)
    counts = {"created": 0, "updated": 0, "unchanged": 0}
    valid_relative_paths: set[Path] = set()

    for source in artifacts:
        rel = source.relative_to(source_dir)
        valid_relative_paths.add(rel)
        target = target_dir / rel
        status = _copy_if_changed(source, target, args.dry_run)
        counts[status] += 1

    pruned = 0
    if args.prune_stale:
        pruned = _prune_stale(target_dir, valid_relative_paths, suffixes, args.dry_run)

    action = "Would sync" if args.dry_run else "Synced"
    print(f"{action} {len(artifacts)} artifact(s)")
    print(
        f"created={counts['created']} updated={counts['updated']} "
        f"unchanged={counts['unchanged']} pruned={pruned}"
    )
    print(f"source -> {source_dir}")
    print(f"target -> {target_dir}")


if __name__ == "__main__":
    main()
