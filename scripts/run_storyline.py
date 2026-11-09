#!/usr/bin/env python3
"""
Run the project in presentation order.

Default: core storyline only.
Optional flags add appendix diagnostics and exploratory pilots.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


CORE_STEPS: list[list[str]] = [
    ["python", "scripts/run_pipeline.py"],
    ["python", "scripts/run_stage2a.py"],
    ["python", "scripts/run_stage2b.py"],
    ["python", "scripts/run_ensemble.py"],
    ["python", "scripts/run_vol_targeting.py", "--all-factors", "--include-ensemble"],
    ["python", "scripts/run_benchmark_regression.py"],
]

DIAGNOSTIC_STEPS: list[list[str]] = [
    ["python", "scripts/run_quantiles.py"],
    ["python", "scripts/run_concentration.py"],
    ["python", "scripts/run_regime.py"],
    ["python", "scripts/run_regime_rotation.py"],
    ["python", "scripts/run_signal_weighted.py"],
    ["python", "scripts/run_beta_four_objects.py"],
]

PILOT_STEPS: list[list[str]] = [
    ["python", "scripts/run_value_reversal.py"],
    ["python", "scripts/run_style_reversal_pilot.py"],
    ["python", "scripts/run_qvr.py", "--artifact-dir", "docs/gallery/qvr"],
]

PUBLISH_STEPS: list[list[str]] = [
    ["python", "scripts/sync_gallery.py"],
]


def _fmt(cmd: list[str]) -> str:
    return " ".join(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run capstone scripts in storyline order")
    ap.add_argument("--include-diagnostics", action="store_true")
    ap.add_argument("--include-pilots", action="store_true")
    ap.add_argument(
        "--sync-gallery",
        action="store_true",
        help="Publish latest runtime/output CSV/PNG/HTML/TXT artifacts to docs/gallery.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = ap.parse_args()

    steps: list[list[str]] = []
    steps.extend(CORE_STEPS)
    if args.include_diagnostics:
        steps.extend(DIAGNOSTIC_STEPS)
    if args.include_pilots:
        steps.extend(PILOT_STEPS)
    if args.sync_gallery:
        steps.extend(PUBLISH_STEPS)

    print("Storyline run plan:")
    for i, cmd in enumerate(steps, 1):
        print(f"  [{i:02d}/{len(steps):02d}] {_fmt(cmd)}")
    if args.dry_run:
        return

    t0 = time.time()
    repo_root = Path(__file__).resolve().parents[1]
    for i, cmd in enumerate(steps, 1):
        print(f"\n===== [{i}/{len(steps)}] START: {_fmt(cmd)}")
        rc = subprocess.run(cmd, cwd=repo_root).returncode
        print(f"===== [{i}/{len(steps)}] END rc={rc}: {_fmt(cmd)}")
        if rc != 0:
            print(f"\nFAILED at step {i}: {_fmt(cmd)}")
            sys.exit(rc)

    print(f"\nAll selected steps finished in {(time.time() - t0)/60:.2f} min")


if __name__ == "__main__":
    main()
