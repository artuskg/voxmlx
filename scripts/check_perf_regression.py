#!/usr/bin/env python3
"""Compare benchmark output to baseline and flag regressions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Check perf regressions")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.20, help="Allowed slowdown ratio; 0.20 = 20%%")
    parser.add_argument("--warn-only", action="store_true")
    args = parser.parse_args()

    if not args.current.exists():
        print(f"[perf] ERROR: current results file missing: {args.current}")
        return 1

    if not args.baseline.exists():
        print(f"[perf] INFO: baseline file missing: {args.baseline}; skipping regression check")
        return 0

    baseline = _load_json(args.baseline)
    current = _load_json(args.current)

    baseline_benchmarks = baseline.get("benchmarks", {})
    current_benchmarks = current.get("benchmarks", {})

    if not baseline_benchmarks:
        print("[perf] INFO: baseline has no benchmarks; skipping regression check")
        return 0

    regressions = []
    for name, baseline_data in baseline_benchmarks.items():
        if name not in current_benchmarks:
            continue

        baseline_us = baseline_data.get("per_op_us")
        current_us = current_benchmarks[name].get("per_op_us")
        if baseline_us is None or current_us is None:
            continue

        allowed = baseline_us * (1.0 + args.threshold)
        if current_us > allowed:
            regressions.append((name, baseline_us, current_us, args.threshold))

    if not regressions:
        print("[perf] OK: no regressions above threshold")
        return 0

    print("[perf] Regressions detected:")
    for name, baseline_us, current_us, threshold in regressions:
        pct = ((current_us / baseline_us) - 1.0) * 100.0
        print(
            f"  - {name}: baseline={baseline_us:.3f} us/op current={current_us:.3f} us/op "
            f"({pct:.1f}% slower; threshold={threshold * 100:.1f}%)"
        )

    if args.warn_only:
        print("[perf] WARN-ONLY mode enabled; not failing")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
