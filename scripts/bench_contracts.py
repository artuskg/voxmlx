#!/usr/bin/env python3
"""Microbenchmarks for pure contract helpers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voxmlx.contracts import make_weight_shards, remap_weight_name


@dataclass
class DummyWeight:
    nbytes: int


def _bench(func, iterations: int) -> dict:
    # warmup
    for _ in range(100):
        func()

    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start

    return {
        "iterations": iterations,
        "total_seconds": elapsed,
        "per_op_us": (elapsed / iterations) * 1_000_000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark voxmlx contract helpers")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("perf/current_contracts.json"))
    args = parser.parse_args()

    known_name = "layers.3.feed_forward.w2.weight"
    unknown_name = "unknown.weight"

    weights = OrderedDict((f"w{i}", DummyWeight(nbytes=(i % 7 + 1) * 1024)) for i in range(200))

    benchmarks = {
        "remap_known": _bench(lambda: remap_weight_name(known_name), args.iterations),
        "remap_unknown": _bench(lambda: remap_weight_name(unknown_name), args.iterations),
        "make_weight_shards": _bench(lambda: make_weight_shards(weights, max_file_size_gb=0), args.iterations),
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmarks": benchmarks,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"[perf] wrote benchmark results to {args.output}")
    for name, data in benchmarks.items():
        print(f"[perf] {name}: {data['per_op_us']:.3f} us/op over {data['iterations']} iterations")


if __name__ == "__main__":
    main()
