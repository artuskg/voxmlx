# Correctness and Performance Workflow

This repository now uses two lanes:

1. Correctness lane (required)
2. Performance lane (tracked, warn-only by default)

## Correctness Lane

Run deterministic contract tests:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

These tests cover the pure helper contracts used by runtime code:
- Weight-name remapping
- Converted/original format detection
- Weight sharding behavior
- Prompt-prefix token construction

## Performance Lane

Generate benchmark results:

```bash
python3 scripts/bench_contracts.py --iterations 20000 --output perf/current_contracts.json
```

Compare against baseline:

```bash
python3 scripts/check_perf_regression.py \
  --baseline perf/baseline_contracts.json \
  --current perf/current_contracts.json \
  --threshold 0.20
```

Use `--warn-only` during baseline calibration or noisy environments.

## CI Behavior

GitHub Actions workflow `.github/workflows/correctness-performance.yml`:
- Always runs correctness tests.
- Runs microbenchmarks and checks regressions in warn-only mode.
- Uploads benchmark JSON as an artifact for trend tracking.
