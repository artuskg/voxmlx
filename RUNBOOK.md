# RUNBOOK.md — voxmlx

## Prerequisites

- Python: `python3 --version` (3.10+ recommended)
- Optional package install for local dev:
  - `python3 -m pip install -e .`
- Optional heavy dependencies for model-backed tests:
  - `mlx`, `mistral-common`, `soundfile`

## Environment/Profile Matrix

- Profile `lightweight` (default): no model downloads; fast correctness + perf guardrails.
- Profile `model-differential` (optional): requires local model path(s) and audio fixture.

## Run commands

- Correctness lane:
  - `python3 -m unittest discover -s tests -p 'test_*.py' -v`
- Perf baseline generation:
  - `python3 scripts/bench_contracts.py --iterations 20000 --output perf/baseline_contracts.json`
- Perf current run:
  - `python3 scripts/bench_contracts.py --iterations 20000 --output perf/current_contracts.json`
- Perf regression check:
  - `python3 scripts/check_perf_regression.py --baseline perf/baseline_contracts.json --current perf/current_contracts.json --threshold 0.20 --warn-only`
- Audio correctness/performance eval (segment-based, reproducible):
  - `PYTHONPATH=. .venv313/bin/python scripts/audio_eval.py --label baseline-v1-clip600-create-gt --commit $(git rev-parse --short HEAD) --mono-path perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_10min.wav --stereo-path perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_stereo_16k_10min.wav --clip-seconds 600 --create-ground-truth`
  - `PYTHONPATH=. .venv313/bin/python scripts/audio_eval.py --label <label> --commit $(git rev-parse --short HEAD) --mono-path perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_10min.wav --stereo-path perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_stereo_16k_10min.wav --clip-seconds 600 --ground-truth-path perf/ground_truth_mono.txt`
- Non-incremental vs incremental divergence trace (debug, low-overhead token tracing):
  - `VOXMLX_STFT_BACKEND=dft PYTHONPATH=. .venv313/bin/python scripts/trace_nonincremental_vs_incremental.py --audio-path perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_10min.wav --clip-seconds 120`
  - Focused step-through around first divergence token:
    - `VOXMLX_STFT_BACKEND=dft PYTHONPATH=. .venv313/bin/python scripts/trace_nonincremental_vs_incremental.py --audio-path perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_10min.wav --clip-seconds 120 --focus-token-index 50 --focus-window 2 --trace-topk 8`
  - Note: keep `--clip-seconds <= 120` for non-incremental path diagnostics.

### Optional model-backed differential tests

- Required env vars:
  - `VOXMLX_ENABLE_MODEL_TESTS=1`
  - `VOXMLX_TEST_MODEL_PATH=/absolute/path/to/model`
  - `VOXMLX_TEST_AUDIO_PATH=/absolute/path/to/audio.wav`
- Optional for format-differential assertion:
  - `VOXMLX_TEST_MODEL_PATH_ORIGINAL=/absolute/path/to/original/model`
  - `VOXMLX_TEST_MODEL_PATH_CONVERTED=/absolute/path/to/converted/model`
- Run:
  - `VOXMLX_ENABLE_MODEL_TESTS=1 VOXMLX_TEST_MODEL_PATH=... VOXMLX_TEST_AUDIO_PATH=... python3 -m unittest tests.test_model_differential -v`

## Batch version runner (Mac Mini workflow)

Use this when local machine load is noisy and you want stronger statistics.

Matrix file (editable): `perf/version_matrix.json`
- Defines versions (`id`, `ref`, `description`) to run.
- Defines baseline version, clip duration, and repeat count.

Dry run (sanity-check refs/commands only):

```bash
python3 scripts/run_version_matrix.py --matrix perf/version_matrix.json --dry-run
```

Actual run (strictly sequential):

```bash
PYTHONPATH=. .venv313/bin/python scripts/run_version_matrix.py \
  --matrix perf/version_matrix.json \
  --campaign macmini-voxtral-clip600
```

Outputs:
- Per-run raw metrics/transcripts: `perf/audio_runs/<campaign>__<version>__rNN/`\n
- Aggregated report: `perf/batch_runs/<campaign>/summary.json`\n
- Flat per-run table: `perf/batch_runs/<campaign>/runs.csv`

## Health checks

- Verify branch and cleanliness:
  - `git status --short --branch`
- Verify remote target before push:
  - `git remote -v`
- Verify lightweight lane still passes:
  - `python3 -m unittest discover -s tests -p 'test_*.py' -v`

## Restart/Resume steps

- If perf check fails unexpectedly:
  1. Re-run benchmark twice to reduce noise.
  2. Compare `perf/current_contracts.json` to `perf/baseline_contracts.json`.
  3. Keep `--warn-only` until a stable new baseline is agreed.
- If model differential tests fail:
  1. Confirm paths and env vars are set correctly.
  2. Re-run at `temperature=0.0` (already enforced by tests).
  3. Check whether failure is deterministic across two runs.

## Outputs/Artifacts

- Baseline perf artifact: `perf/baseline_contracts.json`
- Temporary perf artifact: `perf/current_contracts.json`
- Correctness/perf guide: `docs/correctness_performance.md`
- CI workflow: `.github/workflows/correctness-performance.yml`

## Escalation checklist

1. Include exact failing command and full error.
2. Report whether failure reproduces on a clean rerun.
3. Report current branch, commit SHA, and remote target.
4. For performance issues, include before/after `per_op_us` metrics.
