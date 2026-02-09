# Correctness and Performance Workflow

This repository now uses two lanes:

1. Correctness lane (required)
2. Performance lane (tracked, warn-only by default)
3. Optional model-backed differential lane (off by default)
4. Audio eval lane for mono-ground-truth + speed tracking

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

## Optional Model-Backed Differential Lane

Run only when local model/audio fixtures are available.

Required env vars:
- `VOXMLX_ENABLE_MODEL_TESTS=1`
- `VOXMLX_TEST_MODEL_PATH=/absolute/path/to/model`
- `VOXMLX_TEST_AUDIO_PATH=/absolute/path/to/audio.wav`

Optional env vars (to compare original vs converted format):
- `VOXMLX_TEST_MODEL_PATH_ORIGINAL=/absolute/path/to/original/model`
- `VOXMLX_TEST_MODEL_PATH_CONVERTED=/absolute/path/to/converted/model`

Command:

```bash
VOXMLX_ENABLE_MODEL_TESTS=1 \
VOXMLX_TEST_MODEL_PATH=/path/to/model \
VOXMLX_TEST_AUDIO_PATH=/path/to/audio.wav \
python3 -m unittest tests.test_model_differential -v
```

## CI Behavior

GitHub Actions workflow `.github/workflows/correctness-performance.yml`:
- Always runs correctness tests.
- Runs microbenchmarks and checks regressions in warn-only mode.
- Uploads benchmark JSON as an artifact for trend tracking.
- Does not run model-backed differential tests by default.

## Audio Eval Lane

For real-audio perf/quality runs against the two Voxtral test audio files, use:

```bash
PYTHONPATH=. .venv313/bin/python scripts/audio_eval.py \
  --label baseline-v1-clip600-create-gt \
  --commit $(git rev-parse --short HEAD) \
  --mono-path perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_10min.wav \
  --stereo-path perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_stereo_16k_10min.wav \
  --clip-seconds 600 \
  --create-ground-truth
```

Then run comparison labels using the saved ground truth:

```bash
PYTHONPATH=. .venv313/bin/python scripts/audio_eval.py \
  --label <label> \
  --commit $(git rev-parse --short HEAD) \
  --mono-path perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_10min.wav \
  --stereo-path perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_stereo_16k_10min.wav \
  --clip-seconds 600 \
  --ground-truth-path perf/ground_truth_mono.txt
```

For repeated multi-version runs on a dedicated machine (for better stats under noisy local load), use:

```bash
PYTHONPATH=. .venv313/bin/python scripts/run_version_matrix.py \
  --matrix perf/version_matrix.json \
  --campaign macmini-voxtral-clip600
```

This runs each configured version sequentially and writes aggregate outputs to:
- `perf/batch_runs/<campaign>/summary.json`
- `perf/batch_runs/<campaign>/runs.csv`

## Non-Incremental vs Incremental Divergence Tracing

To pinpoint where non-incremental decoding starts to diverge from incremental decoding
in a semantically meaningful way (ignoring whitespace, `.`/`,` punctuation, and case),
run:

```bash
VOXMLX_STFT_BACKEND=dft PYTHONPATH=. .venv313/bin/python scripts/trace_nonincremental_vs_incremental.py \
  --audio-path perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_10min.wav \
  --clip-seconds 120
```

Outputs are written under `perf/audio_runs/trace-noninc-vs-inc-<timestamp>/`:
- `comparison.json` with token-level diffs, special-token flags, and first-divergence indices
- `summary.md` quick-readable diagnostic summary
- raw token ID and decoded text dumps for both paths

Keep the non-incremental diagnostic clip at `<=120s` to control compute cost.

For focused per-token step-through (logit top-k + producer audio position + embedding deltas):

```bash
VOXMLX_STFT_BACKEND=dft PYTHONPATH=. .venv313/bin/python scripts/trace_nonincremental_vs_incremental.py \
  --audio-path perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_10min.wav \
  --clip-seconds 120 \
  --focus-token-index 50 \
  --focus-window 2 \
  --trace-topk 8
```

This writes:
- `non_incremental_focus_trace.json`
- `incremental_focus_trace.json`
- `focus_pairwise.json`

These files are designed to answer: at the first divergence, did argmax flip because audio
embeddings diverged, context token changed, or both.
