# Bug Report: Non-incremental vs Incremental Encoder Divergence (first token mismatch at index 50)

## Summary
On Voxtral realtime MLX inference, non-incremental file decoding (`generate.py` path using `model.encode`) and incremental decoding (`encode_step` path) diverge deterministically at output token index **50** on the same audio/model/settings.

This appears to be an **encoder transformer full-sequence vs cached-history semantic mismatch**, not a decoder sliding-window issue.

## Scope
- Model: `mlx-community/Voxtral-Mini-4B-Realtime-6bit`
- Audio fixture: `perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_10min.wav`
- Temperature: `0.0` (greedy)
- STFT backend: `dft` (for parity runs)

## Why this report is high confidence
The divergence was reproduced on both:
- current branch state (instrumented runs), and
- original-author commit `e6d193e85e84e30f26e370c66973ce287b8a9d57`.

So this is not introduced by recent local changes.

## Minimal Repro
Run tracer (120s clip):

```bash
VOXMLX_STFT_BACKEND=dft PYTHONPATH=. .venv313/bin/python \
  scripts/trace_nonincremental_vs_incremental.py \
  --audio-path perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_10min.wav \
  --clip-seconds 120
```

Focused view around first mismatch:

```bash
VOXMLX_STFT_BACKEND=dft PYTHONPATH=. .venv313/bin/python \
  scripts/trace_nonincremental_vs_incremental.py \
  --audio-path perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_10min.wav \
  --clip-seconds 120 \
  --focus-token-index 50 \
  --focus-window 2 \
  --trace-topk 8
```

## Key Observations

### 1) Deterministic first divergence at token 50
- first token divergence index: `50`
- first meaningful divergence index: `50`
- non-incremental path becomes more `[STREAMING_PAD]`-heavy after divergence.

Artifacts:
- `perf/audio_runs/trace-noninc-vs-inc-20260209T151236Z/comparison.json`
- `perf/audio_runs/trace-noninc-vs-inc-20260209T154656Z/comparison.json`
- `perf/audio_runs/trace-noninc-vs-inc-20260209T154656Z/focus_pairwise.json`

### 2) Decoder sliding window is not primary cause
A sweep over decoder `sliding_window` (`128, 512, 2048, 8192, 16384`) keeps first divergence pinned at `50`.
Small windows degrade quality further, but do not move divergence onset.

### 3) Stage localization
- Mel parity: effectively equal.
- Conv parity: `forward_conv` vs concatenated `forward_conv_step` is exactly equal through position `133`.
  - first conv nonzero mismatch at `134` (later than divergence trigger region).
- Transformer path: full-sequence vs cached-history diverges from layer 0 with history present.

### 4) Decoder path appears consistent given input embeddings
`generate_nonincremental` == decode-from-nonincremental-embeds replay, while decode from incremental embeds differs with first divergence at token 50.

### 5) Reproduced on original-author commit
Commit `e6d193e85e84e30f26e370c66973ce287b8a9d57`:
- 120s: first divergence index `50`, token norm ~`0.254473`
- 20s: first divergence index `50`, token norm ~`0.065637`

Artifacts:
- `perf/audio_runs/commit-compare-e6d193e-20260209T000000Z/comparison.json`
- `perf/audio_runs/commit-compare-e6d193e-20s-20260209T000000Z/comparison.json`

## Ruled Out / Lower-Probability Causes
- Async scheduling race (`mx.async_eval`) as primary cause.
- STFT/mel mismatch as primary cause.
- Conv-step equivalence as primary cause.
- Decoder cache window wrap as primary cause for token-50 onset.

## Likely Root Cause Class
Encoder transformer full-sequence attention path (`encoder.__call__`) and cached incremental path (`forward_transformer(..., cache=...)`) are not numerically/semantically equivalent with history, causing early embedding drift and autoregressive argmax flip at token 50.

## Suggested Maintainer Actions
1. Treat incremental encoder path as correctness baseline for file decoding.
2. Add a regression check: first divergence index vs incremental reference on 120s fixture should be `None` (or within strict tolerance).
3. Investigate encoder full-vs-cached attention equivalence in layer 0 with history present (mask alignment / cached-history semantics in MLX SDPA).
4. Optionally reintroduce a faster batch path only if it matches incremental-reference behavior on the same trace metrics.

