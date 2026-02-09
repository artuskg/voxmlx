# Investigation Ledger: generate.py batch divergence vs streaming

**Date:** 2026-02-09
**Symptom:** `generate.py` produces correct output for ~50 tokens, then diverges to mostly non-printable/special tokens. `stream.py` (incremental path) works correctly.
**Audio context:** ~50 output tokens ≈ 50 × 1280 samples / 16000 Hz ≈ 4 seconds of audio decoded before collapse.

---

## Prior Art: Consolidated Review (same chat session earlier today)

From our previous consolidated review of 6 independent code reviews, we established:

### What we already verified about `mask="causal"` in MLX:
- MLX **does accept** `mask="causal"` as a string parameter
- MLX uses **lower-right causal alignment** when `T_q <= T_kv` (PR #2967), which is correct for cached attention
- There is an open bug (#2835) showing MLX doesn't match PyTorch when `S_Q != S_KV` — but the mismatch is alignment convention (lower-right vs upper-left), and lower-right is what you want for KV-cached decoding
- The MLX SDPA docs (v0.30.0) confirm: `mask` can be a string `"causal"` or an array

### What we already debunked:
- `_update_in_place` cache corruption: **NOT a bug** — `S` is always 1 in that path (enforced by `update_and_fetch`)
- `forward_conv_step` tail extraction bug: **NOT a bug** — design is correct
- The "offline/streaming STFT mismatch": **NOT a bug** — both paths go through `log_mel_spectrogram_step`

### What we already confirmed as real bugs:
- Offline `generate()` drops the final pending token (but this is unrelated to the 50-token divergence)
- `mx.clear_cache()` counter uses per-call loop index, not global counter

---

## Key Insight from Awni Hannun (MLX author) on Rotating KV Cache

From his tweet about the MLX LM "infinite KV cache":
> "The invariance of self-attention to the order of inputs is a feature here."

This is critical: **self-attention is permutation-invariant with respect to KV ordering** when using `mask=None`. The ring buffer's physical disorder doesn't matter — each query still computes the same attention weights regardless of KV order.

**BUT** this only holds when `mask=None`. If you use `mask="causal"`, the mask assumes position-based ordering. A ring buffer that's physically unordered + a causal mask = incorrect masking.

---

## Hypothesis Analysis

### H1: Batch encoder doesn't enforce sliding window ⭐ STRONGEST LEAD

**The batch path** (`encoder.__call__`):
```python
mask = "causal"
for layer in self.layers:
    x = layer(x, offset=0, mask=mask)  # no cache, full sequence
```
- Processes entire sequence at once
- `mask="causal"` creates full lower-triangular mask (position i attends to all j <= i)
- No sliding window constraint — this IS correct if T_q == T_kv (square mask)

**The streaming path** (`forward_transformer` with cache):
```python
mask = "causal"
for i, layer in enumerate(self.layers):
    layer_cache = cache[i] if cache is not None else None
    x = layer(x, offset=0, mask=mask, cache=layer_cache)
```
- Processes small chunks with KV cache
- `mask="causal"` with T_q < T_kv uses lower-right alignment (correct for cached attention)
- `RotatingKVCache(750)` enforces sliding window by evicting old entries

**The difference:** The batch path allows every encoder position to attend to ALL prior positions (full causal), while the streaming path limits attention to the last 750 positions. If the model was trained with sliding window attention, the batch path violates training assumptions.

**Timing analysis:**
- 50 output tokens × 4 (downsample_factor) = 200 conv2 frames
- Plus 32 left_pad tokens × 4 = 128 conv2 frames
- Total encoder sequence length ≈ 328 frames
- `encoder.sliding_window = 750` (default)
- 328 < 750, so the sliding window ISN'T being exceeded at 50 tokens

**Problem with this hypothesis:** At 50 output tokens, the encoder sequence is still within the window. The sliding window would only matter after ~187 output tokens. So this doesn't explain 50-token divergence directly.

**However:** The batch vs incremental processing could produce different outputs even within the window, because:
1. Batch processes all 328 positions simultaneously with one causal mask
2. Incremental processes them in small chunks (4-8 frames each) with cached attention
3. These should be mathematically equivalent IF the mask and RoPE offsets are correct

### H2: The "other run" hypothesis — mel/encoder/alignment mismatch

The other Claude run you shared focused on:
1. Mel spectrogram computation differences → **Already debunked** (both use `log_mel_spectrogram_step`)
2. Audio padding differences → **Should be equivalent** (same constants)
3. Encoder RoPE offset → The batch path uses `offset=0` for all layers, which is correct because `mx.fast.rope` applies position `arange(L) + offset`. With `offset=0` and full sequence, positions are 0..L-1. With cache, `offset = cache.offset` correctly shifts for the chunk.

That run concluded: "compare the mel spectrograms first" — good advice.

### H3: Ring buffer + causal mask in streaming encoder (LATENT BUG, not current culprit)

From the Awni Hannun insight: ring buffer + `mask=None` is fine. Ring buffer + `mask="causal"` is NOT fine after wrap.

`forward_transformer` uses `mask="causal"` with `RotatingKVCache`. Once `cache.offset > max_size` (750), the cache wraps and becomes physically unordered. The causal mask will then incorrectly mask some valid past positions.

**But:** You say streaming works. So either:
- (a) Your test audio is < 15 seconds (750/4 ≈ 187 tokens × 80ms ≈ 15s), cache never wraps
- (b) The one LLM's suggestion to use `mask=None` when cache is present is correct (and happens to not break anything for streaming because within-chunk causality violation is negligible for chunks of 4 frames)

**Status:** Real bug, but only manifests for audio > 15 seconds. Not the 50-token culprit.

### H4: Numerical divergence accumulation (NEW — from the "other run" analysis)

Even if batch and incremental are mathematically equivalent, numerical differences can accumulate in autoregressive decoding. The decoder feeds back its own outputs — small differences in audio embeddings (even at 1e-6 level) can butterfly-effect into completely different token sequences after enough steps.

**This is the most plausible explanation for 50-token divergence:**
- Batch encoder produces audio embeddings that are numerically close but not identical to incremental encoder
- First ~39 tokens (prefill) absorb the small differences
- Autoregressive loop amplifies differences exponentially
- By token ~50, the sequences have diverged enough that the model makes different argmax choices
- Once one wrong token is chosen, error cascades

**Critical question:** Does the incremental path produce the SAME audio embeddings as the batch path? Even tiny differences matter because of autoregressive amplification.

### H5: `mx.async_eval` race condition

Low probability but easy to test. The double-buffer pattern:
```python
y = sample(logits)
mx.async_eval(y)
for pos in range(prefix_len, N_audio):
    next_y = step(y, pos)  # reads y
    mx.async_eval(next_y)
    token_id = y.item()     # forces eval of y
```

MLX's lazy evaluation should handle this — `y.item()` forces eval. But if there's a bug in async scheduling with in-place cache updates, it could corrupt state.

**Test:** Replace `mx.async_eval` with `mx.eval` and see if behavior changes.

---

## Revised Hypothesis Ranking

1. **H4: Numerical divergence accumulation** — Most consistent with "works for ~50 tokens then diverges"
2. **H1: Batch encoder missing sliding window** — Real difference but doesn't explain 50-token boundary
3. **H5: async_eval race** — Easy to test, could explain gradual degradation
4. **H3: Ring buffer + causal mask** — Real latent bug, not current culprit
5. **H2: Mel/alignment** — Already mostly debunked

## Recommended Diagnostic Sequence

1. **Compare audio embeddings:** Run batch `model.encode(mel)` and incremental `encode_step` chain on same padded audio. Compare L2 distance at each position.
2. **Log divergence point:** Print token ids from both paths side-by-side to find exact divergence token.
3. **Disable async_eval:** Replace with `mx.eval` in generate.py to rule out race conditions.
4. **Test large sliding window:** Force `sliding_window=100000` to rule out any window-related issues.
5. **Identify collapsed tokens:** Use `SpecialTokenPolicy.KEEP` to see what the model actually generates after collapse.

## Key Question to Ask Artus

- How long is the test audio file? If > 15 seconds, H3 (ring buffer + causal mask) becomes more relevant for the streaming path too.
- Are you comparing `generate.py` output vs `stream.py` output on the **same audio file**, or are you saying `stream.py` works on live mic while `generate.py` fails on a file?
- What's the actual `config.get("sliding_window")` value at runtime?
