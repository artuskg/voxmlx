Goal (incl. success criteria):
- Identify the most likely root cause(s) of divergence in non-incremental `generate.py` after ~50 tokens, and provide a concrete, prioritized debug plan with validation steps.

Constraints/Assumptions:
- User requested no file changes except continuity-ledger creation/maintenance.
- Parallel agents may be creating/deleting files in the same repo.
- Analysis remains read-only outside this ledger file.

Key decisions:
- New session ledger: `CONTINUITY_CODEX-generate-divergence-hunt.md`.
- Use existing trace artifacts plus direct stage-by-stage reproduction to localize divergence.
- Treat decoder-window-wrap hypotheses as lower priority unless runtime evidence supports them.

State:
- In progress; cross-ledger synthesis requested and prepared.

Done:
- Verified current code paths in `voxmlx/generate.py`, `voxmlx/stream.py`, `voxmlx/model.py`, `voxmlx/encoder.py`, `voxmlx/cache.py`, and `voxmlx/language_model.py`.
- Reviewed trace artifacts in:
  - `perf/audio_runs/trace-noninc-vs-inc-20260209T151236Z/`
  - `perf/audio_runs/trace-noninc-vs-inc-20260209T154656Z/`
- Confirmed from trace: first token divergence index = 50 with decoder `sliding_window=8192`.
- Confirmed non-incremental replay matches `generate.py` output (`non_incremental_replay_matches_generate=true`).
- Ran stage-by-stage comparison on same padded audio:
  - Mel: effectively identical (max abs ~9.8e-05).
  - Conv: identical in early region including around audio positions 86–90.
  - Encoder transformer (batch vs chunked+cache): divergence starts immediately (idx0 nonzero), grows over time.
  - Final audio embeds: large drift by index ~50+ (e.g., max abs ~0.375 at index 50; ~0.127 at index 88).
- Correlated divergence point:
  - Output token divergence index 50 maps to producer audio position 88.
  - At that point both sides had same input token and cache offset, but different argmax due different logits, consistent with upstream embedding mismatch.
- Received follow-up question: whether varying `sliding_window` can provide complementary diagnostic evidence.
- User requested executing a 4-point sliding-window sweep (values different from prior 8192 run).
- Executed sequential sweep using `scripts/trace_nonincremental_vs_incremental.py` with:
  - `sliding_window=128`
  - `sliding_window=512`
  - `sliding_window=2048`
  - `sliding_window=16384`
- Wrote run artifacts outside repo to avoid workspace churn:
  - `/tmp/voxmlx_sweep/trace-sweep-sw128-20260209/`
  - `/tmp/voxmlx_sweep/trace-sweep-sw512-20260209/`
  - `/tmp/voxmlx_sweep/trace-sweep-sw2048-20260209/`
  - `/tmp/voxmlx_sweep/trace-sweep-sw16384-20260209/`
- Sweep findings:
  - First divergence index remained fixed at `50` for all tested windows.
  - `token_norm_vs_incremental` improved as window increased up to 2048, then saturated:
    - 128: `0.318543`
    - 512: `0.276159`
    - 2048: `0.254305`
    - 16384: `0.254305`
  - Compared with prior 8192 trace: 2048/8192/16384 metrics are identical (same divergence index and ratios).
  - Very small window (128) worsens non-incremental special-token ratio (`0.911921`) and text chars (`607`), but does not move divergence onset.
- Reviewed additional project ledgers:
  - `CONTINUITY_CODEX-correctness-performance.md`
  - `perf/INVESTIGATION_LEDGER_batch_vs_streaming_divergence.md`
- Confirmed runtime/model values directly:
  - `encoder_sliding_window=750`
  - `downsample_factor=4`
  - encoder-window in audio-token units: `187.5`
  - decoder config `sliding_window=8192`
- Cross-ledger evidence alignment:
  - async-eval hypothesis already ruled out by dedicated checks.
  - stage-localization and layerwise checks indicate divergence is introduced in encoder transformer layer 0 for full-vs-cached history path.
  - decode-from-embeds isolation confirms decoder path reproduces whichever embedding stream it is fed; mismatch source is upstream embeddings.
- Ran direct conv-path equivalence check (`forward_conv` vs concatenated `forward_conv_step`) on 20s and 120s clips:
  - Shapes matched exactly (`(1195,1280)` for 20s; `(6195,1280)` for 120s).
  - Outputs are exactly equal through position `133` (including positions `86..90` used at first token divergence).
  - First nonzero mismatch at position `134` (`max_abs=0.0812988`), first >`0.1` at `135`.
  - Conclusion: conv path is not the trigger for token-50 divergence; mismatch enters downstream in transformer path.

Now:
- Report integrated conclusions from all ledgers + Pro/Opus notes, including new conv equivalence result, and propose concrete forward execution plan.

Next:
- Recommend where to instrument first in encoder path.
- Propose smallest experiments to confirm whether batch encode should be replaced by incremental encode path for offline generation.
- Use `sliding_window` sweep outcomes to separate decoder-window effects from encoder-path mismatch effects.
- Propose phased remediation (correctness-first path + optional performance restoration path).

Open questions (UNCONFIRMED if needed):
- UNCONFIRMED: Whether this mismatch is intended (model expects streaming/chunked encoder semantics) vs unintended (MLX cached-causal SDPA mismatch for Q!=K lengths).
- UNCONFIRMED: Whether using chunked incremental encoder embeddings inside offline `generate.py` fully resolves transcript-quality divergence on the target audio set.
- UNCONFIRMED: Whether an encoder-side sweep/chunking experiment (not decoder-window sweep) moves first-divergence index from 50.
- UNCONFIRMED: Whether a banded-causal full-sequence encoder implementation can match incremental/cached encoder outputs closely enough to retain quality and recover speed.

Working set (files/ids/commands):
- Files: `CONTINUITY_CODEX-generate-divergence-hunt.md`, `voxmlx/generate.py`, `voxmlx/stream.py`, `voxmlx/model.py`, `voxmlx/encoder.py`, `voxmlx/cache.py`, `scripts/trace_nonincremental_vs_incremental.py`, `perf/audio_runs/trace-noninc-vs-inc-20260209T154656Z/comparison.json`.
- Commands:
  - `git status --short`
  - `rg -n ...`
  - `nl -ba ... | sed -n ...`
  - `PYTHONPATH=. .venv313/bin/python - <<'PY' ...` (stage comparison scripts)
