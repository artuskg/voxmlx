Goal (incl. success criteria):
- Achieve at least 10% performance improvement on real audio tests while preserving transcription quality against a mono-derived ground truth target.
- Success criteria:
  - Use both files in `../vllm/voxtral_test_audio`.
  - Create ground truth target from the mono file.
  - Maintain a reproducible performance table in this ledger with labels, change chain, commit/config, and metrics.
  - Report deviations from ground truth with meaningful metrics and verify deviations remain small enough.

Constraints/Assumptions:
- Keep changes compatible with existing Python package structure.
- Audio fixtures:
  - `../vllm/voxtral_test_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k.wav`
  - `../vllm/voxtral_test_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_stereo_16k.wav`
- Ground truth target is derived from mono baseline run (`perf/ground_truth_mono.txt`).
- Benchmark config for iteration speed/reproducibility: first 180s of each file + 10s warmup.

Key decisions:
- Use branch `codex/correctness-performance-scaffold`.
- Use `scripts/audio_eval.py` for reproducible runs (fixed clip length, warmup, model path, labeled outputs).
- Use metrics:
  - `norm_edit_distance`: character-level Levenshtein / GT chars
  - `token_error_ratio`: token-level Levenshtein / GT tokens
- Accept quality if deviation does not worsen versus baseline while speed improves.

State:
- Done: implemented and validated >=10% speedup while preserving baseline deviation profile.
- Now: finalize with rerun results recorded and pushed.
- Next: optional larger-window/full-file confirmation runs.

Done:
- Added deterministic correctness/perf scaffold and CI.
- Added optional model-backed differential tests and local project docs (`AGENTS.md`, `RUNBOOK.md`).
- Installed runtime deps in `.venv313` and loaded `mlx-community/Voxtral-Mini-4B-Realtime-6bit`.
- Added reproducible audio evaluator: `scripts/audio_eval.py`.
- Added optimization chain:
  - `voxmlx/generate.py`: reduced cache-clear frequency (`256` -> `2048` tokens)
  - `voxmlx/audio.py`: cached STFT window + DFT basis (manual DFT path preserved)
- Created mono ground truth: `perf/ground_truth_mono.txt`.
- Stored run artifacts in `perf/audio_runs/<label>/`.
- Reran ground truth (`baseline-v2-clip180-create-gt-rerun`) on user request.
- Verified `perf/ground_truth_mono.txt` is unchanged by hash:
  - old/new SHA-256: `c884ae4d490915b3d5bd6cfc61d1cb37485d58df1b4cd92404ec2fbb95348e5f`

Now:
- Commit/push rerun artifacts and ledger update.

Next:
- Optional: run same evaluation with longer clips (e.g., 600s) or full files to confirm scaling.

Open questions (UNCONFIRMED if needed):
- UNCONFIRMED: target clip/window for sign-off beyond 180s (if user wants larger test window now).

Working set (files/ids/commands):
- `scripts/audio_eval.py`
- `voxmlx/audio.py`
- `voxmlx/generate.py`
- `perf/ground_truth_mono.txt`
- `perf/audio_runs/*/metrics.json`
- `perf/audio_runs/*/*_transcript.txt`
- `PYTHONPATH=. .venv313/bin/python scripts/audio_eval.py --label ...`
- Baseline commit: `f4d7d09`
- Optimized commit: `ea25661`

Performance results table:
| Label | Change summary | Commit | Model | Config | Audio | Time (s) | Speedup vs baseline | Deviation vs GT (norm edit / token err) | Notes |
|---|---|---|---|---|---|---:|---:|---:|---|
| env-setup-model-fetch | First dependency/model setup only | f4d7d09 | mlx-community/Voxtral-Mini-4B-Realtime-6bit | load-only | n/a | 2057.900 | n/a | n/a | One-time setup, not a transcription benchmark |
| baseline-v1-clip180-create-gt | Baseline; ground truth generated from mono | f4d7d09 | local HF snapshot | clip=180s,warmup=10s,temp=0.0 | mono+stereo | 66.900 | 0.00% | 0.020814 / 0.031863 | Ground truth saved to `perf/ground_truth_mono.txt` |
| opt-rfft-v1-clip180 | Experiment: FFT STFT path | f4d7d09 (dirty tree) | local HF snapshot | clip=180s,warmup=10s,temp=0.0 | mono+stereo | 65.814 | 1.62% | 0.040241 / 0.056373 | Speed gain small; deviation increased |
| opt-rfft-clearcache2048-clip180 | FFT + clear_cache(2048) | f4d7d09 (dirty tree) | local HF snapshot | clip=180s,warmup=10s,temp=0.0 | mono+stereo | 50.536 | 24.46% | 0.040241 / 0.056373 | Speed strong, deviation increased |
| opt-cachebasis-clearcache2048-clip180 | Revert to manual DFT; cache DFT basis + clear_cache(2048) | f4d7d09 (dirty tree) | local HF snapshot | clip=180s,warmup=10s,temp=0.0 | mono+stereo | 42.601 | 36.32% | 0.020814 / 0.031863 | Best experimental run; deviation back to baseline profile |
| final-ea25661-clip180 | Committed optimization run #1 | ea25661 | local HF snapshot | clip=180s,warmup=10s,temp=0.0 | mono+stereo | 51.083 | 23.64% | 0.020814 / 0.031863 | Meets speed target; baseline-equivalent deviation profile |
| final-ea25661-clip180-r2 | Committed optimization run #2 (repeat) | ea25661 | local HF snapshot | clip=180s,warmup=10s,temp=0.0 | mono+stereo | 57.770 | 13.65% | 0.020814 / 0.031863 | Repeat still above 10% target |
| baseline-v2-clip180-create-gt-rerun | Rerun GT generation on request (machine-load check) | 9a08382 | local HF snapshot | clip=180s,warmup=10s,temp=0.0 | mono+stereo | 45.870 | 31.43% | 0.020814 / 0.031863 | `ground_truth_mono.txt` hash unchanged; GT stable |
