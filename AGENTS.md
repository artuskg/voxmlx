# AGENTS.md — voxmlx

## Shared policy

Follow the managed global policy and relevant skills. This file owns voxmlx-specific guidance; a repo task does not authorize changing global configuration or another repository.

## Continuity Ledger (compaction-safe)

Create a continuity ledger only when requested, using `continuity_ledger`. Resume an existing ledger only when its goal matches the current task; bounded maintenance does not require a ledger offer.

### How it works (if the ledger is active for the current session)

- At the start of every assistant turn: read `CONTINUITY_CODEX-<sessionid>.md`, update it to reflect the latest goal/constraints/decisions/state, then proceed with the work.
- Update `CONTINUITY_CODEX-<sessionid>.md` whenever goal, constraints/assumptions, decisions, progress state (Done/Now/Next), or major tool outcomes change.
- Keep it short and factual; mark uncertainty as `UNCONFIRMED`.

### `CONTINUITY_CODEX-<sessionid>.md` format

- Goal (incl. success criteria):
- Constraints/Assumptions:
- Key decisions:
- State:
- Done:
- Now:
- Next:
- Open questions (UNCONFIRMED if needed):
- Working set (files/ids/commands):

## Execution reliability baseline

- Prefer `python3` (or explicit venv interpreter); do not assume `python` exists.
- Assume macOS/BSD userland by default; avoid GNU-only flags unless available.
- Treat non-zero exits from process-probe commands (`ps -p`, `pgrep`, `lsof` no match) as expected checks, not hard failures.
- Before claiming a background run started, verify expected artifacts exist.
- For long runs (>10 minutes), use resilient orchestration (`tmux`, launchd, or equivalent) and monitor progress freshness.

## Git hygiene baseline

- Use the managed `commit` skill for synchronization, commits, and publication. Publish verified agent-authored changes by default only to destinations it authorizes; preserve unrelated dirty work.
- A pull-only request does not authorize publication. Resolve an unclear destination before pushing; do not invent a branch or remote.

## Local project docs recommendation

- Maintain this policy file (`AGENTS.md`) and an operational file (`RUNBOOK.md`).
- Keep `RUNBOOK.md` procedural and command-first; keep this file policy/rules-first.
- Read `RUNBOOK.md` before executing long-running or high-cost operations.

## Project-specific additions (voxmlx)

- Keep default CI lightweight: correctness tests and pure-function perf checks must run without downloading large models.
- Model-backed differential tests are optional and must be gated behind environment variables.
- For audio performance experiments, run versions sequentially (never in parallel) and prefer repeated runs via `scripts/run_version_matrix.py` + `perf/version_matrix.json` on a dedicated machine.
- When changing correctness/performance workflow, update both:
  - `docs/correctness_performance.md`
  - `RUNBOOK.md`

## Confusion log

- 2026-02-09: Initial local policy file created for voxmlx to make project workflow explicit and compaction-safe.
- 2026-02-09: Creating a new branch and committing succeeded, but push to `origin` failed with `403 (Permission to artuskg/voxmlx.git denied to Crabbotix)`; SSH fallback failed with host-key verification. Resolution: keep local branch/commits ready and report exact blocked push command for user-side credential/remote fix.
- 2026-02-09: `pip install -e` failed in scaffold snapshot `bb7f2c1` after adding committed audio under `perf/reference_audio`, because setuptools auto-discovered multiple top-level packages (`perf`, `voxmlx`). Resolution: for benchmark execution, skip editable install and install runtime dependencies directly (`pip install mlx numpy soundfile sounddevice huggingface-hub mistral-common sentencepiece`) while using `PYTHONPATH=.` for local imports.
- On a policy denial, inspect and report the stated reason. Use another implementation only for a syntax/tool failure when the underlying operation remains authorized; never translate an operation-level denial into another tool.
- 2026-02-09: User-aborted matrix runs can leave child `audio_eval.py` processes alive and writing partial artifacts. Resolution: always probe/stop lingering benchmark PIDs (`pgrep -fl 'run_version_matrix.py|audio_eval.py'`) before starting a new campaign and treat partial campaign folders as non-final.
- 2026-02-09: Apparent long-audio transcription quality was misleading because the pipeline compared against model-derived ground truth. The model generated extensive `[STREAMING_PAD]` tokens (visible with `SpecialTokenPolicy.KEEP`) while `IGNORE` decoding made transcripts look short but stable. Resolution: do not treat model-derived GT as quality truth for long-form audio; inspect raw token composition and compare against human/reference transcripts before drawing correctness conclusions.
- 2026-02-09: Attempted to run `../voxtral.c` against the MLX-converted local snapshot and hit tensor-name/file mismatch errors (expects Mistral layout like `consolidated.safetensors`). Confusing because both are Voxtral-family weights but not format-compatible. Resolution: use `voxtral.c/download_model.sh` to fetch native weights and store them under `/Volumes/BigStore/voxtral-model`; treat MLX and `voxtral.c` model directories as separate artifacts.
- 2026-02-09: Incoming-matrix audio paths that pointed into a temporary scaffold worktree (`.tmp_incoming_scaffold/perf/reference_audio/...`) were missing later in session. Confusing because earlier runs succeeded with those paths. Resolution: prefer stable absolute fixture paths (`/tmp/voxtral_ref_*_10min.wav` or committed `perf/reference_audio/*`) when creating reusable benchmark matrices.
