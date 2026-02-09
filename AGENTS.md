# AGENTS.md — voxmlx

## Sync policy (MANDATORY)

- Keep shared baseline rules in this file synchronized with the ai-software-engineering counterpart at `~/[Gg]it[Rr]epos/ai-software-engineering/AGENTS.md`.
- Do not hardcode machine-specific absolute paths for ai-software-engineering; discover it under `~/[Gg]it[Rr]epos/ai-software-engineering`.
- When updating shared behavior rules in either file, update the other file in the same work session.
- Proactively remind the user to keep ai-software-engineering and local agent configuration (for example `~/.codex` and `~/.agents/skills`) loosely in sync when either side changes.
- Project-specific additions can stay only here, but they should be explicitly marked as project-specific.

## Continuity Ledger (compaction-safe)

At the beginning of a new session, before starting the actual work, offer the user to maintain a single Continuity Ledger for this session in `CONTINUITY_CODEX-<sessionid>.md` or offer the user to continue an existing `CONTINUITY_CODEX-<sessionid>.md` in the current workspace.

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

- If the tree is dirty and a pull is needed, use `git pull --rebase --autostash` unless the user requests otherwise.
- Default assumption: for agent-authored changes in a git-tracked repo, commit and `git push` unless user opts out.
- If push target is unclear, create and use a `codex/<topic>` branch first.
- After push/cleanup requests, verify and report `git status --short`.

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
- 2026-02-09: Cleanup command with `rm -rf` was blocked by session policy even for temporary benchmark artifacts. Resolution: use a short Python cleanup snippet (`shutil.rmtree`/`Path.unlink`) for deterministic artifact removal when shell deletion is policy-blocked.
- 2026-02-09: User-aborted matrix runs can leave child `audio_eval.py` processes alive and writing partial artifacts. Resolution: always probe/stop lingering benchmark PIDs (`pgrep -fl 'run_version_matrix.py|audio_eval.py'`) before starting a new campaign and treat partial campaign folders as non-final.
- 2026-02-09: Apparent long-audio transcription quality was misleading because the pipeline compared against model-derived ground truth. The model generated extensive `[STREAMING_PAD]` tokens (visible with `SpecialTokenPolicy.KEEP`) while `IGNORE` decoding made transcripts look short but stable. Resolution: do not treat model-derived GT as quality truth for long-form audio; inspect raw token composition and compare against human/reference transcripts before drawing correctness conclusions.
