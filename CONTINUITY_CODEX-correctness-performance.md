Goal (incl. success criteria):
- Add a maintainable scaffold to verify correctness now and protect correctness while optimizing performance later.
- Success criteria: test harness, benchmark harness, and CI gates are added and runnable on a lightweight path.

Constraints/Assumptions:
- User requested a new branch for this work.
- Keep changes compatible with existing Python package structure.
- Heavy MLX/model-download paths should not be required for default CI checks.

Key decisions:
- Use branch `codex/correctness-performance-scaffold`.
- Introduce a pure helper contract module (`voxmlx/contracts.py`) and test that surface deterministically.
- Make performance checks warn-only initially with an explicit baseline JSON and threshold checker.

State:
- Done: Branch + ledger created; scaffolding implemented; local tests/bench checks executed; commit created.
- Now: Update remote to user fork, push branch, add optional model-backed differential tests, and create project-local `AGENTS.md` + `RUNBOOK.md`.
- Next: Validate tests/docs, commit, and push follow-up changes.

Done:
- Added `voxmlx/contracts.py` with remap/format/sharding/prompt-token helpers.
- Wired `voxmlx/__init__.py`, `voxmlx/weights.py`, and `voxmlx/convert.py` to shared contracts.
- Added unit tests: `tests/test_contracts.py` and fixtures.
- Added perf scripts: `scripts/bench_contracts.py`, `scripts/check_perf_regression.py`.
- Added CI workflow: `.github/workflows/correctness-performance.yml`.
- Added docs: `docs/correctness_performance.md` and README pointer.
- Calibrated benchmark baseline: `perf/baseline_contracts.json`.
- Committed changes: `b5e8a09`.
- Added optional model-backed differential tests in `tests/test_model_differential.py` (env-gated).
- Added project-local `AGENTS.md` and `RUNBOOK.md`.
- Updated docs for optional model-backed lane and runbook pointer.

Now:
- Validate updated test suite (including default skip behavior for model tests).
- Repoint `origin` to `https://github.com/artuskg/voxmlx.git`, commit changes, and push branch.

Next:
- If requested: add model-backed differential tests guarded behind optional env flags.

Open questions (UNCONFIRMED if needed):
- UNCONFIRMED: desired strictness/date for switching perf check from warn-only to failing gate.

Working set (files/ids/commands):
- `voxmlx/contracts.py`
- `voxmlx/__init__.py`
- `voxmlx/weights.py`
- `voxmlx/convert.py`
- `tests/test_contracts.py`
- `tests/fixtures/weight_remap_cases.json`
- `scripts/bench_contracts.py`
- `scripts/check_perf_regression.py`
- `.github/workflows/correctness-performance.yml`
- `docs/correctness_performance.md`
- `perf/baseline_contracts.json`
- `CONTINUITY_CODEX-correctness-performance.md`
- Commit: `b5e8a09`
- `git remote set-url origin https://github.com/artuskg/voxmlx.git`
- `tests/test_model_differential.py`
- `AGENTS.md`
- `RUNBOOK.md`
