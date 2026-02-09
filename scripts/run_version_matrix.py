#!/usr/bin/env python3
"""Sequential multi-version audio benchmark runner.

This script runs each configured version multiple times (default: 10),
collects metrics from scripts/audio_eval.py, and writes aggregated summaries.

Design goals:
- Strictly sequential execution (no parallel runs)
- Version clarity via an explicit matrix file
- Reproducibility via commit refs + persisted summary artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VersionSpec:
    id: str
    ref: str
    description: str


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None, dry_run: bool = False) -> str:
    print("[matrix]", " ".join(cmd))
    if dry_run:
        return ""

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip())
    return proc.stdout


def _git_output(repo_root: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def _resolve_path(value: str, repo_root: Path) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _load_matrix(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "matrix_name",
        "baseline_version_id",
        "repeats",
        "clip_seconds",
        "warmup_seconds",
        "model_path",
        "mono_path",
        "stereo_path",
        "ground_truth_path",
        "versions",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"matrix missing required keys: {missing}")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _compute_summary(rows: list[dict[str, Any]], baseline_id: str) -> dict[str, Any]:
    by_version: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_version.setdefault(row["version_id"], []).append(row)

    summary_versions: list[dict[str, Any]] = []
    baseline_mean = None
    if baseline_id in by_version:
        baseline_mean = statistics.mean(r["total_elapsed_seconds"] for r in by_version[baseline_id])

    for version_id, items in by_version.items():
        times = [r["total_elapsed_seconds"] for r in items]
        norm_edits = [r["mean_norm_edit_distance"] for r in items]
        token_errs = [r["mean_token_error_ratio"] for r in items]

        mean_time = statistics.mean(times)
        stdev_time = statistics.pstdev(times) if len(times) > 1 else 0.0
        mean_norm_edit = statistics.mean(norm_edits)
        mean_token_err = statistics.mean(token_errs)

        speedup_pct = None
        if baseline_mean and baseline_mean > 0:
            speedup_pct = ((baseline_mean - mean_time) / baseline_mean) * 100.0

        summary_versions.append(
            {
                "version_id": version_id,
                "commit": items[0]["commit"],
                "description": items[0]["description"],
                "runs": len(items),
                "mean_total_elapsed_seconds": mean_time,
                "stdev_total_elapsed_seconds": stdev_time,
                "min_total_elapsed_seconds": min(times),
                "max_total_elapsed_seconds": max(times),
                "speedup_vs_baseline_percent": speedup_pct,
                "mean_norm_edit_distance": mean_norm_edit,
                "mean_token_error_ratio": mean_token_err,
            }
        )

    summary_versions.sort(key=lambda v: v["version_id"])

    return {
        "baseline_version_id": baseline_id,
        "baseline_mean_total_elapsed_seconds": baseline_mean,
        "versions": summary_versions,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "version_id",
        "description",
        "commit",
        "run_index",
        "label",
        "total_elapsed_seconds",
        "mean_norm_edit_distance",
        "mean_token_error_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sequential multi-version audio benchmarks")
    parser.add_argument("--matrix", type=Path, default=Path("perf/version_matrix.json"))
    parser.add_argument("--campaign", default=None, help="Campaign label (default: matrix_name + UTC timestamp)")
    parser.add_argument("--repeats", type=int, default=None, help="Override repeats from matrix")
    parser.add_argument("--python", dest="python_exe", type=Path, default=Path(".venv313/bin/python"))
    parser.add_argument("--audio-eval-script", type=Path, default=Path("scripts/audio_eval.py"))
    parser.add_argument("--keep-worktrees", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instrument", action="store_true", help="Pass instrumentation flags to audio_eval")
    parser.add_argument("--instrument-sample-seconds", type=float, default=1.0)
    args = parser.parse_args()

    repo_root = Path(_git_output(Path.cwd(), ["rev-parse", "--show-toplevel"]))
    matrix_path = _resolve_path(str(args.matrix), repo_root)
    matrix = _load_matrix(matrix_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign = args.campaign or f"{matrix['matrix_name']}__{timestamp}"

    repeats = args.repeats if args.repeats is not None else int(matrix["repeats"])
    baseline_version_id = matrix["baseline_version_id"]

    versions = [
        VersionSpec(
            id=v["id"],
            ref=v["ref"],
            description=v.get("description", ""),
        )
        for v in matrix["versions"]
    ]

    if baseline_version_id not in {v.id for v in versions}:
        raise ValueError(f"baseline_version_id {baseline_version_id!r} not found in versions")

    python_exe = _resolve_path(str(args.python_exe), repo_root)
    audio_eval_script = _resolve_path(str(args.audio_eval_script), repo_root)

    model_path = _resolve_path(matrix["model_path"], repo_root)
    mono_path = _resolve_path(matrix["mono_path"], repo_root)
    stereo_path = _resolve_path(matrix["stereo_path"], repo_root)
    ground_truth_path = _resolve_path(matrix["ground_truth_path"], repo_root)

    clip_seconds = int(matrix["clip_seconds"])
    warmup_seconds = int(matrix["warmup_seconds"])

    summary_dir = repo_root / "perf" / "batch_runs" / campaign
    summary_dir.mkdir(parents=True, exist_ok=True)

    worktree_root = repo_root / ".perf_worktrees" / campaign
    worktree_root.mkdir(parents=True, exist_ok=True)

    print(f"[matrix] campaign={campaign}")
    print(f"[matrix] repeats={repeats} clip_seconds={clip_seconds} warmup_seconds={warmup_seconds}")
    print("[matrix] versions:")
    for version in versions:
        print(f"  - {version.id}: ref={version.ref} desc={version.description}")

    run_rows: list[dict[str, Any]] = []
    created_worktrees: list[Path] = []
    instrument_args: list[str] = []
    if args.instrument:
        instrument_args = [
            "--instrument",
            "--instrument-sample-seconds",
            str(args.instrument_sample_seconds),
        ]

    try:
        # 1) prepare worktrees sequentially
        for version in versions:
            wt = worktree_root / version.id
            if wt.exists() and not args.dry_run:
                _run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo_root, dry_run=False)
            _run(["git", "worktree", "add", "--detach", str(wt), version.ref], cwd=repo_root, dry_run=args.dry_run)
            created_worktrees.append(wt)

        # 2) refresh ground truth from baseline version once per campaign
        baseline = next(v for v in versions if v.id == baseline_version_id)
        baseline_wt = worktree_root / baseline.id
        baseline_commit = baseline.ref
        if not args.dry_run:
            baseline_commit = _git_output(baseline_wt, ["rev-parse", "--short", "HEAD"])

        gt_label = f"{campaign}__groundtruth__{baseline.id}"
        gt_env = dict(os.environ)
        gt_env["PYTHONPATH"] = str(baseline_wt)
        _run(
            [
                str(python_exe),
                str(audio_eval_script),
                "--label",
                gt_label,
                "--commit",
                baseline_commit,
                "--model-path",
                str(model_path),
                "--mono-path",
                str(mono_path),
                "--stereo-path",
                str(stereo_path),
                "--clip-seconds",
                str(clip_seconds),
                "--warmup-seconds",
                str(warmup_seconds),
                "--ground-truth-path",
                str(ground_truth_path),
                "--create-ground-truth",
                *instrument_args,
            ],
            cwd=repo_root,
            env=gt_env,
            dry_run=args.dry_run,
        )

        # 3) run versions sequentially, repeat N times each
        for version in versions:
            wt = worktree_root / version.id
            commit = version.ref
            if not args.dry_run:
                commit = _git_output(wt, ["rev-parse", "--short", "HEAD"])

            env = dict(os.environ)
            env["PYTHONPATH"] = str(wt)

            for idx in range(1, repeats + 1):
                label = f"{campaign}__{version.id}__r{idx:02d}"
                _run(
                    [
                        str(python_exe),
                        str(audio_eval_script),
                        "--label",
                        label,
                        "--commit",
                        commit,
                        "--model-path",
                        str(model_path),
                        "--mono-path",
                        str(mono_path),
                        "--stereo-path",
                        str(stereo_path),
                        "--clip-seconds",
                        str(clip_seconds),
                        "--warmup-seconds",
                        str(warmup_seconds),
                        "--ground-truth-path",
                        str(ground_truth_path),
                        *instrument_args,
                    ],
                    cwd=repo_root,
                    env=env,
                    dry_run=args.dry_run,
                )

                if args.dry_run:
                    continue

                metrics_path = repo_root / "perf" / "audio_runs" / label / "metrics.json"
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                run_rows.append(
                    {
                        "version_id": version.id,
                        "description": version.description,
                        "commit": commit,
                        "run_index": idx,
                        "label": label,
                        "total_elapsed_seconds": metrics["aggregate"]["total_elapsed_seconds"],
                        "mean_norm_edit_distance": metrics["aggregate"]["mean_norm_edit_distance"],
                        "mean_token_error_ratio": metrics["aggregate"]["mean_token_error_ratio"],
                    }
                )

        if not args.dry_run:
            summary = {
                "campaign": campaign,
                "matrix_path": str(matrix_path),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "repeats": repeats,
                "clip_seconds": clip_seconds,
                "warmup_seconds": warmup_seconds,
                "model_path": str(model_path),
                "mono_path": str(mono_path),
                "stereo_path": str(stereo_path),
                "ground_truth_path": str(ground_truth_path),
                "run_rows": run_rows,
                "summary": _compute_summary(run_rows, baseline_version_id),
            }

            _write_json(summary_dir / "summary.json", summary)
            _write_csv(summary_dir / "runs.csv", run_rows)
            print(f"[matrix] wrote {summary_dir / 'summary.json'}")
            print(f"[matrix] wrote {summary_dir / 'runs.csv'}")

    finally:
        if not args.keep_worktrees:
            for wt in created_worktrees:
                if wt.exists() and not args.dry_run:
                    _run(["git", "worktree", "remove", "--force", str(wt)], cwd=repo_root, dry_run=False)
            if worktree_root.exists() and not any(worktree_root.iterdir()) and not args.dry_run:
                worktree_root.rmdir()

    return 0


if __name__ == "__main__":
    sys.exit(main())
