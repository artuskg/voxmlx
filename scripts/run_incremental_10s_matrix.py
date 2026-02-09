#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf


@dataclass(frozen=True)
class VariantSpec:
    id: str
    type: str
    description: str
    ref: str | None = None
    enabled: bool = True
    bin_path: str | None = None
    model_dir: str | None = None
    extra_args: list[str] | None = None


def _run(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    print("[inc-matrix]", " ".join(cmd))
    if dry_run:
        return None
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


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
        "clip_seconds",
        "repeats",
        "warmup_repeats",
        "audio_path",
        "model_path",
        "variants",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"matrix missing required keys: {missing}")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant_id",
        "variant_type",
        "description",
        "commit_or_ref",
        "mean_elapsed_total_seconds",
        "std_elapsed_total_seconds",
        "mean_elapsed_encode_seconds",
        "mean_elapsed_decode_seconds",
        "speedup_vs_baseline_percent",
        "token_count",
        "text_char_count",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _ensure_clip(src: Path, clip_seconds: int, out_path: Path) -> None:
    info = sf.info(str(src))
    n_frames = min(int(info.samplerate * clip_seconds), info.frames)
    with sf.SoundFile(str(src), "r") as f:
        audio = f.read(frames=n_frames, dtype="float32")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, info.samplerate, subtype="PCM_16")


def _parse_incremental_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_voxmlx_variant(
    repo_root: Path,
    variant: VariantSpec,
    worktree_root: Path,
    python_exe: Path,
    eval_script: Path,
    model_path: Path,
    clip_path: Path,
    clip_seconds: int,
    repeats: int,
    warmup_repeats: int,
    output_dir: Path,
    keep_worktrees: bool,
    stft_backend: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    if not variant.ref:
        raise ValueError(f"Variant {variant.id} requires 'ref'")

    worktree_path = worktree_root / variant.id
    if worktree_path.exists():
        shutil.rmtree(worktree_path)

    _run(
        ["git", "worktree", "add", "--detach", str(worktree_path), variant.ref],
        cwd=repo_root,
        dry_run=dry_run,
    )

    try:
        commit = variant.ref
        if not dry_run:
            commit = _git_output(worktree_path, ["rev-parse", "--short", "HEAD"])

        label = f"{output_dir.name}__{variant.id}"
        summary_path = output_dir / label / "summary.json"

        env = os.environ.copy()
        env["PYTHONPATH"] = str(worktree_path)
        if stft_backend:
            env["VOXMLX_STFT_BACKEND"] = stft_backend

        cmd = [
            str(python_exe),
            str(eval_script),
            "--label",
            label,
            "--commit",
            commit,
            "--model-path",
            str(model_path),
            "--audio-path",
            str(clip_path),
            "--clip-seconds",
            str(clip_seconds),
            "--repeats",
            str(repeats),
            "--warmup-repeats",
            str(warmup_repeats),
            "--output-dir",
            str(output_dir),
        ]
        proc = _run(cmd, cwd=worktree_path, env=env, dry_run=dry_run)
        if proc is not None and proc.stdout:
            print(proc.stdout.strip())
        if proc is not None and proc.stderr:
            print(proc.stderr.strip())

        if dry_run:
            return {
                "variant_id": variant.id,
                "variant_type": variant.type,
                "description": variant.description,
                "commit_or_ref": commit,
                "summary_path": str(summary_path),
            }

        summary = _parse_incremental_summary(summary_path)
        return {
            "variant_id": variant.id,
            "variant_type": variant.type,
            "description": variant.description,
            "commit_or_ref": commit,
            "mean_elapsed_total_seconds": summary["mean_elapsed_total_seconds"],
            "std_elapsed_total_seconds": summary["std_elapsed_total_seconds"],
            "mean_elapsed_encode_seconds": summary["mean_elapsed_encode_seconds"],
            "mean_elapsed_decode_seconds": summary["mean_elapsed_decode_seconds"],
            "token_count": summary["token_count"],
            "text_char_count": summary["text_char_count"],
            "summary_path": str(summary_path),
            "output_path": summary["output_path"],
        }
    finally:
        if not keep_worktrees:
            _run(
                ["git", "worktree", "remove", "--force", str(worktree_path)],
                cwd=repo_root,
                dry_run=dry_run,
            )


def _run_voxtralc_variant(
    repo_root: Path,
    variant: VariantSpec,
    clip_path: Path,
    repeats: int,
    output_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    if not variant.bin_path or not variant.model_dir:
        raise ValueError(f"Variant {variant.id} requires bin_path and model_dir")

    label = f"{output_dir.name}__{variant.id}"
    run_dir = output_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(
            "[inc-matrix] dry-run voxtral.c",
            variant.bin_path,
            "-d",
            variant.model_dir,
            "-i",
            str(clip_path),
            *(variant.extra_args or []),
        )
        return {
            "variant_id": variant.id,
            "variant_type": variant.type,
            "description": variant.description,
            "commit_or_ref": "voxtral.c",
            "summary_path": str(run_dir / "summary.json"),
        }

    cmd = [variant.bin_path, "-d", variant.model_dir, "-i", str(clip_path), *(variant.extra_args or [])]
    elapsed_values: list[float] = []
    transcript = ""
    runs_payload: list[dict[str, Any]] = []
    for i in range(1, repeats + 1):
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            check=True,
        )
        elapsed = time.perf_counter() - t0
        elapsed_values.append(elapsed)
        transcript = proc.stdout
        runs_payload.append(
            {
                "repeat_index": i,
                "elapsed_total_seconds": elapsed,
                "stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-8:]),
            }
        )

    mean_elapsed = float(sum(elapsed_values) / len(elapsed_values))
    if len(elapsed_values) > 1:
        var = sum((x - mean_elapsed) ** 2 for x in elapsed_values) / len(elapsed_values)
        std_elapsed = float(var**0.5)
    else:
        std_elapsed = 0.0

    transcript_path = run_dir / "mono_incremental_transcript.txt"
    transcript_path.write_text(transcript, encoding="utf-8")
    summary = {
        "label": label,
        "method": "voxtral.c_cli_file",
        "repeats": repeats,
        "clip_path": str(clip_path),
        "bin_path": str(variant.bin_path),
        "model_dir": str(variant.model_dir),
        "mean_elapsed_total_seconds": mean_elapsed,
        "std_elapsed_total_seconds": std_elapsed,
        "token_count": None,
        "text_char_count": len(transcript),
        "output_path": str(transcript_path),
        "runs": runs_payload,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "variant_id": variant.id,
        "variant_type": variant.type,
        "description": variant.description,
        "commit_or_ref": "voxtral.c",
        "mean_elapsed_total_seconds": mean_elapsed,
        "std_elapsed_total_seconds": std_elapsed,
        "mean_elapsed_encode_seconds": None,
        "mean_elapsed_decode_seconds": None,
        "token_count": None,
        "text_char_count": len(transcript),
        "summary_path": str(summary_path),
        "output_path": str(transcript_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential incremental-only 10s benchmark matrix runner")
    parser.add_argument("--matrix", type=Path, default=Path("perf/incremental_10s_matrix.json"))
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup-repeats", type=int, default=None)
    parser.add_argument("--python", dest="python_exe", type=Path, default=Path(".venv313/bin/python"))
    parser.add_argument("--eval-script", type=Path, default=Path("scripts/incremental_file_eval.py"))
    parser.add_argument("--keep-worktrees", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stft-backend", choices=["dft", "fft"], default=None)
    parser.add_argument(
        "--variant-ids",
        default=None,
        help="Comma-separated subset of variant IDs from matrix",
    )
    args = parser.parse_args()

    repo_root = Path(_git_output(Path.cwd(), ["rev-parse", "--show-toplevel"]))
    matrix_path = _resolve_path(str(args.matrix), repo_root)
    matrix = _load_matrix(matrix_path)

    repeats = int(args.repeats if args.repeats is not None else matrix["repeats"])
    warmup_repeats = int(
        args.warmup_repeats if args.warmup_repeats is not None else matrix["warmup_repeats"]
    )
    clip_seconds = int(matrix["clip_seconds"])
    if clip_seconds <= 0:
        raise ValueError("clip_seconds must be > 0")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign = args.campaign or f"{matrix['matrix_name']}__{timestamp}"

    output_dir = repo_root / "perf" / "audio_runs"
    batch_dir = repo_root / "perf" / "batch_runs" / campaign
    batch_dir.mkdir(parents=True, exist_ok=True)

    clip_source = _resolve_path(matrix["audio_path"], repo_root)
    clip_path = batch_dir / f"{clip_source.stem}__{clip_seconds}s.wav"
    _ensure_clip(clip_source, clip_seconds, clip_path)

    python_exe = _resolve_path(str(args.python_exe), repo_root)
    eval_script = _resolve_path(str(args.eval_script), repo_root)
    model_path = _resolve_path(matrix["model_path"], repo_root)

    all_variants = [
        VariantSpec(
            id=v["id"],
            type=v["type"],
            description=v.get("description", ""),
            ref=v.get("ref"),
            enabled=bool(v.get("enabled", True)),
            bin_path=v.get("bin_path"),
            model_dir=v.get("model_dir"),
            extra_args=v.get("extra_args"),
        )
        for v in matrix["variants"]
    ]
    variants = [v for v in all_variants if v.enabled]

    if args.variant_ids:
        wanted = {x.strip() for x in args.variant_ids.split(",") if x.strip()}
        unknown = sorted(wanted - {v.id for v in variants})
        if unknown:
            raise ValueError(f"Unknown --variant-ids values: {unknown}")
        variants = [v for v in variants if v.id in wanted]

    baseline_id = matrix["baseline_version_id"]
    rows: list[dict[str, Any]] = []
    worktree_root = repo_root / ".tmp_incremental_matrix" / campaign
    worktree_root.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        print(f"[inc-matrix] running {variant.id} ({variant.type})")
        if variant.type == "voxmlx":
            row = _run_voxmlx_variant(
                repo_root=repo_root,
                variant=variant,
                worktree_root=worktree_root,
                python_exe=python_exe,
                eval_script=eval_script,
                model_path=model_path,
                clip_path=clip_path,
                clip_seconds=clip_seconds,
                repeats=repeats,
                warmup_repeats=warmup_repeats,
                output_dir=output_dir,
                keep_worktrees=args.keep_worktrees,
                stft_backend=args.stft_backend,
                dry_run=args.dry_run,
            )
        elif variant.type == "voxtralc":
            row = _run_voxtralc_variant(
                repo_root=repo_root,
                variant=variant,
                clip_path=clip_path,
                repeats=repeats,
                output_dir=output_dir,
                dry_run=args.dry_run,
            )
        else:
            raise ValueError(f"Unsupported variant type {variant.type!r} for {variant.id}")
        rows.append(row)

    baseline_row = next((r for r in rows if r["variant_id"] == baseline_id), None)
    baseline_mean = None if baseline_row is None else baseline_row.get("mean_elapsed_total_seconds")
    for row in rows:
        if baseline_mean is None or row.get("mean_elapsed_total_seconds") is None:
            row["speedup_vs_baseline_percent"] = None
            continue
        row["speedup_vs_baseline_percent"] = (
            (baseline_mean - row["mean_elapsed_total_seconds"]) / baseline_mean
        ) * 100.0

    summary = {
        "campaign": campaign,
        "matrix_path": str(matrix_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "clip_source": str(clip_source),
        "clip_path": str(clip_path),
        "clip_seconds": clip_seconds,
        "repeats": repeats,
        "warmup_repeats": warmup_repeats,
        "baseline_version_id": baseline_id,
        "rows": rows,
    }
    _write_json(batch_dir / "summary.json", summary)
    _write_csv(batch_dir / "summary.csv", rows)
    print(f"[inc-matrix] wrote {batch_dir / 'summary.json'}")
    print(f"[inc-matrix] wrote {batch_dir / 'summary.csv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
