#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from voxmlx import _build_prompt_tokens, load_model
from voxmlx.generate import generate


DEFAULT_MONO = Path("../vllm/voxtral_test_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k.wav")
DEFAULT_STEREO = Path("../vllm/voxtral_test_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_stereo_16k.wav")
DEFAULT_MODEL = Path.home() / ".cache/huggingface/hub/models--mlx-community--Voxtral-Mini-4B-Realtime-6bit/snapshots/02eb0caeb9dafb554c17a72b93dbf40cd3736c31"


@dataclass
class AudioRun:
    name: str
    source_path: str
    clip_seconds: int
    elapsed_seconds: float
    text_length_chars: int
    token_count: int
    norm_edit_distance: float | None
    token_error_ratio: float | None


def _write_clip(source: Path, out_dir: Path, seconds: int) -> Path:
    info = sf.info(str(source))
    max_frames = int(seconds * info.samplerate)
    n_frames = min(max_frames, info.frames)

    with sf.SoundFile(str(source), "r") as in_f:
        audio = in_f.read(frames=n_frames, dtype="float32")

    clip_path = out_dir / f"{source.stem}__{seconds}s.wav"
    sf.write(str(clip_path), audio, info.samplerate)
    return clip_path


def _levenshtein(items_a: list[str], items_b: list[str]) -> int:
    if len(items_a) < len(items_b):
        items_a, items_b = items_b, items_a
    if not items_b:
        return len(items_a)

    prev = list(range(len(items_b) + 1))
    for i, a in enumerate(items_a, start=1):
        curr = [i]
        for j, b in enumerate(items_b, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            subst_cost = prev[j - 1] + (a != b)
            curr.append(min(insert_cost, delete_cost, subst_cost))
        prev = curr
    return prev[-1]


def _char_norm_edit(pred: str, truth: str) -> float:
    denom = max(len(truth), 1)
    dist = _levenshtein(list(pred), list(truth))
    return dist / denom


def _token_error_ratio(pred: str, truth: str) -> float:
    pred_toks = pred.split()
    truth_toks = truth.split()
    denom = max(len(truth_toks), 1)
    dist = _levenshtein(pred_toks, truth_toks)
    return dist / denom


def _transcribe_once(model, sp, audio_path: Path, prompt_tokens: list[int], n_delay_tokens: int) -> tuple[str, int, float]:
    start = time.perf_counter()
    token_ids = generate(
        model,
        str(audio_path),
        prompt_tokens,
        n_delay_tokens=n_delay_tokens,
        temperature=0.0,
        eos_token_id=sp.eos_id,
    )
    elapsed = time.perf_counter() - start
    text = sp.decode(token_ids, special_token_policy=SpecialTokenPolicy.IGNORE)
    return text, len(token_ids), elapsed


def _save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible voxmlx audio eval runs")
    parser.add_argument("--label", required=True, help="Run label for metadata")
    parser.add_argument("--commit", required=True, help="Git commit hash for this run")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mono-path", type=Path, default=DEFAULT_MONO)
    parser.add_argument("--stereo-path", type=Path, default=DEFAULT_STEREO)
    parser.add_argument("--clip-seconds", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=Path("perf/audio_runs"))
    parser.add_argument("--ground-truth-path", type=Path, default=Path("perf/ground_truth_mono.txt"))
    parser.add_argument("--create-ground-truth", action="store_true")
    parser.add_argument("--warmup-seconds", type=int, default=10)
    parser.add_argument("--keep-clips", action="store_true")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"model path not found: {args.model_path}")
    if not args.mono_path.exists() or not args.stereo_path.exists():
        raise FileNotFoundError("audio fixture path missing")

    run_dir = args.output_dir / args.label
    run_dir.mkdir(parents=True, exist_ok=True)

    mono_clip = _write_clip(args.mono_path, run_dir, args.clip_seconds)
    stereo_clip = _write_clip(args.stereo_path, run_dir, args.clip_seconds)

    model, sp, _ = load_model(str(args.model_path))
    prompt_tokens, n_delay_tokens = _build_prompt_tokens(sp)

    # warmup to reduce first-run compilation effects
    warmup_clip = _write_clip(args.mono_path, run_dir, args.warmup_seconds)
    _transcribe_once(model, sp, warmup_clip, prompt_tokens, n_delay_tokens)

    mono_text, mono_tokens, mono_elapsed = _transcribe_once(model, sp, mono_clip, prompt_tokens, n_delay_tokens)
    _save_text(run_dir / "mono_transcript.txt", mono_text)

    if args.create_ground_truth:
        _save_text(args.ground_truth_path, mono_text)

    if not args.ground_truth_path.exists():
        raise FileNotFoundError(
            f"ground truth missing: {args.ground_truth_path}; run once with --create-ground-truth"
        )

    truth = args.ground_truth_path.read_text(encoding="utf-8")

    stereo_text, stereo_tokens, stereo_elapsed = _transcribe_once(model, sp, stereo_clip, prompt_tokens, n_delay_tokens)
    _save_text(run_dir / "stereo_transcript.txt", stereo_text)

    mono_run = AudioRun(
        name="mono",
        source_path=str(args.mono_path),
        clip_seconds=args.clip_seconds,
        elapsed_seconds=mono_elapsed,
        text_length_chars=len(mono_text),
        token_count=mono_tokens,
        norm_edit_distance=_char_norm_edit(mono_text, truth),
        token_error_ratio=_token_error_ratio(mono_text, truth),
    )
    stereo_run = AudioRun(
        name="stereo",
        source_path=str(args.stereo_path),
        clip_seconds=args.clip_seconds,
        elapsed_seconds=stereo_elapsed,
        text_length_chars=len(stereo_text),
        token_count=stereo_tokens,
        norm_edit_distance=_char_norm_edit(stereo_text, truth),
        token_error_ratio=_token_error_ratio(stereo_text, truth),
    )

    payload = {
        "label": args.label,
        "commit": args.commit,
        "model_path": str(args.model_path),
        "clip_seconds": args.clip_seconds,
        "created_ground_truth": bool(args.create_ground_truth),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [asdict(mono_run), asdict(stereo_run)],
        "aggregate": {
            "total_elapsed_seconds": mono_elapsed + stereo_elapsed,
            "mean_norm_edit_distance": (mono_run.norm_edit_distance + stereo_run.norm_edit_distance) / 2,
            "mean_token_error_ratio": (mono_run.token_error_ratio + stereo_run.token_error_ratio) / 2,
        },
    }

    out_json = run_dir / "metrics.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if not args.keep_clips:
        for clip_path in (mono_clip, stereo_clip, warmup_clip):
            if clip_path.exists():
                clip_path.unlink()

    print(f"[audio-eval] wrote {out_json}")
    for run in (mono_run, stereo_run):
        print(
            "[audio-eval]"
            f" {run.name}: elapsed={run.elapsed_seconds:.3f}s"
            f" norm_edit={run.norm_edit_distance:.6f}"
            f" token_err={run.token_error_ratio:.6f}"
            f" tokens={run.token_count}"
        )
    print(f"[audio-eval] total_elapsed_seconds={payload['aggregate']['total_elapsed_seconds']:.3f}")


if __name__ == "__main__":
    main()
