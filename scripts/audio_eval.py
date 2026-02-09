#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
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


class InstrumentSampler:
    def __init__(self, sample_seconds: float) -> None:
        self.sample_seconds = sample_seconds
        self.samples: list[dict[str, float | int | str | bool | None]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._phase = "init"
        self._phase_lock = threading.Lock()
        self._last_cpu_time: float | None = None
        self._last_wall_time: float | None = None

    def set_phase(self, phase: str) -> None:
        with self._phase_lock:
            self._phase = phase

    def _get_phase(self) -> str:
        with self._phase_lock:
            return self._phase

    @staticmethod
    def _read_process_rss_bytes(pid: int) -> int | None:
        try:
            proc = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(pid)],
                text=True,
                capture_output=True,
                check=True,
            )
            rss_kb = int(proc.stdout.strip() or "0")
            return rss_kb * 1024
        except Exception:
            return None

    def _sample_once(self) -> None:
        now_wall = time.perf_counter()
        proc_times = os.times()
        proc_cpu = proc_times.user + proc_times.system
        cpu_percent = None
        if self._last_wall_time is not None and self._last_cpu_time is not None:
            wall_delta = max(now_wall - self._last_wall_time, 1e-9)
            cpu_delta = max(proc_cpu - self._last_cpu_time, 0.0)
            cpu_percent = (cpu_delta / wall_delta) * 100.0
        self._last_wall_time = now_wall
        self._last_cpu_time = proc_cpu

        load1, load5, load15 = os.getloadavg()
        rss_bytes = self._read_process_rss_bytes(os.getpid())

        mlx_active, mlx_peak = _mlx_memory_stats()

        self.samples.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "phase": self._get_phase(),
                "process_cpu_percent": cpu_percent,
                "process_rss_bytes": rss_bytes,
                "load1": load1,
                "load5": load5,
                "load15": load15,
                "mlx_active_memory_bytes": mlx_active,
                "mlx_peak_memory_bytes": mlx_peak,
            }
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self.sample_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._sample_once()


def _safe_stat(values: list[float], fn, default: float | None = None) -> float | None:
    if not values:
        return default
    return float(fn(values))


def _mlx_memory_stats() -> tuple[int | None, int | None]:
    try:
        import mlx.core as mx

        if hasattr(mx, "get_active_memory"):
            active = int(mx.get_active_memory())
        else:
            active = int(mx.metal.get_active_memory())

        if hasattr(mx, "get_peak_memory"):
            peak = int(mx.get_peak_memory())
        else:
            peak = int(mx.metal.get_peak_memory())
        return active, peak
    except Exception:
        return None, None


def _mlx_device_details() -> dict[str, str | bool | None]:
    try:
        import mlx.core as mx

        if hasattr(mx, "device_info"):
            info = mx.device_info()
        else:
            info = mx.metal.device_info()
        return {
            "default_device": str(mx.default_device()),
            "metal_available": bool(mx.metal.is_available()),
            "metal_device_info": str(info),
            "mx_disable_metal": os.environ.get("MX_DISABLE_METAL"),
        }
    except Exception:
        return {
            "default_device": None,
            "metal_available": None,
            "metal_device_info": None,
            "mx_disable_metal": os.environ.get("MX_DISABLE_METAL"),
        }


def _instrumentation_summary(
    sampler: InstrumentSampler,
    sample_seconds: float,
    device_info: dict[str, str | bool | None],
) -> dict[str, object]:
    samples = sampler.samples
    cpu_values = [float(s["process_cpu_percent"]) for s in samples if s["process_cpu_percent"] is not None]
    rss_values = [int(s["process_rss_bytes"]) for s in samples if s["process_rss_bytes"] is not None]
    active_values = [int(s["mlx_active_memory_bytes"]) for s in samples if s["mlx_active_memory_bytes"] is not None]
    peak_values = [int(s["mlx_peak_memory_bytes"]) for s in samples if s["mlx_peak_memory_bytes"] is not None]
    load1_values = [float(s["load1"]) for s in samples if s["load1"] is not None]

    phase_counts: dict[str, int] = {}
    for sample in samples:
        phase = str(sample["phase"])
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    return {
        "enabled": True,
        "sample_seconds": sample_seconds,
        "sample_count": len(samples),
        "device": device_info,
        "phase_counts": phase_counts,
        "process_cpu_percent_mean": _safe_stat(cpu_values, lambda vals: sum(vals) / len(vals)),
        "process_cpu_percent_max": _safe_stat(cpu_values, max),
        "process_rss_bytes_max": _safe_stat([float(v) for v in rss_values], max),
        "system_load1_mean": _safe_stat(load1_values, lambda vals: sum(vals) / len(vals)),
        "system_load1_max": _safe_stat(load1_values, max),
        "mlx_active_memory_bytes_max": _safe_stat([float(v) for v in active_values], max),
        "mlx_peak_memory_bytes_max": _safe_stat([float(v) for v in peak_values], max),
    }


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
    parser.add_argument("--instrument", action="store_true", help="Collect system + MLX utilization samples")
    parser.add_argument("--instrument-sample-seconds", type=float, default=1.0)
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"model path not found: {args.model_path}")
    if not args.mono_path.exists() or not args.stereo_path.exists():
        raise FileNotFoundError("audio fixture path missing")

    run_dir = args.output_dir / args.label
    run_dir.mkdir(parents=True, exist_ok=True)

    mono_clip = _write_clip(args.mono_path, run_dir, args.clip_seconds)
    stereo_clip = _write_clip(args.stereo_path, run_dir, args.clip_seconds)

    instrument_sampler = InstrumentSampler(args.instrument_sample_seconds) if args.instrument else None
    instrumentation: dict[str, object] = {"enabled": False}

    if instrument_sampler is not None:
        instrument_sampler.set_phase("startup")
        instrument_sampler.start()

    try:
        if instrument_sampler is not None:
            instrument_sampler.set_phase("model_load")
        model, sp, _ = load_model(str(args.model_path))
        prompt_tokens, n_delay_tokens = _build_prompt_tokens(sp)

        # warmup to reduce first-run compilation effects
        warmup_clip = _write_clip(args.mono_path, run_dir, args.warmup_seconds)
        if instrument_sampler is not None:
            instrument_sampler.set_phase("warmup")
        _transcribe_once(model, sp, warmup_clip, prompt_tokens, n_delay_tokens)

        if instrument_sampler is not None:
            instrument_sampler.set_phase("mono")
        mono_text, mono_tokens, mono_elapsed = _transcribe_once(model, sp, mono_clip, prompt_tokens, n_delay_tokens)
        _save_text(run_dir / "mono_transcript.txt", mono_text)

        if args.create_ground_truth:
            _save_text(args.ground_truth_path, mono_text)

        if not args.ground_truth_path.exists():
            raise FileNotFoundError(
                f"ground truth missing: {args.ground_truth_path}; run once with --create-ground-truth"
            )

        truth = args.ground_truth_path.read_text(encoding="utf-8")

        if instrument_sampler is not None:
            instrument_sampler.set_phase("stereo")
        stereo_text, stereo_tokens, stereo_elapsed = _transcribe_once(
            model, sp, stereo_clip, prompt_tokens, n_delay_tokens
        )
        _save_text(run_dir / "stereo_transcript.txt", stereo_text)
    finally:
        if instrument_sampler is not None:
            instrument_sampler.set_phase("finalize")
            instrument_sampler.stop()
            device_info = _mlx_device_details()
            instrumentation = _instrumentation_summary(
                sampler=instrument_sampler,
                sample_seconds=args.instrument_sample_seconds,
                device_info=device_info,
            )
            samples_path = run_dir / "system_samples.json"
            samples_path.write_text(
                json.dumps(instrument_sampler.samples, indent=2, sort_keys=True),
                encoding="utf-8",
            )

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
        "instrumentation": instrumentation,
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
