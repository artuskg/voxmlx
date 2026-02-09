#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from voxmlx import _build_prompt_tokens, load_model
from voxmlx.audio import SAMPLES_PER_TOKEN, load_audio, log_mel_spectrogram_step, pad_audio
from voxmlx.cache import RotatingKVCache
from voxmlx.constants import DEFAULT_DECODER_SLIDING_WINDOW


def _git_short_sha(cwd: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def _build_incremental_audio_embeds(model, audio: np.ndarray) -> tuple[mx.array, int]:
    audio_padded = pad_audio(audio)

    audio_tail = None
    conv1_tail = None
    conv2_tail = None
    encoder_cache = None
    ds_buf = None
    parts: list[mx.array] = []

    for start in range(0, len(audio_padded), SAMPLES_PER_TOKEN):
        chunk = audio_padded[start : start + SAMPLES_PER_TOKEN]
        mel_step, audio_tail = log_mel_spectrogram_step(chunk, audio_tail)
        out, conv1_tail, conv2_tail, encoder_cache, ds_buf = model.encode_step(
            mel_step,
            conv1_tail,
            conv2_tail,
            encoder_cache,
            ds_buf,
        )
        if out is not None and out.shape[0] > 0:
            mx.eval(out)
            parts.append(out)

    hidden_dim = int(getattr(model.language_model, "_dim", 0))
    if not parts:
        return mx.zeros((0, hidden_dim), dtype=mx.float32), len(audio_padded)
    return mx.concatenate(parts, axis=0), len(audio_padded)


def _decode_from_audio_embeds(
    model,
    sp,
    audio_embeds: mx.array,
    prompt_tokens: list[int],
    n_delay_tokens: int,
    temperature: float,
    sliding_window: int,
) -> list[int]:
    n_audio = int(audio_embeds.shape[0])
    prefix_len = len(prompt_tokens)
    if n_audio < prefix_len:
        return []

    t_cond = model.time_embedding(mx.array([n_delay_tokens], dtype=mx.float32))
    prompt_ids = mx.array([prompt_tokens])
    text_embeds = model.language_model.embed(prompt_ids)[0]
    prefix_embeds = (text_embeds + audio_embeds[:prefix_len])[None, :, :]

    n_layers = len(model.language_model.layers)
    cache = [RotatingKVCache(sliding_window) for _ in range(n_layers)]

    def sample(logits: mx.array) -> mx.array:
        if temperature <= 0:
            return mx.argmax(logits[0, -1:], axis=-1).squeeze()
        return mx.random.categorical(logits[0, -1:] / temperature).squeeze()

    prefill_logits = model.decode(prefix_embeds, t_cond, "causal", cache)
    mx.eval(prefill_logits, *[x for c in cache for x in (c.keys, c.values)])
    y = sample(prefill_logits)

    output_tokens: list[int] = []
    for pos in range(prefix_len, n_audio):
        token_id = int(y.item())
        if token_id == sp.eos_id:
            break
        output_tokens.append(token_id)

        token_embed = model.language_model.embed(y.reshape(1, 1))[0, 0]
        step_embed = (audio_embeds[pos] + token_embed)[None, None, :]
        logits = model.decode(step_embed, t_cond, mask=None, cache=cache)
        mx.eval(logits)
        y = sample(logits)

    # Match generate.py semantics: flush final pending token when not EOS.
    token_id = int(y.item())
    if token_id != sp.eos_id:
        output_tokens.append(token_id)

    return output_tokens


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(sum(values) / len(values))
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, float(var**0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental-file transcription eval (encode_step path only)")
    parser.add_argument("--label", required=True, help="Run label")
    parser.add_argument("--commit", default=None, help="Commit SHA label (default: git short sha)")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--audio-path", required=True, type=Path)
    parser.add_argument("--clip-seconds", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("perf/audio_runs"))
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save-all-repeats", action="store_true")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"model path not found: {args.model_path}")
    if not args.audio_path.exists():
        raise FileNotFoundError(f"audio path not found: {args.audio_path}")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")

    run_dir = args.output_dir / args.label
    run_dir.mkdir(parents=True, exist_ok=True)

    model, sp, config = load_model(str(args.model_path))
    prompt_tokens, n_delay_tokens = _build_prompt_tokens(sp)
    if args.sliding_window is None:
        sliding_window = int(config.get("sliding_window", DEFAULT_DECODER_SLIDING_WINDOW))
    else:
        sliding_window = int(args.sliding_window)

    audio = load_audio(str(args.audio_path))
    if args.clip_seconds > 0:
        max_samples = args.clip_seconds * 16000
        audio = audio[:max_samples]

    def run_once() -> dict[str, object]:
        t0 = time.perf_counter()
        audio_embeds, padded_samples = _build_incremental_audio_embeds(model, audio)
        t1 = time.perf_counter()
        token_ids = _decode_from_audio_embeds(
            model=model,
            sp=sp,
            audio_embeds=audio_embeds,
            prompt_tokens=prompt_tokens,
            n_delay_tokens=n_delay_tokens,
            temperature=args.temperature,
            sliding_window=sliding_window,
        )
        t2 = time.perf_counter()

        text_ignore = sp.decode(token_ids, special_token_policy=SpecialTokenPolicy.IGNORE)
        text_keep = sp.decode(token_ids, special_token_policy=SpecialTokenPolicy.KEEP)
        return {
            "padded_samples": int(padded_samples),
            "audio_tokens": int(audio_embeds.shape[0]),
            "token_count": int(len(token_ids)),
            "elapsed_encode_seconds": float(t1 - t0),
            "elapsed_decode_seconds": float(t2 - t1),
            "elapsed_total_seconds": float(t2 - t0),
            "text_ignore": text_ignore,
            "text_keep": text_keep,
        }

    for _ in range(args.warmup_repeats):
        _ = run_once()

    repeats_payload: list[dict[str, object]] = []
    totals: list[float] = []
    encodes: list[float] = []
    decodes: list[float] = []

    for i in range(1, args.repeats + 1):
        payload = run_once()
        payload["repeat_index"] = i
        repeats_payload.append(payload)
        totals.append(float(payload["elapsed_total_seconds"]))
        encodes.append(float(payload["elapsed_encode_seconds"]))
        decodes.append(float(payload["elapsed_decode_seconds"]))

        if args.save_all_repeats:
            (run_dir / f"mono_incremental_transcript_r{i:02d}.txt").write_text(
                str(payload["text_ignore"]),
                encoding="utf-8",
            )

    first = repeats_payload[0]
    (run_dir / "mono_incremental_transcript.txt").write_text(str(first["text_ignore"]), encoding="utf-8")
    (run_dir / "mono_incremental_transcript_keep.txt").write_text(str(first["text_keep"]), encoding="utf-8")

    mean_total, std_total = _mean_std(totals)
    mean_encode, std_encode = _mean_std(encodes)
    mean_decode, std_decode = _mean_std(decodes)

    commit = args.commit or _git_short_sha(Path.cwd())
    summary = {
        "label": args.label,
        "commit": commit,
        "method": "incremental_file_pipeline",
        "audio_path": str(args.audio_path.resolve()),
        "model_path": str(args.model_path.resolve()),
        "clip_seconds": int(args.clip_seconds),
        "temperature": float(args.temperature),
        "sliding_window": int(sliding_window),
        "repeats": int(args.repeats),
        "warmup_repeats": int(args.warmup_repeats),
        "stft_backend_env": str(os.environ.get("VOXMLX_STFT_BACKEND", "<default>")),
        "mean_elapsed_total_seconds": mean_total,
        "std_elapsed_total_seconds": std_total,
        "mean_elapsed_encode_seconds": mean_encode,
        "std_elapsed_encode_seconds": std_encode,
        "mean_elapsed_decode_seconds": mean_decode,
        "std_elapsed_decode_seconds": std_decode,
        "text_char_count": len(str(first["text_ignore"])),
        "token_count": int(first["token_count"]),
        "audio_tokens": int(first["audio_tokens"]),
        "n_left_pad_tokens": len(prompt_tokens) - 1 - n_delay_tokens,
        "n_delay_tokens": int(n_delay_tokens),
        "output_path": str((run_dir / "mono_incremental_transcript.txt").resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [
            {
                "repeat_index": int(r["repeat_index"]),
                "elapsed_total_seconds": float(r["elapsed_total_seconds"]),
                "elapsed_encode_seconds": float(r["elapsed_encode_seconds"]),
                "elapsed_decode_seconds": float(r["elapsed_decode_seconds"]),
                "token_count": int(r["token_count"]),
            }
            for r in repeats_payload
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[incremental-eval] wrote {run_dir / 'summary.json'}")
    print(
        "[incremental-eval]"
        f" mean_total={mean_total:.6f}s"
        f" std_total={std_total:.6f}s"
        f" mean_encode={mean_encode:.6f}s"
        f" mean_decode={mean_decode:.6f}s"
        f" token_count={int(first['token_count'])}"
    )


if __name__ == "__main__":
    main()
