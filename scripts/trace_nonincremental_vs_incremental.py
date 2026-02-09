#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from voxmlx import _build_prompt_tokens, load_model
from voxmlx import audio as audio_module
from voxmlx.audio import (
    SAMPLES_PER_TOKEN,
    load_audio,
    log_mel_spectrogram,
    log_mel_spectrogram_step,
    pad_audio,
)
from voxmlx.cache import RotatingKVCache
from voxmlx.constants import DEFAULT_CLEAR_CACHE_INTERVAL, DEFAULT_DECODER_SLIDING_WINDOW
from voxmlx.generate import generate


DEFAULT_AUDIO = Path(
    "perf/reference_audio/Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_10min.wav"
)
DEFAULT_MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--mlx-community--Voxtral-Mini-4B-Realtime-6bit/snapshots/02eb0caeb9dafb554c17a72b93dbf40cd3736c31"
)


@dataclass
class RunStats:
    method: str
    elapsed_seconds: float
    token_count: int
    special_count: int
    special_ratio: float
    text_chars_keep: int
    text_chars_ignore: int


def _levenshtein(seq_a: list[Any], seq_b: list[Any]) -> int:
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    if not seq_b:
        return len(seq_a)

    prev = list(range(len(seq_b) + 1))
    for i, a in enumerate(seq_a, start=1):
        cur = [i]
        for j, b in enumerate(seq_b, start=1):
            cur.append(
                min(
                    cur[j - 1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + (a != b),
                )
            )
        prev = cur
    return prev[-1]


def _normalize_meaning(text: str) -> str:
    # Ignore whitespace and only "." / "," punctuation; ignore capitalization.
    # Do not normalize special tokens away.
    out = []
    for ch in text:
        if ch.isspace() or ch in {".", ","}:
            continue
        out.append(ch.lower())
    return "".join(out)


def _decode_ids(sp: Any, token_ids: list[int], policy: SpecialTokenPolicy) -> str:
    if not token_ids:
        return ""
    return sp.decode(token_ids, special_token_policy=policy)


def _decode_one(sp: Any, token_id: int) -> str:
    return _decode_ids(sp, [int(token_id)], SpecialTokenPolicy.KEEP)


def _count_special(sp: Any, token_ids: list[int]) -> int:
    return sum(1 for tid in token_ids if bool(sp.is_special(tid)))


def _first_divergence_index(a: list[int], b: list[int]) -> int | None:
    lim = min(len(a), len(b))
    for i in range(lim):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return lim
    return None


def _build_nonincremental_audio_embeds(model: Any, audio: np.ndarray) -> mx.array:
    mel = log_mel_spectrogram(pad_audio(audio))
    return model.encode(mel)


def _build_incremental_audio_embeds(model: Any, audio: np.ndarray) -> mx.array:
    audio_padded = pad_audio(audio)

    audio_tail = None
    conv1_tail = None
    conv2_tail = None
    encoder_cache = None
    ds_buf = None
    parts = []

    for i in range(0, len(audio_padded), SAMPLES_PER_TOKEN):
        chunk = audio_padded[i : i + SAMPLES_PER_TOKEN]
        mel_step, audio_tail = log_mel_spectrogram_step(chunk, audio_tail)
        out, conv1_tail, conv2_tail, encoder_cache, ds_buf = model.encode_step(
            mel_step,
            conv1_tail,
            conv2_tail,
            encoder_cache,
            ds_buf,
        )
        if out is not None and out.shape[0] > 0:
            parts.append(out)

    hidden = int(getattr(model.language_model, "_dim", 0))
    if not parts:
        return mx.zeros((0, hidden), dtype=mx.float32)
    return mx.concatenate(parts, axis=0)


def _topk_logits(
    logits: mx.array,
    sp: Any | None,
    topk: int,
) -> dict[str, Any]:
    arr = np.array(logits[0, -1].astype(mx.float32))
    k = max(1, min(int(topk), int(arr.shape[0])))
    idx = np.argpartition(arr, -k)[-k:]
    idx = idx[np.argsort(arr[idx])[::-1]]
    ids = [int(i) for i in idx.tolist()]
    vals = [float(arr[i]) for i in ids]
    payload: dict[str, Any] = {
        "argmax_id": ids[0],
        "argmax_logit": vals[0],
        "top_ids": ids,
        "top_logits": vals,
    }
    if sp is not None:
        payload["top_text_keep"] = [_decode_one(sp, i) for i in ids]
        payload["top_is_special"] = [bool(sp.is_special(i)) for i in ids]
    return payload


def _vector_stats(vec: mx.array) -> dict[str, float]:
    v = np.array(vec.astype(mx.float32))
    return {
        "l2_norm": float(np.linalg.norm(v)),
        "mean_abs": float(np.mean(np.abs(v))),
        "max_abs": float(np.max(np.abs(v))),
    }


def _vector_diff_stats(vec_a: mx.array, vec_b: mx.array) -> dict[str, float]:
    a = np.array(vec_a.astype(mx.float32))
    b = np.array(vec_b.astype(mx.float32))
    d = a - b
    return {
        "l2_norm_diff": float(np.linalg.norm(d)),
        "mean_abs_diff": float(np.mean(np.abs(d))),
        "max_abs_diff": float(np.max(np.abs(d))),
    }


def _decode_from_audio_embeds(
    model: Any,
    audio_embeds: mx.array,
    prompt_tokens: list[int],
    n_delay_tokens: int,
    eos_token_id: int,
    temperature: float,
    sliding_window: int,
    sp: Any | None = None,
    focus_indices: set[int] | None = None,
    trace_topk: int = 8,
) -> tuple[list[int], list[dict[str, Any]]]:
    n_audio = int(audio_embeds.shape[0])
    if n_audio == 0:
        return [], []

    t_cond = model.time_embedding(mx.array([n_delay_tokens], dtype=mx.float32))
    prefix_len = len(prompt_tokens)
    if n_audio < prefix_len:
        return [], []

    prompt_ids = mx.array([prompt_tokens])
    text_embeds = model.language_model.embed(prompt_ids)[0]
    prefix_embeds = (text_embeds + audio_embeds[:prefix_len])[None, :, :]

    n_layers = len(model.language_model.layers)
    cache = [RotatingKVCache(sliding_window) for _ in range(n_layers)]

    def sample(logits: mx.array) -> mx.array:
        if temperature <= 0:
            return mx.argmax(logits[0, -1:], axis=-1).squeeze()
        return mx.random.categorical(logits[0, -1:] / temperature).squeeze()

    traces: list[dict[str, Any]] = []
    layer_probe_indices = sorted(set([0, max(0, n_layers // 2), max(0, n_layers - 1)]))

    prefill_logits = model.decode(prefix_embeds, t_cond, "causal", cache)
    mx.eval(prefill_logits, *[x for c in cache for x in (c.keys, c.values)])

    y = sample(prefill_logits)
    mx.async_eval(y)

    producer_diag: dict[str, Any] = {
        "source": "prefill",
        "audio_pos": None,
        "input_token_id": None,
        "input_token_text_keep": None,
        "audio_embed_stats": None,
        "step_embed_stats": None,
        "logits_topk": (
            _topk_logits(prefill_logits, sp, trace_topk)
            if focus_indices is not None and 0 in focus_indices
            else None
        ),
    }

    output_tokens: list[int] = []
    for pos in range(prefix_len, n_audio):
        output_index = len(output_tokens)
        trace_next = focus_indices is not None and (output_index + 1) in focus_indices

        # Compute next token first (matching generate.py semantics).
        token_input_id = int(y.item())
        token_embed = model.language_model.embed(y.reshape(1, 1))[0, 0]
        audio_vec = audio_embeds[pos]
        step_embed = (audio_vec + token_embed)[None, None, :]
        next_logits = model.decode(step_embed, t_cond, mask=None, cache=cache)
        next_y = sample(next_logits)
        mx.async_eval(next_y)

        next_producer_diag: dict[str, Any] = {
            "source": "step",
            "audio_pos": int(pos),
            "input_token_id": token_input_id,
            "input_token_text_keep": _decode_one(sp, token_input_id) if sp is not None else None,
            "audio_embed_stats": _vector_stats(audio_vec) if trace_next else None,
            "step_embed_stats": _vector_stats(step_embed[0, 0]) if trace_next else None,
            "logits_topk": _topk_logits(next_logits, sp, trace_topk) if trace_next else None,
        }

        token_id = token_input_id
        if focus_indices is not None and output_index in focus_indices:
            traces.append(
                {
                    "output_index": int(output_index),
                    "token_id": int(token_id),
                    "token_text_keep": _decode_one(sp, token_id) if sp is not None else None,
                    "token_is_special": bool(sp.is_special(token_id)) if sp is not None else None,
                    "producer": producer_diag,
                    "cache_offsets": {
                        str(i): int(cache[i].offset) for i in layer_probe_indices
                    },
                }
            )

        if token_id == eos_token_id:
            break
        output_tokens.append(token_id)

        if pos % DEFAULT_CLEAR_CACHE_INTERVAL == 0:
            mx.clear_cache()

        y = next_y
        producer_diag = next_producer_diag

    # Flush last pending token as generate.py does.
    token_id = int(y.item())
    output_index = len(output_tokens)
    if focus_indices is not None and output_index in focus_indices:
        traces.append(
            {
                "output_index": int(output_index),
                "token_id": int(token_id),
                "token_text_keep": _decode_one(sp, token_id) if sp is not None else None,
                "token_is_special": bool(sp.is_special(token_id)) if sp is not None else None,
                "producer": producer_diag,
                "cache_offsets": {
                    str(i): int(cache[i].offset) for i in layer_probe_indices
                },
                "flush_pending": True,
            }
        )

    if token_id != eos_token_id:
        output_tokens.append(token_id)

    return output_tokens, traces


def _pair_focus_traces(
    noninc_trace: list[dict[str, Any]],
    inc_trace: list[dict[str, Any]],
    noninc_embeds: mx.array,
    inc_embeds: mx.array,
) -> list[dict[str, Any]]:
    map_non = {int(x["output_index"]): x for x in noninc_trace}
    map_inc = {int(x["output_index"]): x for x in inc_trace}
    keys = sorted(set(map_non) | set(map_inc))
    paired: list[dict[str, Any]] = []

    for idx in keys:
        left = map_non.get(idx)
        right = map_inc.get(idx)
        if left is None or right is None:
            paired.append(
                {
                    "output_index": idx,
                    "present_in_non_incremental": left is not None,
                    "present_in_incremental": right is not None,
                }
            )
            continue

        lp = left.get("producer", {})
        rp = right.get("producer", {})
        l_audio_pos = lp.get("audio_pos")
        r_audio_pos = rp.get("audio_pos")
        audio_diff = None
        if l_audio_pos is not None and r_audio_pos is not None:
            if 0 <= int(l_audio_pos) < int(noninc_embeds.shape[0]) and 0 <= int(r_audio_pos) < int(inc_embeds.shape[0]):
                audio_diff = _vector_diff_stats(
                    noninc_embeds[int(l_audio_pos)],
                    inc_embeds[int(r_audio_pos)],
                )

        l_top = lp.get("logits_topk") or {}
        r_top = rp.get("logits_topk") or {}
        l_top_ids = set(int(x) for x in l_top.get("top_ids", []))
        r_top_ids = set(int(x) for x in r_top.get("top_ids", []))

        paired.append(
            {
                "output_index": idx,
                "token_non_incremental": left.get("token_id"),
                "token_incremental": right.get("token_id"),
                "token_equal": left.get("token_id") == right.get("token_id"),
                "token_non_incremental_text_keep": left.get("token_text_keep"),
                "token_incremental_text_keep": right.get("token_text_keep"),
                "producer_source_non_incremental": lp.get("source"),
                "producer_source_incremental": rp.get("source"),
                "producer_audio_pos_non_incremental": l_audio_pos,
                "producer_audio_pos_incremental": r_audio_pos,
                "producer_input_token_non_incremental": lp.get("input_token_id"),
                "producer_input_token_incremental": rp.get("input_token_id"),
                "producer_argmax_non_incremental": l_top.get("argmax_id"),
                "producer_argmax_incremental": r_top.get("argmax_id"),
                "producer_topk_overlap_count": len(l_top_ids & r_top_ids),
                "producer_topk_overlap_ids": sorted(l_top_ids & r_top_ids),
                "audio_embed_diff": audio_diff,
            }
        )
    return paired


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace and compare non-incremental vs incremental decoding on the same clipped audio."
    )
    parser.add_argument("--audio-path", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--clip-seconds", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("perf/audio_runs"))
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--max-diff-spans", type=int, default=50)
    parser.add_argument("--focus-token-index", type=int, default=None)
    parser.add_argument("--focus-window", type=int, default=2)
    parser.add_argument("--trace-topk", type=int, default=8)
    args = parser.parse_args()

    if not args.audio_path.exists():
        raise FileNotFoundError(f"audio path not found: {args.audio_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"model path not found: {args.model_path}")
    if args.clip_seconds <= 0:
        raise ValueError("--clip-seconds must be > 0")
    if args.clip_seconds > 120:
        raise ValueError("--clip-seconds must be <= 120 for this debug workflow")
    if args.focus_window < 0:
        raise ValueError("--focus-window must be >= 0")
    if args.trace_topk <= 0:
        raise ValueError("--trace-topk must be > 0")

    focus_indices: set[int] | None = None
    if args.focus_token_index is not None:
        if args.focus_token_index < 0:
            raise ValueError("--focus-token-index must be >= 0")
        lo = max(0, int(args.focus_token_index) - int(args.focus_window))
        hi = int(args.focus_token_index) + int(args.focus_window)
        focus_indices = set(range(lo, hi + 1))

    label = args.label or f"trace-noninc-vs-inc-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = args.output_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)

    model, sp, config = load_model(str(args.model_path))
    prompt_tokens, n_delay_tokens = _build_prompt_tokens(sp)
    sliding_window = (
        int(args.sliding_window)
        if args.sliding_window is not None
        else int(config.get("sliding_window", DEFAULT_DECODER_SLIDING_WINDOW))
    )

    raw_audio = load_audio(str(args.audio_path))
    clip_samples = int(args.clip_seconds * 16000)
    audio = raw_audio[:clip_samples]

    # Non-incremental baseline path.
    t0 = time.perf_counter()
    noninc_ids = generate(
        model,
        str(args.audio_path),
        prompt_tokens,
        n_delay_tokens=n_delay_tokens,
        temperature=args.temperature,
        eos_token_id=sp.eos_id,
        sliding_window=sliding_window,
        audio_override=audio,
    )
    noninc_elapsed = time.perf_counter() - t0

    # Incremental path + trace support.
    t1 = time.perf_counter()
    inc_embeds = _build_incremental_audio_embeds(model, audio)
    inc_ids, inc_trace = _decode_from_audio_embeds(
        model=model,
        audio_embeds=inc_embeds,
        prompt_tokens=prompt_tokens,
        n_delay_tokens=n_delay_tokens,
        eos_token_id=sp.eos_id,
        temperature=args.temperature,
        sliding_window=sliding_window,
        sp=sp,
        focus_indices=focus_indices,
        trace_topk=args.trace_topk,
    )
    inc_elapsed = time.perf_counter() - t1

    # Optional deep replay for non-incremental (needed for detailed per-step comparisons).
    noninc_trace: list[dict[str, Any]] = []
    noninc_embeds = None
    noninc_replay_matches_generate = None
    paired_focus = []
    if focus_indices is not None:
        noninc_embeds = _build_nonincremental_audio_embeds(model, audio)
        noninc_ids_replay, noninc_trace = _decode_from_audio_embeds(
            model=model,
            audio_embeds=noninc_embeds,
            prompt_tokens=prompt_tokens,
            n_delay_tokens=n_delay_tokens,
            eos_token_id=sp.eos_id,
            temperature=args.temperature,
            sliding_window=sliding_window,
            sp=sp,
            focus_indices=focus_indices,
            trace_topk=args.trace_topk,
        )
        noninc_replay_matches_generate = noninc_ids_replay == noninc_ids
        paired_focus = _pair_focus_traces(noninc_trace, inc_trace, noninc_embeds, inc_embeds)

    noninc_text_keep = _decode_ids(sp, noninc_ids, SpecialTokenPolicy.KEEP)
    inc_text_keep = _decode_ids(sp, inc_ids, SpecialTokenPolicy.KEEP)
    noninc_text_ignore = _decode_ids(sp, noninc_ids, SpecialTokenPolicy.IGNORE)
    inc_text_ignore = _decode_ids(sp, inc_ids, SpecialTokenPolicy.IGNORE)

    noninc_special = _count_special(sp, noninc_ids)
    inc_special = _count_special(sp, inc_ids)

    noninc_stats = RunStats(
        method="non_incremental",
        elapsed_seconds=noninc_elapsed,
        token_count=len(noninc_ids),
        special_count=noninc_special,
        special_ratio=(noninc_special / len(noninc_ids)) if noninc_ids else 0.0,
        text_chars_keep=len(noninc_text_keep),
        text_chars_ignore=len(noninc_text_ignore),
    )
    inc_stats = RunStats(
        method="incremental",
        elapsed_seconds=inc_elapsed,
        token_count=len(inc_ids),
        special_count=inc_special,
        special_ratio=(inc_special / len(inc_ids)) if inc_ids else 0.0,
        text_chars_keep=len(inc_text_keep),
        text_chars_ignore=len(inc_text_ignore),
    )

    token_dist = _levenshtein(noninc_ids, inc_ids)
    token_norm = token_dist / max(1, len(inc_ids))
    char_dist_keep = _levenshtein(list(noninc_text_keep), list(inc_text_keep))
    char_norm_keep = char_dist_keep / max(1, len(inc_text_keep))
    first_div = _first_divergence_index(noninc_ids, inc_ids)

    matcher = SequenceMatcher(a=noninc_ids, b=inc_ids, autojunk=False)
    diff_spans = []
    meaningful_spans = 0
    non_equal_span_count = 0
    first_meaningful_index = None
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        non_equal_span_count += 1
        left_ids = noninc_ids[i1:i2]
        right_ids = inc_ids[j1:j2]
        left_text_keep = _decode_ids(sp, left_ids, SpecialTokenPolicy.KEEP)
        right_text_keep = _decode_ids(sp, right_ids, SpecialTokenPolicy.KEEP)
        left_norm = _normalize_meaning(left_text_keep)
        right_norm = _normalize_meaning(right_text_keep)
        is_meaningful = left_norm != right_norm
        if is_meaningful:
            meaningful_spans += 1
            if first_meaningful_index is None:
                first_meaningful_index = i1
        if len(diff_spans) < args.max_diff_spans:
            diff_spans.append(
                {
                    "tag": tag,
                    "noninc_range": [i1, i2],
                    "inc_range": [j1, j2],
                    "noninc_ids": left_ids,
                    "inc_ids": right_ids,
                    "noninc_is_special": [bool(sp.is_special(t)) for t in left_ids],
                    "inc_is_special": [bool(sp.is_special(t)) for t in right_ids],
                    "noninc_text_keep": left_text_keep,
                    "inc_text_keep": right_text_keep,
                    "noninc_norm": left_norm,
                    "inc_norm": right_norm,
                    "meaningful": is_meaningful,
                }
            )

    (run_dir / "non_incremental_token_ids.json").write_text(
        json.dumps(noninc_ids),
        encoding="utf-8",
    )
    (run_dir / "incremental_token_ids.json").write_text(
        json.dumps(inc_ids),
        encoding="utf-8",
    )
    (run_dir / "non_incremental_text_keep.txt").write_text(noninc_text_keep, encoding="utf-8")
    (run_dir / "incremental_text_keep.txt").write_text(inc_text_keep, encoding="utf-8")
    (run_dir / "non_incremental_text_ignore.txt").write_text(noninc_text_ignore, encoding="utf-8")
    (run_dir / "incremental_text_ignore.txt").write_text(inc_text_ignore, encoding="utf-8")

    if focus_indices is not None:
        (run_dir / "non_incremental_focus_trace.json").write_text(
            json.dumps(noninc_trace, indent=2),
            encoding="utf-8",
        )
        (run_dir / "incremental_focus_trace.json").write_text(
            json.dumps(inc_trace, indent=2),
            encoding="utf-8",
        )
        (run_dir / "focus_pairwise.json").write_text(
            json.dumps(paired_focus, indent=2),
            encoding="utf-8",
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audio_path": str(args.audio_path),
        "model_path": str(args.model_path),
        "clip_seconds": args.clip_seconds,
        "stft_backend": getattr(audio_module, "_STFT_BACKEND", "unknown"),
        "temperature": args.temperature,
        "sliding_window": sliding_window,
        "focus": {
            "focus_token_index": args.focus_token_index,
            "focus_window": args.focus_window,
            "focus_indices": sorted(focus_indices) if focus_indices is not None else [],
            "trace_topk": args.trace_topk,
            "non_incremental_replay_matches_generate": noninc_replay_matches_generate,
        },
        "stats": {
            "non_incremental": asdict(noninc_stats),
            "incremental": asdict(inc_stats),
        },
        "distance": {
            "token_levenshtein": token_dist,
            "token_norm_vs_incremental": token_norm,
            "char_levenshtein_keep": char_dist_keep,
            "char_norm_keep_vs_incremental": char_norm_keep,
        },
        "divergence": {
            "first_token_divergence_index": first_div,
            "first_meaningful_divergence_index": first_meaningful_index,
            "non_equal_span_count": non_equal_span_count,
            "meaningful_span_count": meaningful_spans,
        },
        "diff_spans": diff_spans,
        "focus_pairwise": paired_focus,
    }
    (run_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        "# Non-incremental vs Incremental Trace Comparison",
        "",
        f"- audio: `{args.audio_path}`",
        f"- clip_seconds: `{args.clip_seconds}`",
        f"- stft_backend: `{payload['stft_backend']}`",
        f"- non_incremental elapsed: `{noninc_elapsed:.3f}s`",
        f"- incremental elapsed: `{inc_elapsed:.3f}s`",
        f"- token distance: `{token_dist}` (norm `{token_norm:.6f}`)",
        f"- char distance (KEEP): `{char_dist_keep}` (norm `{char_norm_keep:.6f}`)",
        f"- first token divergence index: `{first_div}`",
        f"- first meaningful divergence index: `{first_meaningful_index}`",
        f"- meaningful span count: `{meaningful_spans}`",
    ]
    if focus_indices is not None:
        summary_lines.extend(
            [
                f"- focus indices: `{sorted(focus_indices)}`",
                f"- non_incremental replay matches generate: `{noninc_replay_matches_generate}`",
            ]
        )
    summary_lines.extend(["", "## First diff spans"])

    for idx, span in enumerate(diff_spans[:10], start=1):
        summary_lines.append(
            f"{idx}. `{span['tag']}` noninc[{span['noninc_range'][0]}:{span['noninc_range'][1]}] "
            f"vs inc[{span['inc_range'][0]}:{span['inc_range'][1]}], meaningful=`{span['meaningful']}`"
        )
        summary_lines.append(f"   - noninc: `{span['noninc_text_keep']}`")
        summary_lines.append(f"   - inc: `{span['inc_text_keep']}`")

    if paired_focus:
        summary_lines.extend(["", "## Focus Pairwise"])
        for row in paired_focus[: min(10, len(paired_focus))]:
            if not row.get("present_in_non_incremental", True) or not row.get("present_in_incremental", True):
                summary_lines.append(f"- idx {row['output_index']}: missing in one side")
                continue
            summary_lines.append(
                f"- idx {row['output_index']}: "
                f"token_equal={row['token_equal']} "
                f"noninc={row['token_non_incremental']} "
                f"inc={row['token_incremental']} "
                f"audio_diff_max_abs={None if row['audio_embed_diff'] is None else row['audio_embed_diff']['max_abs_diff']:.6f}"
                if row["audio_embed_diff"] is not None
                else f"- idx {row['output_index']}: token_equal={row['token_equal']} noninc={row['token_non_incremental']} inc={row['token_incremental']}"
            )

    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[trace] wrote {run_dir / 'comparison.json'}")
    print(
        "[trace] "
        f"non_incremental={noninc_elapsed:.3f}s "
        f"incremental={inc_elapsed:.3f}s "
        f"token_norm={token_norm:.6f} "
        f"first_div={first_div} "
        f"first_meaningful_div={first_meaningful_index}"
    )
    if focus_indices is not None:
        print(
            "[trace-focus] "
            f"indices={sorted(focus_indices)} "
            f"noninc_replay_matches_generate={noninc_replay_matches_generate} "
            f"pair_rows={len(paired_focus)}"
        )


if __name__ == "__main__":
    main()
