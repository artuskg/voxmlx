import argparse
import threading
import time
from collections import deque

import mlx.core as mx
import numpy as np
import sounddevice as sd
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from . import _build_prompt_tokens, load_model
from .audio import SAMPLES_PER_TOKEN, log_mel_spectrogram_step
from .cache import RotatingKVCache
from .constants import (
    DEFAULT_CLEAR_CACHE_INTERVAL,
    DEFAULT_DECODER_SLIDING_WINDOW,
    DEFAULT_RIGHT_PAD_TOKENS,
)


class AudioSampleQueue:
    """Chunked audio buffer that avoids O(n^2) append behavior."""

    def __init__(self):
        self._chunks = deque()
        self._len = 0

    def __len__(self):
        return self._len

    def clear(self):
        self._chunks.clear()
        self._len = 0

    def append(self, samples: np.ndarray):
        if samples is None or samples.size == 0:
            return
        arr = np.asarray(samples, dtype=np.float32)
        self._chunks.append(arr)
        self._len += arr.shape[0]

    def append_many(self, chunks):
        for chunk in chunks:
            self.append(chunk)

    def pop(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        if n > self._len:
            raise ValueError(f"Requested {n} samples with only {self._len} buffered")

        pieces = []
        remain = n
        while remain > 0:
            head = self._chunks[0]
            if head.shape[0] <= remain:
                pieces.append(self._chunks.popleft())
                remain -= head.shape[0]
            else:
                pieces.append(head[:remain])
                self._chunks[0] = head[remain:]
                remain = 0

        self._len -= n
        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces)


class EmbeddingQueue:
    """Chunked embedding buffer that minimizes repeated mx.concatenate calls."""

    def __init__(self):
        self._chunks = deque()
        self._len = 0
        self._feature_dim = None

    def __len__(self):
        return self._len

    def clear(self):
        self._chunks.clear()
        self._len = 0
        self._feature_dim = None

    def append(self, embeds: mx.array | None):
        if embeds is None:
            return
        if embeds.shape[0] == 0:
            return
        if self._feature_dim is None:
            self._feature_dim = embeds.shape[1]
        self._chunks.append(embeds)
        self._len += embeds.shape[0]

    def pop(self, n: int) -> mx.array:
        if n <= 0:
            hidden_dim = 0 if self._feature_dim is None else self._feature_dim
            return mx.zeros((0, hidden_dim))
        if n > self._len:
            raise ValueError(f"Requested {n} embeds with only {self._len} buffered")

        parts = []
        remain = n
        while remain > 0:
            head = self._chunks[0]
            h = head.shape[0]
            if h <= remain:
                parts.append(self._chunks.popleft())
                remain -= h
            else:
                parts.append(head[:remain])
                self._chunks[0] = head[remain:]
                remain = 0

        self._len -= n
        if len(parts) == 1:
            return parts[0]
        return mx.concatenate(parts, axis=0)


class StreamingTranscriber:
    """Incremental live transcriber with explicit encoder/decoder state."""

    def __init__(self, model, sp, config, temperature: float, sliding_window: int | None):
        self.model = model
        self.sp = sp
        self.config = config
        self.temperature = temperature

        prompt_tokens, n_delay_tokens = _build_prompt_tokens(sp)
        self.n_delay_tokens = int(n_delay_tokens)
        self.prefix_len = len(prompt_tokens)
        self.n_left_pad_tokens = self.prefix_len - 1 - self.n_delay_tokens
        if self.n_left_pad_tokens < 0:
            raise ValueError(
                f"Invalid prompt token layout: prefix_len={self.prefix_len}, "
                f"n_delay_tokens={self.n_delay_tokens}"
            )
        self.eos_token_id = sp.eos_id

        self.t_cond = model.time_embedding(mx.array([self.n_delay_tokens], dtype=mx.float32))
        mx.eval(self.t_cond)

        prompt_ids = mx.array([prompt_tokens])
        self.text_embeds = model.language_model.embed(prompt_ids)[0]  # [prefix_len, hidden]
        mx.eval(self.text_embeds)

        self.n_layers = len(model.language_model.layers)
        if sliding_window is None:
            sliding_window = int(config.get("sliding_window", DEFAULT_DECODER_SLIDING_WINDOW))
        self.sliding_window = sliding_window

        self.lock = threading.Lock()
        self.callback_chunks = deque()
        self._audio_status_count = 0
        self._last_audio_status = None

        self.reset_all_state()

    def reset_all_state(self):
        # Decoder state
        self.cache = None
        self.y = None

        # Incremental encoder state
        self.audio_tail = None
        self.conv1_tail = None
        self.conv2_tail = None
        self.encoder_cache = None
        self.ds_buf = None

        # Chunked buffers and counters
        self.pending_audio = AudioSampleQueue()
        self.audio_embeds = EmbeddingQueue()
        self.n_audio_samples_fed = 0
        self.n_total_decoded = 0
        self.first_cycle = True
        self.prefilled = False

    def callback(self, indata, frames, time_info, status):
        del frames, time_info
        chunk = indata[:, 0].copy()
        with self.lock:
            self.callback_chunks.append(chunk)
            if status:
                self._audio_status_count += 1
                self._last_audio_status = str(status)

    def _drain_callback_chunks(self):
        with self.lock:
            chunks = list(self.callback_chunks)
            self.callback_chunks.clear()
        return chunks

    def _drain_audio_status(self):
        with self.lock:
            count = self._audio_status_count
            last = self._last_audio_status
            self._audio_status_count = 0
            self._last_audio_status = None
        return count, last

    def sample(self, logits):
        if self.temperature <= 0:
            return mx.argmax(logits[0, -1:], axis=-1).squeeze()
        return mx.random.categorical(logits[0, -1:] / self.temperature).squeeze()

    def _encode_audio_chunk(self, chunk: np.ndarray):
        mel, self.audio_tail = log_mel_spectrogram_step(chunk, self.audio_tail)
        new_embeds, self.conv1_tail, self.conv2_tail, self.encoder_cache, self.ds_buf = (
            self.model.encode_step(
                mel, self.conv1_tail, self.conv2_tail, self.encoder_cache, self.ds_buf
            )
        )
        if new_embeds is not None:
            mx.eval(new_embeds)
            self.audio_embeds.append(new_embeds)

    def _feed_available_audio(self):
        if self.first_cycle and len(self.pending_audio) >= SAMPLES_PER_TOKEN:
            left_pad = np.zeros(
                self.n_left_pad_tokens * SAMPLES_PER_TOKEN, dtype=np.float32
            )
            n_feed = (len(self.pending_audio) // SAMPLES_PER_TOKEN) * SAMPLES_PER_TOKEN
            fed_audio = self.pending_audio.pop(n_feed)
            chunk = np.concatenate([left_pad, fed_audio])
            self.n_audio_samples_fed += n_feed
            self._encode_audio_chunk(chunk)
            self.first_cycle = False
            return

        if not self.first_cycle and len(self.pending_audio) >= SAMPLES_PER_TOKEN:
            n_feed = (len(self.pending_audio) // SAMPLES_PER_TOKEN) * SAMPLES_PER_TOKEN
            chunk = self.pending_audio.pop(n_feed)
            self.n_audio_samples_fed += n_feed
            self._encode_audio_chunk(chunk)

    def _n_decodable(self):
        safe_total = (
            self.n_left_pad_tokens + self.n_audio_samples_fed // SAMPLES_PER_TOKEN
        )
        return min(len(self.audio_embeds), safe_total - self.n_total_decoded)

    def _maybe_prefill(self):
        if self.prefilled:
            return True
        if self.n_total_decoded + len(self.audio_embeds) < self.prefix_len:
            return False

        self.cache = [RotatingKVCache(self.sliding_window) for _ in range(self.n_layers)]
        prefix_audio = self.audio_embeds.pop(self.prefix_len)
        prefix_embeds = (self.text_embeds + prefix_audio)[None, :, :]

        logits = self.model.decode(prefix_embeds, self.t_cond, "causal", self.cache)
        mx.eval(logits, *[x for c in self.cache for x in (c.keys, c.values)])

        self.y = self.sample(logits)
        mx.async_eval(self.y)
        self.n_total_decoded = self.prefix_len
        self.prefilled = True
        return True

    def decode_steps(self, embeds, n_to_decode):
        """Decode n_to_decode positions from embeds[0..n_to_decode-1]."""
        for i in range(n_to_decode):
            token_embed = self.model.language_model.embed(self.y.reshape(1, 1))[0, 0]
            step_embed = (embeds[i] + token_embed)[None, None, :]
            logits = self.model.decode(step_embed, self.t_cond, mask=None, cache=self.cache)
            next_y = self.sample(logits)
            mx.async_eval(next_y)

            token_id = self.y.item()
            if token_id == self.eos_token_id:
                print(flush=True)
                self.cache = None
                self.y = None
                return i, True

            text = self.sp.decode([token_id], special_token_policy=SpecialTokenPolicy.IGNORE)
            print(text, end="", flush=True)

            global_decode_pos = self.n_total_decoded + i
            if global_decode_pos > 0 and global_decode_pos % DEFAULT_CLEAR_CACHE_INTERVAL == 0:
                mx.clear_cache()

            self.y = next_y

        return n_to_decode, False

    def _final_flush(self):
        # Feed remaining audio + right padding through incremental pipeline,
        # then decode remaining positions.
        if self.cache is None or self.y is None:
            return

        self.pending_audio.append_many(self._drain_callback_chunks())
        right_pad = np.zeros(DEFAULT_RIGHT_PAD_TOKENS * SAMPLES_PER_TOKEN, dtype=np.float32)
        if len(self.pending_audio) > 0:
            flush_chunk = np.concatenate(
                [self.pending_audio.pop(len(self.pending_audio)), right_pad]
            )
        else:
            flush_chunk = right_pad

        self._encode_audio_chunk(flush_chunk)
        if len(self.audio_embeds) > 0:
            remaining = self.audio_embeds.pop(len(self.audio_embeds))
            self.decode_steps(remaining, remaining.shape[0])

    def _flush_last_pending_token(self):
        if self.y is None:
            return
        token_id = self.y.item()
        if token_id == self.eos_token_id:
            return
        text = self.sp.decode([token_id], special_token_policy=SpecialTokenPolicy.IGNORE)
        print(text, end="", flush=True)

    def run(self):
        print("Listening... (Ctrl+C to stop)\n", flush=True)
        stream = sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="float32",
            blocksize=SAMPLES_PER_TOKEN,
            callback=self.callback,
        )
        stream.start()

        try:
            start_time = time.monotonic()
            warned_no_audio = False
            while True:
                new_chunks = self._drain_callback_chunks()
                if new_chunks:
                    self.pending_audio.append_many(new_chunks)
                status_count, last_status = self._drain_audio_status()
                if status_count > 0:
                    print(
                        f"Warning: audio callback reported {status_count} status event(s)"
                        f" (last: {last_status}).",
                        flush=True,
                    )

                if self.first_cycle and len(self.pending_audio) < SAMPLES_PER_TOKEN:
                    elapsed = time.monotonic() - start_time
                    if not warned_no_audio and elapsed > 2.0:
                        warned_no_audio = True
                        print(
                            "Warning: No audio received. Check that your terminal app "
                            "has microphone permission in System Settings > Privacy & "
                            "Security > Microphone.",
                            flush=True,
                        )
                    time.sleep(0.02)
                    continue

                self._feed_available_audio()
                if len(self.audio_embeds) == 0:
                    time.sleep(0.02)
                    continue

                n_decodable = self._n_decodable()
                if n_decodable <= 0:
                    time.sleep(0.02)
                    continue

                if not self._maybe_prefill():
                    time.sleep(0.02)
                    continue

                n_decodable = self._n_decodable()
                if n_decodable <= 0:
                    time.sleep(0.02)
                    continue

                decode_embeds = self.audio_embeds.pop(n_decodable)
                n_consumed, hit_eos = self.decode_steps(decode_embeds, n_decodable)
                self.n_total_decoded += n_consumed
                if hit_eos:
                    self.reset_all_state()

        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()
            self._final_flush()
            self._flush_last_pending_token()
            print()


def stream_transcribe(
    model_path: str = "mlx-community/Voxtral-Mini-4B-Realtime-6bit",
    temperature: float = 0.0,
    sliding_window: int | None = None,
):
    model, sp, config = load_model(model_path)
    StreamingTranscriber(
        model=model,
        sp=sp,
        config=config,
        temperature=temperature,
        sliding_window=sliding_window,
    ).run()


def main():
    parser = argparse.ArgumentParser(
        description="Live streaming speech-to-text with Voxtral"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Voxtral-Mini-4B-Realtime-6bit",
        help="Model path or HF model ID",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=None,
        help=f"Decoder KV sliding window size (defaults to model config or {DEFAULT_DECODER_SLIDING_WINDOW})",
    )
    args = parser.parse_args()

    stream_transcribe(
        model_path=args.model,
        temperature=args.temp,
        sliding_window=args.sliding_window,
    )
