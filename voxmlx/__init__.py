from __future__ import annotations

__version__ = "0.0.2"

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .constants import DEFAULT_DELAY_TOKENS, DEFAULT_DECODER_SLIDING_WINDOW, DEFAULT_LEFT_PAD_TOKENS
from .contracts import build_prompt_tokens as _build_prompt_tokens_contract

if TYPE_CHECKING:
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer


LoadedModelBundle = tuple[Any, Any, dict]
_EXPECTED_DOWNSAMPLE_FACTOR = 4


def _load_tokenizer(model_path: Path) -> Tekkenizer:
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer

    tekken_path = model_path / "tekken.json"
    return Tekkenizer.from_file(str(tekken_path))


def _build_prompt_tokens(
    sp: Tekkenizer,
    n_left_pad_tokens: int = DEFAULT_LEFT_PAD_TOKENS,
    num_delay_tokens: int = DEFAULT_DELAY_TOKENS,
) -> tuple[list[int], int]:
    return _build_prompt_tokens_contract(
        tokenizer=sp,
        n_left_pad_tokens=n_left_pad_tokens,
        num_delay_tokens=num_delay_tokens,
    )


def load_model(model_path: str = "mlx-community/Voxtral-Mini-4B-Realtime-6bit"):
    from .weights import download_model, load_model as _load_weights

    if not Path(model_path).exists():
        model_path = download_model(model_path)
    else:
        model_path = Path(model_path)

    model, config = _load_weights(model_path)
    _validate_model_config(config)
    sp = _load_tokenizer(model_path)
    return model, sp, config


def _validate_model_config(config: dict):
    """Fail fast when config values would invalidate runtime token timing assumptions."""
    downsample_factor = (
        config.get("multimodal", {})
        .get("whisper_model_args", {})
        .get("downsample_args", {})
        .get("downsample_factor")
    )
    if downsample_factor is None:
        return
    if int(downsample_factor) != _EXPECTED_DOWNSAMPLE_FACTOR:
        raise ValueError(
            "Unsupported config downsample_factor="
            f"{downsample_factor}. Expected {_EXPECTED_DOWNSAMPLE_FACTOR} for "
            "the current streaming/token alignment constants."
        )


class Transcriber:
    """Reusable transcriber that avoids reloading model/tokenizer per call."""

    def __init__(self, model: Any, tokenizer: Any, config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.prompt_tokens, self.n_delay_tokens = _build_prompt_tokens(self.tokenizer)

    @classmethod
    def from_model_path(cls, model_path: str = "mlx-community/Voxtral-Mini-4B-Realtime-6bit"):
        model, tokenizer, config = load_model(model_path)
        return cls(model, tokenizer, config)

    def transcribe(
        self,
        audio_path: str,
        temperature: float = 0.0,
        sliding_window: int | None = None,
    ) -> str:
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

        from .generate import generate

        if sliding_window is None:
            sliding_window = int(
                self.config.get("sliding_window", DEFAULT_DECODER_SLIDING_WINDOW)
            )

        output_tokens = generate(
            self.model,
            audio_path,
            self.prompt_tokens,
            n_delay_tokens=self.n_delay_tokens,
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_id,
            sliding_window=sliding_window,
        )
        return self.tokenizer.decode(
            output_tokens,
            special_token_policy=SpecialTokenPolicy.IGNORE,
        )


def transcribe(
    audio_path: str,
    model_path: str = "mlx-community/Voxtral-Mini-4B-Realtime-6bit",
    temperature: float = 0.0,
    sliding_window: int | None = None,
    model_bundle: LoadedModelBundle | None = None,
) -> str:
    if model_bundle is None:
        model, sp, config = load_model(model_path)
    else:
        model, sp, config = model_bundle

    return Transcriber(model, sp, config).transcribe(
        audio_path,
        temperature=temperature,
        sliding_window=sliding_window,
    )


def main():
    parser = argparse.ArgumentParser(description="Voxtral Mini Realtime speech-to-text")
    parser.add_argument("--audio", default=None, help="Path to audio file (omit to stream from mic)")
    parser.add_argument("--model", default="mlx-community/Voxtral-Mini-4B-Realtime-6bit", help="Model path or HF model ID")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=None,
        help=f"Decoder KV sliding window size (defaults to model config or {DEFAULT_DECODER_SLIDING_WINDOW})",
    )
    args = parser.parse_args()

    if args.audio is not None:
        text = transcribe(
            args.audio,
            model_path=args.model,
            temperature=args.temp,
            sliding_window=args.sliding_window,
        )
        print(text)
    else:
        from .stream import stream_transcribe

        stream_transcribe(
            model_path=args.model,
            temperature=args.temp,
            sliding_window=args.sliding_window,
        )
