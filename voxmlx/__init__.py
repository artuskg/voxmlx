from __future__ import annotations

__version__ = "0.0.2"

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import build_prompt_tokens as _build_prompt_tokens_contract

if TYPE_CHECKING:
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer


def _load_tokenizer(model_path: Path) -> Tekkenizer:
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer

    tekken_path = model_path / "tekken.json"
    return Tekkenizer.from_file(str(tekken_path))


def _build_prompt_tokens(
    sp: Tekkenizer,
    n_left_pad_tokens: int = 32,
    num_delay_tokens: int = 6,
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
    sp = _load_tokenizer(model_path)
    return model, sp, config


def transcribe(
    audio_path: str,
    model_path: str = "mlx-community/Voxtral-Mini-4B-Realtime-6bit",
    temperature: float = 0.0,
) -> str:
    from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

    from .generate import generate

    model, sp, config = load_model(model_path)

    prompt_tokens, n_delay_tokens = _build_prompt_tokens(sp)

    output_tokens = generate(
        model,
        audio_path,
        prompt_tokens,
        n_delay_tokens=n_delay_tokens,
        temperature=temperature,
        eos_token_id=sp.eos_id,
    )

    return sp.decode(output_tokens, special_token_policy=SpecialTokenPolicy.IGNORE)


def main():
    parser = argparse.ArgumentParser(description="Voxtral Mini Realtime speech-to-text")
    parser.add_argument("--audio", default=None, help="Path to audio file (omit to stream from mic)")
    parser.add_argument("--model", default="mlx-community/Voxtral-Mini-4B-Realtime-6bit", help="Model path or HF model ID")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    args = parser.parse_args()

    if args.audio is not None:
        text = transcribe(
            args.audio,
            model_path=args.model,
            temperature=args.temp,
        )
        print(text)
    else:
        from .stream import stream_transcribe

        stream_transcribe(
            model_path=args.model,
            temperature=args.temp,
        )
