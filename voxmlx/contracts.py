"""Pure contract helpers used by runtime modules and tests."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Weight name remapping patterns: (regex, replacement)
REMAP_PATTERNS = [
    # Encoder conv layers
    (r"whisper_encoder\.conv_layers\.0\.conv\.(.*)", r"encoder.conv1.\1"),
    (r"whisper_encoder\.conv_layers\.1\.conv\.(.*)", r"encoder.conv2.\1"),
    # Encoder transformer layers
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wq\.(.*)", r"encoder.layers.\1.attention.q_proj.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wk\.(.*)", r"encoder.layers.\1.attention.k_proj.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wv\.(.*)", r"encoder.layers.\1.attention.v_proj.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wo\.(.*)", r"encoder.layers.\1.attention.o_proj.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.attention_norm\.(.*)", r"encoder.layers.\1.attn_norm.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w1\.(.*)", r"encoder.layers.\1.mlp.gate_proj.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w2\.(.*)", r"encoder.layers.\1.mlp.down_proj.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w3\.(.*)", r"encoder.layers.\1.mlp.up_proj.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.ffn_norm\.(.*)", r"encoder.layers.\1.ffn_norm.\2"),
    (r"whisper_encoder\.transformer\.norm\.(.*)", r"encoder.norm.\1"),
    # Adapter
    (r"audio_language_projection\.0\.weight", r"adapter.w_in.weight"),
    (r"audio_language_projection\.2\.weight", r"adapter.w_out.weight"),
    # Language model embedding
    (r"tok_embeddings\.weight", r"language_model.embed_tokens.weight"),
    # Language model layers
    (r"layers\.(\d+)\.attention\.wq\.weight", r"language_model.layers.\1.attention.q_proj.weight"),
    (r"layers\.(\d+)\.attention\.wk\.weight", r"language_model.layers.\1.attention.k_proj.weight"),
    (r"layers\.(\d+)\.attention\.wv\.weight", r"language_model.layers.\1.attention.v_proj.weight"),
    (r"layers\.(\d+)\.attention\.wo\.weight", r"language_model.layers.\1.attention.o_proj.weight"),
    (r"layers\.(\d+)\.attention_norm\.weight", r"language_model.layers.\1.attn_norm.weight"),
    (r"layers\.(\d+)\.feed_forward\.w1\.weight", r"language_model.layers.\1.mlp.gate_proj.weight"),
    (r"layers\.(\d+)\.feed_forward\.w2\.weight", r"language_model.layers.\1.mlp.down_proj.weight"),
    (r"layers\.(\d+)\.feed_forward\.w3\.weight", r"language_model.layers.\1.mlp.up_proj.weight"),
    (r"layers\.(\d+)\.ffn_norm\.weight", r"language_model.layers.\1.ffn_norm.weight"),
    (r"layers\.(\d+)\.ada_rms_norm_t_cond\.0\.weight", r"language_model.layers.\1.ada_norm.linear_in.weight"),
    (r"layers\.(\d+)\.ada_rms_norm_t_cond\.2\.weight", r"language_model.layers.\1.ada_norm.linear_out.weight"),
    # Language model output norm
    (r"norm\.weight", r"language_model.norm.weight"),
]


def remap_weight_name(name: str) -> str | None:
    """Remap source checkpoint names to voxmlx model names."""
    name = re.sub(r"^(mm_streams_embeddings\.embedding_module|mm_whisper_embeddings)\.", "", name)
    for pattern, replacement in REMAP_PATTERNS:
        new_name, n = re.subn(f"^{pattern}$", replacement, name)
        if n > 0:
            return new_name
    return None


def is_conv_weight_name(name: str) -> bool:
    return ("conv1.weight" in name or "conv2.weight" in name) and "bias" not in name


def is_converted_model_format(model_path: Path) -> bool:
    """Check if path is voxmlx-converted format and not original source layout."""
    return (model_path / "config.json").exists() and not (model_path / "consolidated.safetensors").exists()


def make_weight_shards(weights: dict[str, Any], max_file_size_gb: int = 5) -> list[dict[str, Any]]:
    """Split ordered weight dict into size-bounded shards.

    A single overweight tensor is allowed to occupy a shard by itself.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards: list[dict[str, Any]] = []
    shard: dict[str, Any] = {}
    shard_size = 0

    for key, value in weights.items():
        nbytes = getattr(value, "nbytes", None)
        if nbytes is None:
            raise TypeError(f"Weight {key!r} has no 'nbytes' attribute")

        # Only flush if we already have content; avoids creating an empty leading shard.
        if shard and shard_size + nbytes > max_file_size_bytes:
            shards.append(shard)
            shard = {}
            shard_size = 0

        shard[key] = value
        shard_size += nbytes

    if shard:
        shards.append(shard)

    # Keep previous empty-input behavior compatible with callers.
    return shards if shards else [{}]


def build_prompt_tokens(
    tokenizer: Any,
    n_left_pad_tokens: int = 32,
    num_delay_tokens: int = 6,
) -> tuple[list[int], int]:
    """Build the streaming prefix tokens contract used by transcription."""
    streaming_pad = tokenizer.get_special_token("[STREAMING_PAD]")
    prefix_len = n_left_pad_tokens + num_delay_tokens
    tokens = [tokenizer.bos_id] + [streaming_pad] * prefix_len
    return tokens, num_delay_tokens
