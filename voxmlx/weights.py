import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from .contracts import (
    is_conv_weight_name,
    is_converted_model_format,
    remap_weight_name,
)
from .model import VoxtralRealtime


def download_model(model_id: str = "mistralai/Voxtral-Mini-4B-Realtime-2602") -> Path:
    path = snapshot_download(
        model_id,
        allow_patterns=[
            "consolidated.safetensors",
            "model*.safetensors",
            "model.safetensors.index.json",
            "params.json",
            "config.json",
            "tekken.json",
        ],
    )
    return Path(path)


def _remap_name(name: str) -> str | None:
    return remap_weight_name(name)


def _is_conv_weight(name: str) -> bool:
    return is_conv_weight_name(name)


def _is_converted_format(model_path: Path) -> bool:
    """Check if the model is in voxmlx converted format (has config.json + model.safetensors)."""
    return is_converted_model_format(model_path)


def _load_converted(model_path: Path) -> tuple[VoxtralRealtime, dict]:
    """Load a model saved in voxmlx converted format."""
    with open(model_path / "config.json") as f:
        config = json.load(f)

    quant_config = config.get("quantization")
    model = VoxtralRealtime(config)

    # Apply quantization structure before loading weights
    if quant_config is not None:
        group_size = quant_config["group_size"]

        def predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            if module.weight.shape[-1] % group_size != 0:
                return False
            return True

        nn.quantize(
            model,
            group_size=group_size,
            bits=quant_config["bits"],
            class_predicate=predicate,
        )

    # Load weights — either single file or sharded
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        weights = {}
        for shard_file in shard_files:
            weights.update(mx.load(str(model_path / shard_file)))
    else:
        weights = mx.load(str(model_path / "model.safetensors"))

    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    return model, config


def _load_original(model_path: Path) -> tuple[VoxtralRealtime, dict]:
    """Load a model in the original Mistral format (consolidated.safetensors + params.json)."""
    with open(model_path / "params.json") as f:
        config = json.load(f)

    model = VoxtralRealtime(config)

    weights = mx.load(str(model_path / "consolidated.safetensors"))

    remapped = {}
    skipped = []
    for name, tensor in weights.items():
        if name == "output.weight":
            continue

        new_name = _remap_name(name)
        if new_name is None:
            skipped.append(name)
            continue

        # Transpose conv weights: PyTorch [C_out, C_in, K] -> MLX [C_out, K, C_in]
        if _is_conv_weight(new_name):
            tensor = mx.swapaxes(tensor, 1, 2)

        remapped[new_name] = tensor

    if skipped:
        print(f"Warning: skipped {len(skipped)} unrecognized weights: {skipped[:5]}...")

    model.load_weights(list(remapped.items()))
    mx.eval(model.parameters())

    return model, config


def load_model(model_path: str | Path) -> tuple[VoxtralRealtime, dict]:
    model_path = Path(model_path)
    if _is_converted_format(model_path):
        return _load_converted(model_path)
    return _load_original(model_path)
