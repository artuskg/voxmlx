import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

from voxmlx.contracts import (
    build_prompt_tokens,
    is_conv_weight_name,
    is_converted_model_format,
    make_weight_shards,
    remap_weight_name,
)


class _DummyWeight:
    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class _FakeTokenizer:
    bos_id = 1

    def get_special_token(self, token: str) -> int:
        if token != "[STREAMING_PAD]":
            raise ValueError(token)
        return 99


class ContractsTest(unittest.TestCase):
    def test_weight_remap_cases(self):
        cases_path = Path(__file__).parent / "fixtures" / "weight_remap_cases.json"
        cases = json.loads(cases_path.read_text())
        for case in cases:
            with self.subTest(name=case["input"]):
                self.assertEqual(remap_weight_name(case["input"]), case["expected"])

    def test_conv_weight_detection(self):
        self.assertTrue(is_conv_weight_name("encoder.conv1.weight"))
        self.assertTrue(is_conv_weight_name("encoder.conv2.weight"))
        self.assertFalse(is_conv_weight_name("encoder.conv1.bias"))
        self.assertFalse(is_conv_weight_name("language_model.layers.0.attention.q_proj.weight"))

    def test_converted_model_format_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)

            self.assertFalse(is_converted_model_format(path))

            (path / "config.json").write_text("{}")
            self.assertTrue(is_converted_model_format(path))

            (path / "consolidated.safetensors").write_text("x")
            self.assertFalse(is_converted_model_format(path))

    def test_make_weight_shards_keeps_order_and_no_empty_shards(self):
        weights = OrderedDict(
            [
                ("a", _DummyWeight(nbytes=10)),
                ("b", _DummyWeight(nbytes=20)),
                ("c", _DummyWeight(nbytes=30)),
            ]
        )

        # max_file_size_gb=0 -> every addition exceeds the limit after first item.
        shards = make_weight_shards(weights, max_file_size_gb=0)

        self.assertEqual(len(shards), 3)
        self.assertEqual(list(shards[0].keys()), ["a"])
        self.assertEqual(list(shards[1].keys()), ["b"])
        self.assertEqual(list(shards[2].keys()), ["c"])
        self.assertTrue(all(len(shard) > 0 for shard in shards))

    def test_make_weight_shards_empty_input(self):
        self.assertEqual(make_weight_shards({}, max_file_size_gb=1), [{}])

    def test_build_prompt_tokens_contract(self):
        tokens, n_delay = build_prompt_tokens(_FakeTokenizer(), n_left_pad_tokens=3, num_delay_tokens=2)
        self.assertEqual(n_delay, 2)
        self.assertEqual(tokens[0], 1)
        self.assertEqual(tokens[1:], [99, 99, 99, 99, 99])


if __name__ == "__main__":
    unittest.main()
