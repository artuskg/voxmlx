import importlib.util
import os
import unittest
from pathlib import Path


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class ModelDifferentialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _env_truthy("VOXMLX_ENABLE_MODEL_TESTS"):
            raise unittest.SkipTest("Set VOXMLX_ENABLE_MODEL_TESTS=1 to run model-backed differential tests")

        missing = [
            module
            for module in ("mlx", "mistral_common", "soundfile")
            if importlib.util.find_spec(module) is None
        ]
        if missing:
            raise unittest.SkipTest(f"Missing optional dependencies: {', '.join(missing)}")

        model_path = os.getenv("VOXMLX_TEST_MODEL_PATH")
        audio_path = os.getenv("VOXMLX_TEST_AUDIO_PATH")
        if not model_path or not audio_path:
            raise unittest.SkipTest(
                "Set VOXMLX_TEST_MODEL_PATH and VOXMLX_TEST_AUDIO_PATH to run differential tests"
            )

        cls.model_path = Path(model_path)
        cls.audio_path = Path(audio_path)

        if not cls.model_path.exists():
            raise unittest.SkipTest(f"Model path not found: {cls.model_path}")
        if not cls.audio_path.exists():
            raise unittest.SkipTest(f"Audio path not found: {cls.audio_path}")

        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
        from voxmlx import _build_prompt_tokens, load_model, transcribe
        from voxmlx.generate import generate

        cls.special_token_policy = SpecialTokenPolicy
        cls.build_prompt_tokens = _build_prompt_tokens
        cls.load_model = load_model
        cls.transcribe = transcribe
        cls.generate = generate

        cls.model, cls.tokenizer, _ = cls.load_model(str(cls.model_path))
        cls.prompt_tokens, cls.n_delay_tokens = cls.build_prompt_tokens(cls.tokenizer)

    def _generate_text(self, model, tokenizer, prompt_tokens, n_delay_tokens) -> str:
        token_ids = self.generate(
            model,
            str(self.audio_path),
            prompt_tokens,
            n_delay_tokens=n_delay_tokens,
            temperature=0.0,
            eos_token_id=tokenizer.eos_id,
        )
        return tokenizer.decode(
            token_ids,
            special_token_policy=self.special_token_policy.IGNORE,
        )

    def test_public_transcribe_matches_manual_pipeline(self):
        text_api = self.transcribe(
            str(self.audio_path),
            model_path=str(self.model_path),
            temperature=0.0,
        )
        text_manual = self._generate_text(
            self.model,
            self.tokenizer,
            self.prompt_tokens,
            self.n_delay_tokens,
        )
        self.assertEqual(text_api, text_manual)

    def test_generation_is_deterministic_at_temperature_zero(self):
        text_a = self._generate_text(
            self.model,
            self.tokenizer,
            self.prompt_tokens,
            self.n_delay_tokens,
        )
        text_b = self._generate_text(
            self.model,
            self.tokenizer,
            self.prompt_tokens,
            self.n_delay_tokens,
        )
        self.assertEqual(text_a, text_b)

    def test_original_and_converted_formats_match_when_both_paths_provided(self):
        original = os.getenv("VOXMLX_TEST_MODEL_PATH_ORIGINAL")
        converted = os.getenv("VOXMLX_TEST_MODEL_PATH_CONVERTED")
        if not original or not converted:
            self.skipTest(
                "Set VOXMLX_TEST_MODEL_PATH_ORIGINAL and VOXMLX_TEST_MODEL_PATH_CONVERTED to compare both formats"
            )

        original_path = Path(original)
        converted_path = Path(converted)
        if not original_path.exists() or not converted_path.exists():
            self.skipTest("Original/converted model path does not exist")

        model_original, tokenizer_original, _ = self.load_model(str(original_path))
        model_converted, tokenizer_converted, _ = self.load_model(str(converted_path))
        prompt_original, n_delay_original = self.build_prompt_tokens(tokenizer_original)
        prompt_converted, n_delay_converted = self.build_prompt_tokens(tokenizer_converted)

        text_original = self._generate_text(
            model_original,
            tokenizer_original,
            prompt_original,
            n_delay_original,
        )
        text_converted = self._generate_text(
            model_converted,
            tokenizer_converted,
            prompt_converted,
            n_delay_converted,
        )

        self.assertEqual(text_original, text_converted)


if __name__ == "__main__":
    unittest.main()
