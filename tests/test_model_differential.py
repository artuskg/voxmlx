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
        repo_root = Path(__file__).resolve().parents[1]
        default_reference_mono = (
            repo_root
            / "perf"
            / "reference_audio"
            / "Paul_Solt_Ideating-and-developing-with-ChatGPT-Pro_mono_16k_10min.wav"
        )
        cls.reference_mono_path = Path(
            os.getenv("VOXMLX_TEST_REFERENCE_MONO_PATH", str(default_reference_mono))
        )

        if not cls.model_path.exists():
            raise unittest.SkipTest(f"Model path not found: {cls.model_path}")
        if not cls.audio_path.exists():
            raise unittest.SkipTest(f"Audio path not found: {cls.audio_path}")

        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
        from voxmlx import _build_prompt_tokens, load_model, transcribe
        from voxmlx.audio import (
            SAMPLE_RATE,
            SAMPLES_PER_TOKEN,
            load_audio,
            log_mel_spectrogram,
            log_mel_spectrogram_step,
            pad_audio,
        )
        from voxmlx.generate import generate

        cls.special_token_policy = SpecialTokenPolicy
        cls.build_prompt_tokens = _build_prompt_tokens
        cls.load_model = load_model
        cls.transcribe = transcribe
        cls.generate = generate
        cls.sample_rate = SAMPLE_RATE
        cls.samples_per_token = SAMPLES_PER_TOKEN
        cls.load_audio = staticmethod(load_audio)
        cls.log_mel_spectrogram = staticmethod(log_mel_spectrogram)
        cls.log_mel_spectrogram_step = staticmethod(log_mel_spectrogram_step)
        cls.pad_audio = staticmethod(pad_audio)

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

    def test_encode_contract_reference_mono_offline_vs_incremental(self):
        import mlx.core as mx

        if not self.reference_mono_path.exists():
            self.skipTest(f"Reference mono audio not found: {self.reference_mono_path}")

        max_seconds = float(os.getenv("VOXMLX_TEST_REFERENCE_MONO_MAX_SECONDS", "600"))
        if max_seconds <= 0:
            self.skipTest("VOXMLX_TEST_REFERENCE_MONO_MAX_SECONDS must be > 0")

        audio = self.load_audio(str(self.reference_mono_path))
        max_samples = int(max_seconds * self.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        audio = self.pad_audio(audio)

        mel_full = self.log_mel_spectrogram(audio)
        offline = self.model.encode(mel_full)

        audio_tail = None
        conv1_tail = None
        conv2_tail = None
        encoder_cache = None
        ds_buf = None
        parts = []
        for i in range(0, len(audio), self.samples_per_token):
            chunk = audio[i : i + self.samples_per_token]
            mel_step, audio_tail = self.log_mel_spectrogram_step(chunk, audio_tail)
            out, conv1_tail, conv2_tail, encoder_cache, ds_buf = self.model.encode_step(
                mel_step,
                conv1_tail,
                conv2_tail,
                encoder_cache,
                ds_buf,
            )
            if out is not None and out.shape[0] > 0:
                parts.append(out)

        hidden = int(offline.shape[1]) if offline.ndim == 2 else 0
        incremental = (
            mx.concatenate(parts, axis=0)
            if parts
            else mx.zeros((0, hidden), dtype=offline.dtype)
        )

        self.assertEqual(tuple(offline.shape), tuple(incremental.shape))
        if offline.shape[0] == 0:
            return

        off_f32 = offline.astype(mx.float32)
        inc_f32 = incremental.astype(mx.float32)
        abs_err = mx.max(mx.abs(off_f32 - inc_f32))
        rel_err = mx.max(mx.abs(off_f32 - inc_f32) / mx.maximum(mx.abs(off_f32), 1e-6))
        self.assertLessEqual(float(abs_err), 1e-4)
        self.assertLessEqual(float(rel_err), 1e-4)


if __name__ == "__main__":
    unittest.main()
