import importlib.util
import os
import unittest


class MlxRuntimeOptionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        enabled = os.getenv("VOXMLX_ENABLE_MLX_RUNTIME_TESTS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled:
            raise unittest.SkipTest(
                "Set VOXMLX_ENABLE_MLX_RUNTIME_TESTS=1 to run MLX runtime optional tests"
            )

        if importlib.util.find_spec("mlx") is None:
            raise unittest.SkipTest("mlx is required for runtime optional tests")

    def test_rotating_kv_cache_concat_respects_max_size(self):
        import mlx.core as mx

        from voxmlx.cache import RotatingKVCache

        cache = RotatingKVCache(max_size=8)

        for span in (3, 4, 5):
            keys = mx.arange(span, dtype=mx.float32).reshape(1, 1, span, 1)
            values = mx.arange(span, dtype=mx.float32).reshape(1, 1, span, 1)
            k, v = cache.update_and_fetch(keys, values)
            self.assertLessEqual(k.shape[2], 8)
            self.assertLessEqual(v.shape[2], 8)

    def test_offline_and_streaming_mel_match(self):
        import mlx.core as mx
        import numpy as np

        from voxmlx.audio import (
            SAMPLES_PER_TOKEN,
            log_mel_spectrogram,
            log_mel_spectrogram_step,
        )

        rng = np.random.default_rng(0)
        audio = rng.standard_normal(123456).astype(np.float32) * 0.01

        offline = log_mel_spectrogram(audio)

        tail = None
        chunks = []
        for i in range(0, len(audio), SAMPLES_PER_TOKEN):
            mel, tail = log_mel_spectrogram_step(audio[i : i + SAMPLES_PER_TOKEN], tail)
            chunks.append(mel)

        streaming = mx.concatenate(chunks, axis=1)
        self.assertEqual(tuple(offline.shape), tuple(streaming.shape))

        offline_np = np.array(offline)
        streaming_np = np.array(streaming)
        self.assertTrue(np.allclose(offline_np, streaming_np, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
