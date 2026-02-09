import importlib.util
import os
import unittest


class KVCacheRopeScaffoldOptionalTests(unittest.TestCase):
    """Scaffold tests for future ring-buffer KV cache replacement.

    These tests intentionally lock external cache+RoPE behavior and should remain
    valid when swapping in a new cache implementation.
    """

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

    @staticmethod
    def _mk_seq(mx, total: int, n_heads: int = 2, head_dim: int = 8):
        vals = mx.arange(total * n_heads * head_dim, dtype=mx.float32)
        return vals.reshape(1, n_heads, total, head_dim)

    @staticmethod
    def _assert_allclose(np, a, b, atol=1e-5, rtol=1e-5):
        if not np.allclose(np.array(a), np.array(b), atol=atol, rtol=rtol):
            raise AssertionError("arrays are not close")

    @staticmethod
    def _sorted_by_token_id(np, x):
        arr = np.array(x)
        order = np.argsort(arr[0, 0, :, 0])
        return np.take(arr, order, axis=2)

    def test_cache_tail_matches_reference_for_mixed_updates(self):
        import mlx.core as mx
        import numpy as np

        from voxmlx.cache import RotatingKVCache

        max_size = 8
        spans = [3, 1, 4, 2, 5, 1]
        total = sum(spans)

        keys_all = self._mk_seq(mx, total)
        vals_all = self._mk_seq(mx, total) + 10_000.0
        cache = RotatingKVCache(max_size=max_size)

        cursor = 0
        for span in spans:
            k = keys_all[..., cursor : cursor + span, :]
            v = vals_all[..., cursor : cursor + span, :]
            out_k, out_v = cache.update_and_fetch(k, v)
            cursor += span

            start = max(0, cursor - max_size)
            ref_k = keys_all[..., start:cursor, :]
            ref_v = vals_all[..., start:cursor, :]

            # Mixed single-token/chunk updates may return ring order on the
            # in-place path. Verify set-equivalence of cached pairs.
            self._assert_allclose(
                np,
                self._sorted_by_token_id(np, out_k),
                self._sorted_by_token_id(np, ref_k),
            )
            self._assert_allclose(
                np,
                self._sorted_by_token_id(np, out_v),
                self._sorted_by_token_id(np, ref_v),
            )
            self.assertEqual(cache.offset, cursor)

    def test_concat_only_updates_return_temporal_tail(self):
        import mlx.core as mx
        import numpy as np

        from voxmlx.cache import RotatingKVCache

        max_size = 8
        spans = [3, 4, 5, 6]
        total = sum(spans)

        keys_all = self._mk_seq(mx, total)
        vals_all = self._mk_seq(mx, total) + 15_000.0
        cache = RotatingKVCache(max_size=max_size)

        cursor = 0
        for span in spans:
            k = keys_all[..., cursor : cursor + span, :]
            v = vals_all[..., cursor : cursor + span, :]
            out_k, out_v = cache.update_and_fetch(k, v)
            cursor += span

            start = max(0, cursor - max_size)
            ref_k = keys_all[..., start:cursor, :]
            ref_v = vals_all[..., start:cursor, :]
            self._assert_allclose(np, out_k, ref_k)
            self._assert_allclose(np, out_v, ref_v)
            self.assertEqual(cache.offset, cursor)

    def test_concat_update_equivalent_to_tokenwise_updates(self):
        import mlx.core as mx
        import numpy as np

        from voxmlx.cache import RotatingKVCache

        max_size = 6
        spans = [2, 5, 6]
        total = sum(spans)

        keys_all = self._mk_seq(mx, total)
        vals_all = self._mk_seq(mx, total) + 20_000.0

        cache_chunked = RotatingKVCache(max_size=max_size)
        cache_tokenwise = RotatingKVCache(max_size=max_size)

        cursor = 0
        for span in spans:
            k_chunk = keys_all[..., cursor : cursor + span, :]
            v_chunk = vals_all[..., cursor : cursor + span, :]
            out_k_chunk, out_v_chunk = cache_chunked.update_and_fetch(k_chunk, v_chunk)

            for _ in range(span):
                k_tok = keys_all[..., cursor : cursor + 1, :]
                v_tok = vals_all[..., cursor : cursor + 1, :]
                out_k_tok, out_v_tok = cache_tokenwise.update_and_fetch(k_tok, v_tok)
                cursor += 1

            # Compare as sets (sorted by first channel) because tokenwise path
            # may return ring order while chunked path returns temporal order.
            self._assert_allclose(
                np,
                self._sorted_by_token_id(np, out_k_chunk),
                self._sorted_by_token_id(np, out_k_tok),
            )
            self._assert_allclose(
                np,
                self._sorted_by_token_id(np, out_v_chunk),
                self._sorted_by_token_id(np, out_v_tok),
            )

    def test_rope_chunk_offsets_match_full_rope(self):
        import mlx.core as mx
        import numpy as np

        total = 17
        spans = [1, 4, 3, 2, 7]
        keys_all = self._mk_seq(mx, total)

        full = mx.fast.rope(
            keys_all, keys_all.shape[-1], traditional=True, base=1e6, scale=1.0, offset=0
        )

        parts = []
        cursor = 0
        for span in spans:
            chunk = keys_all[..., cursor : cursor + span, :]
            roped = mx.fast.rope(
                chunk, chunk.shape[-1], traditional=True, base=1e6, scale=1.0, offset=cursor
            )
            parts.append(roped)
            cursor += span

        chunked = mx.concatenate(parts, axis=2)
        self._assert_allclose(np, chunked, full)

    def test_decode_attention_with_cache_matches_reference_tail(self):
        import mlx.core as mx
        import numpy as np

        from voxmlx.cache import RotatingKVCache

        max_size = 7
        total = 20
        n_heads = 2
        head_dim = 8
        scale = head_dim ** -0.5

        k_base = self._mk_seq(mx, total, n_heads=n_heads, head_dim=head_dim)
        v_all = self._mk_seq(mx, total, n_heads=n_heads, head_dim=head_dim) + 30_000.0
        q_base = self._mk_seq(mx, total, n_heads=n_heads, head_dim=head_dim) + 40_000.0

        k_all_roped = mx.fast.rope(
            k_base, head_dim, traditional=True, base=1e6, scale=1.0, offset=0
        )

        cache = RotatingKVCache(max_size=max_size)
        for t in range(total):
            k_t = k_base[..., t : t + 1, :]
            v_t = v_all[..., t : t + 1, :]
            q_t = q_base[..., t : t + 1, :]

            k_t_roped = mx.fast.rope(
                k_t, head_dim, traditional=True, base=1e6, scale=1.0, offset=t
            )
            q_t_roped = mx.fast.rope(
                q_t, head_dim, traditional=True, base=1e6, scale=1.0, offset=t
            )

            k_cache, v_cache = cache.update_and_fetch(k_t_roped, v_t)
            out_cache = mx.fast.scaled_dot_product_attention(
                q_t_roped, k_cache, v_cache, scale=scale, mask=None
            )

            start = max(0, t + 1 - max_size)
            k_ref = k_all_roped[..., start : t + 1, :]
            v_ref = v_all[..., start : t + 1, :]
            out_ref = mx.fast.scaled_dot_product_attention(
                q_t_roped, k_ref, v_ref, scale=scale, mask=None
            )

            self._assert_allclose(np, out_cache, out_ref, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
