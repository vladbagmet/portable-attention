"""Correctness tests for the CPU reference SDPA.

M0.1 keeps this to a trivially-passing sanity check plus the load-bearing
invariants; the exhaustive shape/dtype/mask suite is M0.2.
"""

from __future__ import annotations

import numpy as np

from portable_attention import scaled_dot_product_attention


def _naive_sdpa(q, k, v, scale):
    """Independent, obviously-correct oracle (no max-subtraction trick)."""
    scores = (q @ np.swapaxes(k, -1, -2)) * scale
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ v


def test_output_shape_and_dtype():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((2, 3, 5, 8)).astype(np.float32)
    k = rng.standard_normal((2, 3, 7, 8)).astype(np.float32)
    v = rng.standard_normal((2, 3, 7, 4)).astype(np.float32)

    out = scaled_dot_product_attention(q, k, v)

    assert out.shape == (2, 3, 5, 4)
    assert out.dtype == np.float32


def test_matches_naive_oracle():
    rng = np.random.default_rng(1)
    q = rng.standard_normal((4, 8))
    k = rng.standard_normal((6, 8))
    v = rng.standard_normal((6, 3))
    scale = 1.0 / np.sqrt(8)

    out = scaled_dot_product_attention(q, k, v)
    expected = _naive_sdpa(q, k, v, scale)

    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-10)


def test_weights_form_convex_combination():
    # Output rows must lie in the convex hull of value rows: attention is a
    # weighted average, so each output coordinate is bounded by value extremes.
    rng = np.random.default_rng(2)
    q = rng.standard_normal((5, 8))
    k = rng.standard_normal((6, 8))
    v = rng.standard_normal((6, 3))

    out = scaled_dot_product_attention(q, k, v)

    assert np.all(out <= v.max(axis=0) + 1e-9)
    assert np.all(out >= v.min(axis=0) - 1e-9)


def test_is_causal_hides_the_future():
    # With a causal mask, position 0 attends only to key 0, so its output must
    # equal value row 0 exactly.
    rng = np.random.default_rng(3)
    q = rng.standard_normal((4, 8))
    k = rng.standard_normal((4, 8))
    v = rng.standard_normal((4, 3))

    out = scaled_dot_product_attention(q, k, v, is_causal=True)

    np.testing.assert_allclose(out[0], v[0], rtol=1e-12, atol=1e-12)


def test_causal_and_mask_are_mutually_exclusive():
    q = np.zeros((2, 4))
    k = np.zeros((2, 4))
    v = np.zeros((2, 4))
    mask = np.ones((2, 2), dtype=bool)

    try:
        scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)
    except ValueError:
        return
    raise AssertionError("expected ValueError for mask + is_causal")
