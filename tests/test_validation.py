"""Validation and masking edge-case guards for the CPU reference SDPA.

These lock the argument-checking `raise` paths and the two explicit-mask
behaviors (boolean keep-mask, additive-float mask) plus the fully-masked-row
convention. The exhaustive parametrized oracle sweep remains M0.2; this module
just keeps the error/edge branches covered.
"""

from __future__ import annotations

import numpy as np
import pytest

from portable_attention import scaled_dot_product_attention


def test_rejects_fewer_than_two_dims():
    v = np.zeros((3, 4))
    with pytest.raises(ValueError, match="at least 2 dims"):
        scaled_dot_product_attention(np.zeros(4), np.zeros((3, 4)), v)


def test_rejects_query_key_embedding_mismatch():
    q = np.zeros((5, 8))
    k = np.zeros((6, 7))
    v = np.zeros((6, 3))
    with pytest.raises(ValueError, match="embedding dims differ"):
        scaled_dot_product_attention(q, k, v)


def test_rejects_key_value_sequence_mismatch():
    q = np.zeros((5, 8))
    k = np.zeros((6, 8))
    v = np.zeros((7, 3))
    with pytest.raises(ValueError, match="sequence dims differ"):
        scaled_dot_product_attention(q, k, v)


def test_boolean_mask_excludes_false_positions():
    # A boolean mask that keeps only key 0 makes every output row equal v[0].
    rng = np.random.default_rng(10)
    q = rng.standard_normal((3, 8))
    k = rng.standard_normal((4, 8))
    v = rng.standard_normal((4, 3))
    mask = np.zeros((3, 4), dtype=bool)
    mask[:, 0] = True

    out = scaled_dot_product_attention(q, k, v, attn_mask=mask)

    np.testing.assert_allclose(out, np.broadcast_to(v[0], out.shape), atol=1e-12)


def test_additive_float_mask_matches_manual_bias():
    # An additive -inf bias on all-but-one column matches the bool-mask result.
    rng = np.random.default_rng(11)
    q = rng.standard_normal((3, 8))
    k = rng.standard_normal((4, 8))
    v = rng.standard_normal((4, 3))
    bias = np.full((3, 4), -np.inf)
    bias[:, 1] = 0.0

    out = scaled_dot_product_attention(q, k, v, attn_mask=bias)

    np.testing.assert_allclose(out, np.broadcast_to(v[1], out.shape), atol=1e-12)


def test_fully_masked_row_outputs_zero():
    # A row masked out entirely is 0/0; the convention defines its output as 0.
    q = np.zeros((2, 4))
    k = np.zeros((3, 4))
    v = np.ones((3, 5))
    mask = np.ones((2, 3), dtype=bool)
    mask[0, :] = False  # first query attends to nothing

    out = scaled_dot_product_attention(q, k, v, attn_mask=mask)

    np.testing.assert_allclose(out[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(out[1], 1.0, atol=1e-12)
