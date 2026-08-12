"""Tests for the blocked (flash-style) attention specification.

The single claim this module has to earn is that tiling changes nothing: for
any legal tile shape, the streamed online-softmax result must equal what the
reference oracle computes in one shot. Every test below is some corner of that
claim — trailing partial tiles, causal tile skipping, tile shapes down to 1x1,
and rows whose softmax denominator is zero.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from portable_attention.blocked import blocked_attention
from portable_attention.reference import scaled_dot_product_attention as reference_sdpa
from portable_attention.tiling import (
    V3D_LIMITS,
    DeviceLimits,
    TilePlan,
    plan_tiles,
)

# Sequence lengths deliberately off the tile grid: 1 is degenerate, 5 and 13 are
# prime, 17 = 16 + 1 forces a one-row trailing tile behind a full one.
SEQ_LENS = [1, 5, 13, 16, 17]
HEAD_DIMS = [8, 64]


def make_inputs(stack, seq_q, seq_k, head_dim, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    shape_q = (stack, seq_q, head_dim)
    shape_kv = (stack, seq_k, head_dim)
    return (
        rng.standard_normal(shape_q).astype(dtype),
        rng.standard_normal(shape_kv).astype(dtype),
        rng.standard_normal(shape_kv).astype(dtype),
    )


def tiny_plan(head_dim, dtype_bytes=4, block_q=1, block_k=1):
    """A hand-built plan, for tile shapes the policy would never choose."""
    return TilePlan(
        block_q=block_q,
        block_k=block_k,
        threads_per_group=block_q * block_k,
        shared_memory_bytes=0,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
        limits=V3D_LIMITS,
    )


@pytest.mark.parametrize(
    ("seq_q", "seq_k", "head_dim"),
    [(q, k, e) for q, k, e in itertools.product(SEQ_LENS, SEQ_LENS, HEAD_DIMS)],
)
def test_matches_reference_across_shapes(seq_q, seq_k, head_dim):
    query, key, value = make_inputs(2, seq_q, seq_k, head_dim)
    plan = plan_tiles(head_dim, 4, V3D_LIMITS)
    out = blocked_attention(query, key, value, plan)
    expected = reference_sdpa(query, key, value)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
    assert out.shape == query.shape
    assert out.dtype == query.dtype


@pytest.mark.parametrize("seq", SEQ_LENS)
def test_matches_reference_when_causal(seq):
    query, key, value = make_inputs(2, seq, seq, 16, seed=1)
    plan = plan_tiles(16, 4, V3D_LIMITS)
    out = blocked_attention(query, key, value, plan, is_causal=True)
    expected = reference_sdpa(query, key, value, is_causal=True)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_causal_skips_tiles_beyond_the_diagonal():
    # A key sequence far longer than the queries: everything past the last
    # query row is masked, so the skip has to leave the answer untouched.
    query, key, value = make_inputs(1, 3, 96, 16, seed=2)
    plan = plan_tiles(16, 4, V3D_LIMITS, seq_len_q=3)
    out = blocked_attention(query, key, value, plan, is_causal=True)
    expected = reference_sdpa(query, key, value, is_causal=True)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("block_q", "block_k"), [(1, 1), (1, 8), (8, 1), (2, 4), (32, 32)]
)
def test_result_is_independent_of_tile_shape(block_q, block_k):
    query, key, value = make_inputs(2, 17, 21, 8, seed=3)
    out = blocked_attention(
        query, key, value, tiny_plan(8, block_q=block_q, block_k=block_k)
    )
    expected = reference_sdpa(query, key, value)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_scale_is_honoured():
    query, key, value = make_inputs(1, 6, 6, 8, seed=4)
    plan = plan_tiles(8, 4, V3D_LIMITS)
    out = blocked_attention(query, key, value, plan, scale=0.05)
    expected = reference_sdpa(query, key, value, scale=0.05)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
    # The default is 1/sqrt(E), so a different scale must give a different answer.
    assert not np.allclose(out, blocked_attention(query, key, value, plan))


def test_float64_plan_computes_in_float64():
    query, key, value = make_inputs(1, 9, 9, 8, dtype=np.float64, seed=5)
    plan = plan_tiles(8, 8, V3D_LIMITS)
    out = blocked_attention(query, key, value, plan)
    expected = reference_sdpa(query, key, value)
    assert out.dtype == np.float64
    # float64 in, float64 through: this should be near machine precision, not
    # the loose float32 tolerance the other tests use.
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_large_scores_do_not_overflow():
    # Scores near the float32 exponent limit: the online rescale must subtract
    # the running max before exponentiating, or these rows come back as nan.
    query, key, value = make_inputs(1, 12, 40, 16, seed=6)
    query = (query * 40.0).astype(np.float32)
    key = (key * 40.0).astype(np.float32)
    plan = plan_tiles(16, 4, V3D_LIMITS)
    out = blocked_attention(query, key, value, plan)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(
        out, reference_sdpa(query, key, value), rtol=1e-4, atol=1e-4
    )


def test_padded_lanes_do_not_leak_into_the_result():
    # seq_q = 1 with block_q = 16 means 15 of the 16 query rows in the tile are
    # padding. Their scores must be masked out, not merely small.
    query, key, value = make_inputs(1, 1, 1, 8, seed=7)
    out = blocked_attention(query, key, value, tiny_plan(8, block_q=16, block_k=16))
    np.testing.assert_allclose(
        out, reference_sdpa(query, key, value), rtol=1e-6, atol=1e-6
    )


def test_stack_entries_are_independent():
    query, key, value = make_inputs(3, 10, 12, 8, seed=8)
    plan = plan_tiles(8, 4, V3D_LIMITS)
    out = blocked_attention(query, key, value, plan)
    for n in range(3):
        single = blocked_attention(
            query[n : n + 1], key[n : n + 1], value[n : n + 1], plan
        )
        np.testing.assert_array_equal(out[n : n + 1], single)


def test_zero_length_key_sequence_yields_exact_zeros():
    # No keys at all: every row's softmax denominator is zero and the contract
    # (shared with the reference backend) makes the output exactly zero.
    query, key, value = make_inputs(1, 4, 0, 8, seed=9)
    plan = plan_tiles(8, 4, V3D_LIMITS)
    out = blocked_attention(query, key, value, plan)
    np.testing.assert_array_equal(out, np.zeros_like(out))


@pytest.mark.parametrize("bad", ["query", "key", "value"])
def test_inputs_must_be_three_dimensional(bad):
    arrays = dict(zip(("query", "key", "value"), make_inputs(1, 4, 4, 8)))
    arrays[bad] = arrays[bad][0]
    with pytest.raises(ValueError, match="shape \\(n, seq, head_dim\\)"):
        blocked_attention(**arrays, plan=plan_tiles(8, 4, V3D_LIMITS))


def test_stack_dimensions_must_agree():
    query, key, value = make_inputs(2, 4, 4, 8)
    with pytest.raises(ValueError, match="stack dimension"):
        blocked_attention(query, key[:1], value[:1], plan_tiles(8, 4, V3D_LIMITS))


def test_key_and_value_sequence_lengths_must_agree():
    query, key, value = make_inputs(1, 4, 6, 8)
    with pytest.raises(ValueError, match="sequence dims differ"):
        blocked_attention(query, key, value[:, :4], plan_tiles(8, 4, V3D_LIMITS))


def test_one_head_dim_across_all_three_inputs():
    query, key, value = make_inputs(1, 4, 4, 8)
    with pytest.raises(ValueError, match="one head dim"):
        blocked_attention(query, key, value[..., :4], plan_tiles(8, 4, V3D_LIMITS))


def test_plan_head_dim_must_match_the_inputs():
    query, key, value = make_inputs(1, 4, 4, 8)
    with pytest.raises(ValueError, match="sized for head_dim=16"):
        blocked_attention(query, key, value, plan_tiles(16, 4, V3D_LIMITS))


def test_plan_element_size_must_name_a_compute_dtype():
    query, key, value = make_inputs(1, 4, 4, 8, dtype=np.float16)
    # float16 is a legal tile element size, but this host model has no half
    # precision compute path; it must say so rather than silently upcast.
    plan = plan_tiles(8, 2, DeviceLimits(16384, 256, 16, name="half-precision"))
    with pytest.raises(ValueError, match="no compute dtype"):
        blocked_attention(query, key, value, plan)
