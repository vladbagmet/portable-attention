"""Correctness suite for the CPU reference backward pass (VJP).

The analytic gradients are checked against central finite differences of the
forward oracle: for each input element, ``d/dx sum(g * sdpa(...))`` is measured
numerically and compared with the corresponding entry of ``dq``/``dk``/``dv``.
That makes the test independent of the derivation in ``backward.py`` — it only
trusts the forward pass, which has its own suite.

Coverage matrix mirrors the forward one: leading dims (none, batch, batch+head),
``L != S`` and ``E != Ev``, default and explicit ``scale``, causal / boolean /
additive masking, fully-masked rows, grouped-query attention, broadcast
key/value, and dtype preservation.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from portable_attention import scaled_dot_product_attention
from portable_attention.backward import scaled_dot_product_attention_backward

Array = NDArray[np.floating]
F64 = NDArray[np.float64]

# Central differences on a float64 forward pass: the truncation error is
# O(eps^2) and the round-off O(1e-16/eps), which balance near 1e-5.
_EPS = 1e-5


def _numeric_grad(
    forward: Callable[[F64], F64], x: F64, grad_output: F64, eps: float = _EPS
) -> F64:
    """Central-difference gradient of ``sum(grad_output * forward(x))``."""
    grad = np.zeros_like(x)
    flat = x.reshape(-1)
    for index in range(flat.size):
        original = flat[index]
        probe = x.copy()
        probe.reshape(-1)[index] = original + eps
        plus = float(np.sum(grad_output * forward(probe)))
        probe.reshape(-1)[index] = original - eps
        minus = float(np.sum(grad_output * forward(probe)))
        grad.reshape(-1)[index] = (plus - minus) / (2.0 * eps)
    return grad


def _random(rng: np.random.Generator, *shape: int) -> F64:
    return rng.standard_normal(shape)


def _check_against_finite_differences(
    query: F64,
    key: F64,
    value: F64,
    grad_output: F64,
    **kwargs: object,
) -> None:
    """Assert the analytic (dq, dk, dv) match finite differences."""
    forward = scaled_dot_product_attention

    def call(q: F64, k: F64, v: F64) -> F64:
        out: F64 = forward(q, k, v, **kwargs)  # type: ignore[arg-type]
        return out

    dq, dk, dv = scaled_dot_product_attention_backward(
        grad_output, query, key, value, **kwargs
    )
    expected_dq = _numeric_grad(lambda q: call(q, key, value), query, grad_output)
    expected_dk = _numeric_grad(lambda k: call(query, k, value), key, grad_output)
    expected_dv = _numeric_grad(lambda v: call(query, key, v), value, grad_output)

    np.testing.assert_allclose(dq, expected_dq, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(dk, expected_dk, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(dv, expected_dv, rtol=1e-5, atol=1e-8)


def test_signature_mirrors_the_forward_pass() -> None:
    """Same argument list, prefixed by the upstream gradient, minus dropout."""
    forward = list(inspect.signature(scaled_dot_product_attention).parameters)
    backward = list(inspect.signature(scaled_dot_product_attention_backward).parameters)
    assert backward == ["grad_output", *(p for p in forward if p != "dropout_p")]


@pytest.mark.parametrize(
    ("lead", "length", "source", "embed", "value_embed"),
    [
        ((), 3, 4, 3, 3),
        ((2,), 4, 4, 3, 2),
        ((2, 3), 3, 5, 4, 3),
    ],
)
def test_gradients_match_finite_differences(
    lead: tuple[int, ...], length: int, source: int, embed: int, value_embed: int
) -> None:
    rng = np.random.default_rng(0)
    query = _random(rng, *lead, length, embed)
    key = _random(rng, *lead, source, embed)
    value = _random(rng, *lead, source, value_embed)
    grad_output = _random(rng, *lead, length, value_embed)
    _check_against_finite_differences(query, key, value, grad_output)


def test_gradients_with_explicit_scale() -> None:
    rng = np.random.default_rng(1)
    query, key, value = (
        _random(rng, 2, 3, 4),
        _random(rng, 2, 5, 4),
        _random(rng, 2, 5, 4),
    )
    grad_output = _random(rng, 2, 3, 4)
    _check_against_finite_differences(query, key, value, grad_output, scale=0.37)


def test_gradients_causal() -> None:
    rng = np.random.default_rng(2)
    query, key, value = (
        _random(rng, 2, 4, 3),
        _random(rng, 2, 4, 3),
        _random(rng, 2, 4, 2),
    )
    grad_output = _random(rng, 2, 4, 2)
    _check_against_finite_differences(query, key, value, grad_output, is_causal=True)


def test_gradients_boolean_mask() -> None:
    rng = np.random.default_rng(3)
    query, key, value = _random(rng, 3, 4), _random(rng, 5, 4), _random(rng, 5, 3)
    grad_output = _random(rng, 3, 3)
    mask = rng.random((3, 5)) > 0.4
    mask[:, 0] = True  # keep every row attending to something
    _check_against_finite_differences(query, key, value, grad_output, attn_mask=mask)


def test_gradients_additive_mask() -> None:
    rng = np.random.default_rng(4)
    query, key, value = (
        _random(rng, 2, 3, 4),
        _random(rng, 2, 5, 4),
        _random(rng, 2, 5, 4),
    )
    grad_output = _random(rng, 2, 3, 4)
    bias = _random(rng, 1, 3, 5)
    _check_against_finite_differences(query, key, value, grad_output, attn_mask=bias)


def test_gradients_grouped_query_attention() -> None:
    rng = np.random.default_rng(5)
    query = _random(rng, 2, 6, 3, 4)
    key = _random(rng, 2, 2, 5, 4)
    value = _random(rng, 2, 2, 5, 4)
    grad_output = _random(rng, 2, 6, 3, 4)
    _check_against_finite_differences(query, key, value, grad_output, enable_gqa=True)


def test_gqa_with_one_head_per_group_is_plain_attention() -> None:
    """``H_q == H_kv`` takes the no-repeat path and must equal the plain call."""
    rng = np.random.default_rng(6)
    query = _random(rng, 2, 3, 4)
    key = _random(rng, 2, 5, 4)
    value = _random(rng, 2, 5, 4)
    grad_output = _random(rng, 2, 3, 4)
    grouped = scaled_dot_product_attention_backward(
        grad_output, query, key, value, enable_gqa=True
    )
    plain = scaled_dot_product_attention_backward(grad_output, query, key, value)
    for actual, expected in zip(grouped, plain):
        np.testing.assert_array_equal(actual, expected)


def test_broadcast_key_value_gradients_are_summed() -> None:
    """A key/value the forward pass broadcast gets the sum over the batch."""
    rng = np.random.default_rng(7)
    query = _random(rng, 3, 2, 4)
    key = _random(rng, 1, 5, 4)
    value = _random(rng, 1, 5, 4)
    grad_output = _random(rng, 3, 2, 4)

    dq, dk, dv = scaled_dot_product_attention_backward(grad_output, query, key, value)
    assert dq.shape == query.shape
    assert dk.shape == key.shape
    assert dv.shape == value.shape

    # Same gradients as tiling the key/value out by hand and summing the batch.
    tiled_key = np.broadcast_to(key, (3, 5, 4)).copy()
    tiled_value = np.broadcast_to(value, (3, 5, 4)).copy()
    _, tiled_dk, tiled_dv = scaled_dot_product_attention_backward(
        grad_output, query, tiled_key, tiled_value
    )
    np.testing.assert_allclose(dk, tiled_dk.sum(axis=0, keepdims=True))
    np.testing.assert_allclose(dv, tiled_dv.sum(axis=0, keepdims=True))


def test_unbatched_query_against_batched_key_reduces_leading_axis() -> None:
    """A 2-D query under a batched key collects gradient from every slice."""
    rng = np.random.default_rng(8)
    query = _random(rng, 3, 4)
    key = _random(rng, 2, 5, 4)
    value = _random(rng, 2, 5, 4)
    grad_output = _random(rng, 2, 3, 4)
    _check_against_finite_differences(query, key, value, grad_output)


def test_fully_masked_rows_have_zero_gradient() -> None:
    rng = np.random.default_rng(9)
    query, key, value = _random(rng, 2, 4), _random(rng, 3, 4), _random(rng, 3, 4)
    grad_output = np.ones((2, 4))
    mask = np.zeros((2, 3), dtype=bool)
    mask[0] = True

    dq, dk, dv = scaled_dot_product_attention_backward(
        grad_output, query, key, value, attn_mask=mask
    )
    assert np.all(np.isfinite(dq)) and np.all(np.isfinite(dk))
    np.testing.assert_array_equal(dq[1], np.zeros(4))

    # The masked-out row contributes nothing, so dropping it changes nothing.
    kept = scaled_dot_product_attention_backward(
        grad_output[:1], query[:1], key, value, attn_mask=mask[:1]
    )
    np.testing.assert_allclose(dk, kept[1])
    np.testing.assert_allclose(dv, kept[2])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gradient_dtypes_follow_their_inputs(dtype: type[np.floating]) -> None:
    rng = np.random.default_rng(10)
    query = _random(rng, 2, 3, 4).astype(dtype)
    key = _random(rng, 2, 5, 4).astype(np.float64)
    value = _random(rng, 2, 5, 4).astype(dtype)
    grad_output = _random(rng, 2, 3, 4).astype(dtype)

    dq, dk, dv = scaled_dot_product_attention_backward(grad_output, query, key, value)
    assert dq.dtype == np.dtype(dtype)
    assert dk.dtype == np.dtype(np.float64)
    assert dv.dtype == np.dtype(dtype)


def test_zero_upstream_gradient_gives_zero_gradients() -> None:
    rng = np.random.default_rng(11)
    query, key, value = (
        _random(rng, 2, 3, 4),
        _random(rng, 2, 5, 4),
        _random(rng, 2, 5, 4),
    )
    dq, dk, dv = scaled_dot_product_attention_backward(
        np.zeros((2, 3, 4)), query, key, value
    )
    for grad in (dq, dk, dv):
        np.testing.assert_array_equal(grad, np.zeros_like(grad))


def test_grad_output_shape_is_checked() -> None:
    rng = np.random.default_rng(12)
    query, key, value = (
        _random(rng, 2, 3, 4),
        _random(rng, 2, 5, 4),
        _random(rng, 2, 5, 4),
    )
    with pytest.raises(ValueError, match=r"grad_output has shape \(2, 3, 3\)"):
        scaled_dot_product_attention_backward(np.zeros((2, 3, 3)), query, key, value)


def test_causal_and_mask_together_are_rejected() -> None:
    rng = np.random.default_rng(13)
    query, key, value = _random(rng, 3, 4), _random(rng, 3, 4), _random(rng, 3, 4)
    with pytest.raises(ValueError, match="either is_causal=True or attn_mask"):
        scaled_dot_product_attention_backward(
            np.zeros((3, 4)),
            query,
            key,
            value,
            attn_mask=np.ones((3, 3), dtype=bool),
            is_causal=True,
        )


def test_mismatched_embedding_dims_are_rejected() -> None:
    rng = np.random.default_rng(14)
    query, key, value = _random(rng, 3, 4), _random(rng, 3, 5), _random(rng, 3, 5)
    with pytest.raises(ValueError, match="query/key embedding dims differ"):
        scaled_dot_product_attention_backward(np.zeros((3, 5)), query, key, value)


def test_invalid_gqa_head_counts_are_rejected() -> None:
    rng = np.random.default_rng(15)
    query = _random(rng, 5, 3, 4)
    key = _random(rng, 2, 3, 4)
    value = _random(rng, 2, 3, 4)
    with pytest.raises(ValueError, match="must be a positive multiple"):
        scaled_dot_product_attention_backward(
            np.zeros((5, 3, 4)), query, key, value, enable_gqa=True
        )
