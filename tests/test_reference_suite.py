"""Exhaustive correctness sweep for the CPU reference SDPA (M0.2).

Every case is checked against an independent naive einsum-softmax oracle. The
oracle is written to a different formulation than the implementation under test
(``np.einsum`` contractions, an explicit boolean/additive mask branch, and a
``nan_to_num`` softmax) so agreement is evidence of correctness rather than a
copied algorithm. Inputs are kept at modest magnitudes (unit-variance normals,
small embed dims) so the un-shifted oracle softmax cannot overflow, letting it
stay deliberately simple.

Coverage matrix:
  * leading dims: none, one batch axis, batch+head axes, and broadcast masks
  * L != S and E != Ev (non-square scores, differing value width)
  * dtypes: float32 and float64
  * scale: default (1/sqrt(E)) and explicit values
  * masking: none, boolean keep-mask, additive float bias, is_causal
  * degenerate: fully-masked rows resolve to a finite zero output
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from portable_attention import scaled_dot_product_attention

# Tolerances: float64 is compared tightly; float32 inputs are computed by the
# reference in float64 then cast back, so the cast dominates the error budget.
_TOL = {
    np.dtype(np.float64): dict(rtol=1e-11, atol=1e-11),
    np.dtype(np.float32): dict(rtol=1e-5, atol=1e-6),
}


def _oracle(
    q: NDArray[np.floating],
    k: NDArray[np.floating],
    v: NDArray[np.floating],
    attn_mask: NDArray[np.floating] | NDArray[np.bool_] | None,
    is_causal: bool,
    scale: float | None,
) -> NDArray[np.float64]:
    """Independent reference: naive einsum scores + nan-safe softmax."""
    q64 = q.astype(np.float64)
    k64 = k.astype(np.float64)
    v64 = v.astype(np.float64)
    e = q64.shape[-1]
    s = (1.0 / np.sqrt(e)) if scale is None else scale

    scores = np.einsum("...ie,...je->...ij", q64, k64) * s

    if is_causal:
        length, source = scores.shape[-2], scores.shape[-1]
        keep = np.tril(np.ones((length, source), dtype=bool))
        scores = np.where(keep, scores, -np.inf)
    elif attn_mask is not None:
        if attn_mask.dtype == np.bool_:
            scores = np.where(attn_mask, scores, -np.inf)
        else:
            scores = scores + attn_mask.astype(np.float64)

    weights = np.exp(scores)  # safe: modest inputs, no overflow
    denom = weights.sum(axis=-1, keepdims=True)
    # Fully-masked rows are 0/0 -> define as 0 (matches the implementation).
    weights = np.where(denom > 0, weights / np.where(denom > 0, denom, 1.0), 0.0)
    return np.einsum("...ij,...jd->...id", weights, v64)


# (leading, L, S, E, Ev) — exercises L!=S, E!=Ev, and 0/1/2 leading axes.
_SHAPES = [
    ((), 4, 4, 8, 8),
    ((), 3, 5, 8, 4),
    ((), 5, 3, 6, 7),
    ((2,), 4, 6, 8, 3),
    ((2, 3), 5, 5, 4, 4),
    ((1, 2), 7, 2, 5, 6),
]
_DTYPES = [np.float32, np.float64]
_SCALES = [None, 0.5, 1.0, 0.123]


def _randn(rng: np.random.Generator, shape: tuple[int, ...], dtype: type) -> NDArray:
    return rng.standard_normal(shape).astype(dtype)


@pytest.mark.parametrize("leading,length,source,e,ev", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_unmasked_matches_oracle(leading, length, source, e, ev, dtype):
    rng = np.random.default_rng(hash((leading, length, source, e, ev)) % 2**32)
    q = _randn(rng, (*leading, length, e), dtype)
    k = _randn(rng, (*leading, source, e), dtype)
    v = _randn(rng, (*leading, source, ev), dtype)

    out = scaled_dot_product_attention(q, k, v)
    expected = _oracle(q, k, v, None, False, None)

    assert out.shape == (*leading, length, ev)
    assert out.dtype == np.dtype(dtype)
    np.testing.assert_allclose(out, expected, **_TOL[np.dtype(dtype)])


@pytest.mark.parametrize("scale", _SCALES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_explicit_scale_matches_oracle(scale, dtype):
    rng = np.random.default_rng(100 + int((scale or 0) * 1000))
    q = _randn(rng, (2, 4, 8), dtype)
    k = _randn(rng, (2, 6, 8), dtype)
    v = _randn(rng, (2, 6, 5), dtype)

    out = scaled_dot_product_attention(q, k, v, scale=scale)
    expected = _oracle(q, k, v, None, False, scale)

    assert out.dtype == np.dtype(dtype)
    np.testing.assert_allclose(out, expected, **_TOL[np.dtype(dtype)])


@pytest.mark.parametrize("leading,length,source,e,ev", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_is_causal_matches_triangular_oracle(leading, length, source, e, ev, dtype):
    rng = np.random.default_rng(7_000 + length * 31 + source)
    q = _randn(rng, (*leading, length, e), dtype)
    k = _randn(rng, (*leading, source, e), dtype)
    v = _randn(rng, (*leading, source, ev), dtype)

    out = scaled_dot_product_attention(q, k, v, is_causal=True)
    expected = _oracle(q, k, v, None, True, None)

    assert out.dtype == np.dtype(dtype)
    np.testing.assert_allclose(out, expected, **_TOL[np.dtype(dtype)])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_boolean_mask_matches_oracle(dtype):
    rng = np.random.default_rng(222)
    q = _randn(rng, (2, 3, 5, 8), dtype)
    k = _randn(rng, (2, 3, 6, 8), dtype)
    v = _randn(rng, (2, 3, 6, 4), dtype)
    # Random keep-mask, but guarantee at least one kept key per row (no fully
    # masked rows here — that degenerate case has its own test).
    mask = rng.random((2, 3, 5, 6)) > 0.5
    mask[..., 0] = True

    out = scaled_dot_product_attention(q, k, v, attn_mask=mask)
    expected = _oracle(q, k, v, mask, False, None)

    assert out.dtype == np.dtype(dtype)
    np.testing.assert_allclose(out, expected, **_TOL[np.dtype(dtype)])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_additive_float_mask_matches_oracle(dtype):
    rng = np.random.default_rng(333)
    q = _randn(rng, (2, 4, 8), dtype)
    k = _randn(rng, (2, 7, 8), dtype)
    v = _randn(rng, (2, 7, 3), dtype)
    bias = rng.standard_normal((2, 4, 7)).astype(np.float64)

    out = scaled_dot_product_attention(q, k, v, attn_mask=bias)
    expected = _oracle(q, k, v, bias, False, None)

    assert out.dtype == np.dtype(dtype)
    np.testing.assert_allclose(out, expected, **_TOL[np.dtype(dtype)])


def test_mask_broadcasts_over_leading_dims():
    # A (L, S) mask must broadcast across batch/head axes and agree with the
    # oracle applying the same broadcast.
    rng = np.random.default_rng(444)
    q = _randn(rng, (2, 3, 5, 8), np.float64)
    k = _randn(rng, (2, 3, 6, 8), np.float64)
    v = _randn(rng, (2, 3, 6, 4), np.float64)
    mask = rng.random((5, 6)) > 0.4
    mask[:, 0] = True  # avoid fully-masked rows

    out = scaled_dot_product_attention(q, k, v, attn_mask=mask)
    expected = _oracle(q, k, v, mask, False, None)

    np.testing.assert_allclose(out, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_fully_masked_rows_are_finite_zero(dtype):
    # Some rows attend to nothing (bool mask all False): output must be exactly
    # zero and never NaN/inf, while the surviving rows still match the oracle.
    rng = np.random.default_rng(555)
    q = _randn(rng, (3, 4), dtype)
    k = _randn(rng, (5, 4), dtype)
    v = _randn(rng, (5, 6), dtype)
    mask = np.ones((3, 5), dtype=bool)
    mask[0, :] = False  # row 0 fully masked
    mask[1, 2:] = False  # row 1 partially masked

    out = scaled_dot_product_attention(q, k, v, attn_mask=mask)
    expected = _oracle(q, k, v, mask, False, None)

    assert out.dtype == np.dtype(dtype)
    assert np.all(np.isfinite(out))
    # Fully-masked rows are exactly zero, not merely close to it.
    np.testing.assert_array_equal(out[0], np.zeros_like(out[0]))
    np.testing.assert_allclose(out, expected, **_TOL[np.dtype(dtype)])


def test_causal_first_row_is_first_value():
    # Structural invariant independent of the oracle: with a causal mask the
    # first query sees only key 0, so its output equals v[0] exactly.
    rng = np.random.default_rng(666)
    q = _randn(rng, (2, 6, 8), np.float64)
    k = _randn(rng, (2, 6, 8), np.float64)
    v = _randn(rng, (2, 6, 3), np.float64)

    out = scaled_dot_product_attention(q, k, v, is_causal=True)

    np.testing.assert_allclose(out[:, 0], v[:, 0], rtol=1e-12, atol=1e-12)
