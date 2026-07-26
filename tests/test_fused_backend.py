"""Conformance suite for the fused CPU backend vs the reference oracle.

The fused backend is a performance-oriented reimplementation (native precision,
single-threaded BLAS) of the same forward attention. Correctness is defined as
agreement with the CPU ``reference`` backend across the full contract matrix:
leading dims, non-square scores, dtypes, scale, and every masking mode. These
tests also lock the two hard guarantees the fast path must never break —
``query``'s dtype is preserved, and fully-masked rows resolve to exact zeros —
and confirm the backend is registered and satisfies the protocol.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from portable_attention import (
    SdpaBackend,
    available_backends,
    get_backend,
)
from portable_attention.fused import scaled_dot_product_attention as fused
from portable_attention.reference import (
    scaled_dot_product_attention as reference,
)

# float32 is computed natively by the fused backend, so it carries a wider error
# budget than the float64 path (which both backends compute identically).
_TOL = {
    np.dtype(np.float64): dict(rtol=1e-11, atol=1e-11),
    np.dtype(np.float32): dict(rtol=1e-4, atol=1e-5),
}


def _inputs(
    shape_q: tuple[int, ...],
    shape_k: tuple[int, ...],
    shape_v: tuple[int, ...],
    dtype: np.dtype[np.floating],
    seed: int = 0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(shape_q).astype(dtype)
    k = rng.standard_normal(shape_k).astype(dtype)
    v = rng.standard_normal(shape_v).astype(dtype)
    return q, k, v


def test_fused_is_registered_and_available() -> None:
    assert "fused" in available_backends()


def test_fused_satisfies_protocol() -> None:
    assert isinstance(get_backend("fused"), SdpaBackend)


def test_fused_lookup_returns_the_module_callable() -> None:
    assert get_backend("fused") is fused


DTYPES = [np.float32, np.float64]

# (batch dims, L, S, E, Ev) — includes no-batch, single-batch, and batch+head
# leading dims, plus non-square scores (L != S) and differing value width.
SHAPES = [
    ((), 4, 4, 8, 8),
    ((3,), 5, 7, 8, 4),
    ((2, 3), 5, 7, 6, 4),
    ((1, 8), 16, 16, 64, 64),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lead,length,source,e,ev", SHAPES)
def test_matches_reference_unmasked(
    dtype: type[np.floating],
    lead: tuple[int, ...],
    length: int,
    source: int,
    e: int,
    ev: int,
) -> None:
    dt = np.dtype(dtype)
    q, k, v = _inputs((*lead, length, e), (*lead, source, e), (*lead, source, ev), dt)
    out = fused(q, k, v)
    assert out.dtype == dt
    np.testing.assert_allclose(out, reference(q, k, v), **_TOL[dt])


@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_reference_scaled(dtype: type[np.floating]) -> None:
    dt = np.dtype(dtype)
    q, k, v = _inputs((2, 3, 5, 8), (2, 3, 7, 8), (2, 3, 7, 4), dt)
    out = fused(q, k, v, scale=0.3)
    assert out.dtype == dt
    np.testing.assert_allclose(out, reference(q, k, v, scale=0.3), **_TOL[dt])


@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_reference_causal(dtype: type[np.floating]) -> None:
    dt = np.dtype(dtype)
    q, k, v = _inputs((2, 4, 6, 8), (2, 4, 6, 8), (2, 4, 6, 4), dt)
    out = fused(q, k, v, is_causal=True)
    assert out.dtype == dt
    np.testing.assert_allclose(out, reference(q, k, v, is_causal=True), **_TOL[dt])


@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_reference_bool_mask(dtype: type[np.floating]) -> None:
    dt = np.dtype(dtype)
    q, k, v = _inputs((2, 5, 8), (2, 7, 8), (2, 7, 4), dt)
    rng = np.random.default_rng(1)
    mask = rng.random((5, 7)) > 0.3  # broadcast over the batch axis
    out = fused(q, k, v, attn_mask=mask)
    assert out.dtype == dt
    np.testing.assert_allclose(out, reference(q, k, v, attn_mask=mask), **_TOL[dt])


@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_reference_additive_mask(dtype: type[np.floating]) -> None:
    dt = np.dtype(dtype)
    q, k, v = _inputs((2, 5, 8), (2, 7, 8), (2, 7, 4), dt)
    rng = np.random.default_rng(2)
    bias = rng.standard_normal((5, 7)).astype(dt)
    out = fused(q, k, v, attn_mask=bias)
    assert out.dtype == dt
    np.testing.assert_allclose(out, reference(q, k, v, attn_mask=bias), **_TOL[dt])


def test_float16_input_is_promoted_and_matches_reference() -> None:
    q, k, v = _inputs((2, 4, 6, 8), (2, 4, 6, 8), (2, 4, 6, 4), np.dtype(np.float16))
    out = fused(q, k, v)
    assert out.dtype == np.dtype(np.float16)
    # Compare in float32: both promote float16 for compute, so agreement is
    # tight relative to the float16 output granularity.
    np.testing.assert_allclose(
        out.astype(np.float32),
        reference(q, k, v).astype(np.float32),
        rtol=2e-3,
        atol=2e-3,
    )


def test_fully_masked_rows_are_exact_zero() -> None:
    q, k, v = _inputs((3, 5, 8), (3, 7, 8), (3, 7, 4), np.dtype(np.float32))
    mask = np.ones((3, 5, 7), dtype=bool)
    mask[0] = False  # every key masked for the first batch element
    out = fused(q, k, v, attn_mask=mask)
    np.testing.assert_array_equal(out[0], np.zeros_like(out[0]))
    assert np.all(np.isfinite(out))


def test_rejects_dropout() -> None:
    q, k, v = _inputs((2, 4, 8), (2, 4, 8), (2, 4, 8), np.dtype(np.float32))
    with pytest.raises(NotImplementedError, match="dropout_p"):
        fused(q, k, v, dropout_p=0.1)


def test_rejects_enable_gqa() -> None:
    q, k, v = _inputs((2, 4, 8), (2, 4, 8), (2, 4, 8), np.dtype(np.float32))
    with pytest.raises(NotImplementedError, match="enable_gqa"):
        fused(q, k, v, enable_gqa=True)


def test_rejects_causal_with_mask() -> None:
    q, k, v = _inputs((4, 4, 8), (4, 4, 8), (4, 4, 8), np.dtype(np.float32))
    mask = np.ones((4, 4), dtype=bool)
    with pytest.raises(ValueError, match="either is_causal"):
        fused(q, k, v, attn_mask=mask, is_causal=True)


def test_rejects_low_rank_inputs() -> None:
    vec = np.ones(8, dtype=np.float32)
    with pytest.raises(ValueError, match="at least 2 dims"):
        fused(vec, vec, vec)


def test_rejects_mismatched_embedding_dim() -> None:
    q = np.ones((4, 8), dtype=np.float32)
    k = np.ones((4, 6), dtype=np.float32)
    v = np.ones((4, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="embedding dims differ"):
        fused(q, k, v)


def test_rejects_mismatched_key_value_length() -> None:
    q = np.ones((4, 8), dtype=np.float32)
    k = np.ones((5, 8), dtype=np.float32)
    v = np.ones((6, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="sequence dims differ"):
        fused(q, k, v)


def test_threadpoolctl_probe_returns_module_when_present() -> None:
    import portable_attention.fused as fused_mod

    assert fused_mod._threadpoolctl() is not None


def test_threadpoolctl_probe_returns_none_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    import portable_attention.fused as fused_mod

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "threadpoolctl":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert fused_mod._threadpoolctl() is None


def test_runs_without_threadpoolctl(monkeypatch: pytest.MonkeyPatch) -> None:
    # When threadpoolctl is unavailable the backend must still compute correctly
    # (it simply skips the BLAS thread pin).
    import portable_attention.fused as fused_mod

    monkeypatch.setattr(fused_mod, "_threadpoolctl", lambda: None)
    q, k, v = _inputs((2, 4, 6, 8), (2, 4, 6, 8), (2, 4, 6, 4), np.dtype(np.float32))
    np.testing.assert_allclose(
        fused(q, k, v), reference(q, k, v), **_TOL[np.dtype(np.float32)]
    )
