"""Backend registry, dispatch, and the shared conformance contract.

These lock the M1 seam: the ``reference`` backend is always registered and is
the oracle; ``get_backend`` resolves names (including ``"auto"``); registration
is guarded; and any backend that satisfies :class:`SdpaBackend` must agree with
the reference output (the conformance promise a second backend will lean on).
"""

from __future__ import annotations

import numpy as np
import pytest

import portable_attention
from portable_attention import (
    SdpaBackend,
    available_backends,
    get_backend,
    register_backend,
    scaled_dot_product_attention,
)
from portable_attention.dispatch import _REGISTRY


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot and restore the registry so tests can register freely."""
    saved = dict(_REGISTRY)
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


def _inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((2, 3, 5, 8))
    k = rng.standard_normal((2, 3, 7, 8))
    v = rng.standard_normal((2, 3, 7, 4))
    return q, k, v


def test_reference_is_registered_and_available():
    assert "reference" in available_backends()


def test_available_backends_is_sorted_and_excludes_auto():
    names = available_backends()
    assert names == sorted(names)
    assert "auto" not in names


def test_auto_resolves_to_reference():
    assert get_backend("auto") is get_backend("reference")


def test_default_get_backend_is_auto():
    assert get_backend() is get_backend("auto")


def test_reference_backend_satisfies_protocol():
    assert isinstance(get_backend("reference"), SdpaBackend)


def test_unknown_backend_raises_with_listing():
    with pytest.raises(ValueError, match="unknown backend 'nope'"):
        get_backend("nope")


def test_public_dispatch_matches_reference_backend():
    q, k, v = _inputs()
    np.testing.assert_array_equal(
        scaled_dot_product_attention(q, k, v),
        get_backend("reference")(q, k, v),
    )


def test_public_dispatch_forwards_all_arguments():
    q, k, v = _inputs()
    mask = np.tril(np.ones((5, 7), dtype=bool))
    got = scaled_dot_product_attention(q, k, v, mask, scale=0.25)
    want = get_backend("reference")(q, k, v, mask, scale=0.25)
    np.testing.assert_array_equal(got, want)


def test_register_and_get_roundtrip():
    def backend(query, key, value, *args, **kwargs):
        return get_backend("reference")(query, key, value, *args, **kwargs)

    register_backend("mirror", backend)
    assert "mirror" in available_backends()
    assert get_backend("mirror") is backend


def test_register_duplicate_raises_without_overwrite():
    register_backend("dup", get_backend("reference"))
    with pytest.raises(ValueError, match="already registered"):
        register_backend("dup", get_backend("reference"))


def test_register_overwrite_replaces():
    first = get_backend("reference")

    def second(query, key, value, *args, **kwargs):
        return first(query, key, value, *args, **kwargs)

    register_backend("swap", first)
    register_backend("swap", second, overwrite=True)
    assert get_backend("swap") is second


def test_register_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty"):
        register_backend("", get_backend("reference"))


def test_register_rejects_reserved_auto():
    with pytest.raises(ValueError, match="reserved"):
        register_backend("auto", get_backend("reference"))


def test_conformance_registered_backend_agrees_with_oracle():
    # A well-behaved backend (here a thin wrapper) must match the reference on
    # the same inputs — the contract every future backend is held to.
    reference = get_backend("reference")

    def wrapper(query, key, value, *args, **kwargs):
        return reference(query, key, value, *args, **kwargs)

    register_backend("wrapper", wrapper)
    q, k, v = _inputs(seed=1)
    for is_causal in (False, True):
        np.testing.assert_allclose(
            get_backend("wrapper")(q, k, v, is_causal=is_causal),
            reference(q, k, v, is_causal=is_causal),
            rtol=1e-12,
            atol=1e-12,
        )


def test_dispatch_module_all_is_reexported():
    # Everything dispatch exports (minus the SDPA entry, already public) is
    # surfaced on the package so the seam has a single public import path.
    for name in portable_attention.dispatch.__all__:
        assert hasattr(portable_attention, name)
