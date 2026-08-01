"""Shared conformance kit: the contract every backend must satisfy.

The project's portability promise is *developer parity*: code written against
:func:`portable_attention.scaled_dot_product_attention` must behave the same on
any backend, not just the one the author happened to test on. This module makes
that promise executable. It defines a canonical matrix of attention cases and a
runner that checks a backend's output against the CPU ``reference`` oracle, so
every registered backend — and any third-party backend an author writes — can be
held to exactly one bar.

A backend *conforms* when, for every case, it:

* returns the same shape as the oracle;
* preserves ``query``'s dtype (no silent upcast to float64);
* produces only finite values; and
* agrees with the oracle output to a dtype-appropriate floating tolerance,
  including exact zeros for fully-masked rows.

Typical use, both in the project's own tests and downstream::

    from portable_attention import get_backend
    from portable_attention.conformance import assert_conforms

    assert_conforms(get_backend("fused"))

The cases are exposed as data (:func:`conformance_cases`) so a test suite can
parametrize over them, and :func:`check_backend` returns structured results for
programmatic inspection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .reference import scaled_dot_product_attention as _reference

__all__ = [
    "ConformanceCase",
    "ConformanceResult",
    "assert_conforms",
    "check_backend",
    "check_case",
    "conformance_cases",
]

Array = NDArray[np.floating]

# A backend callable matching ``scaled_dot_product_attention``. Kept as a plain
# alias (not the runtime protocol) so the kit imports nothing from ``dispatch``
# and can validate any conforming callable, including bare functions.
Backend = Callable[..., Array]

# The oracle, viewed through the generic backend alias so ``**kwargs`` (whose
# values are ``object``) type-checks against its call rather than the concrete
# strongly-typed reference signature.
_oracle: Backend = _reference

# Per-dtype agreement budget against the float64 oracle. float64 backends should
# match it almost exactly; lower-precision backends compute natively and so
# carry the wider budget their arithmetic granularity requires.
_TOL: dict[np.dtype[np.floating], dict[str, float]] = {
    np.dtype(np.float64): {"rtol": 1e-11, "atol": 1e-11},
    np.dtype(np.float32): {"rtol": 1e-4, "atol": 1e-5},
    np.dtype(np.float16): {"rtol": 2e-3, "atol": 2e-3},
}


def _tol_for(dtype: np.dtype[np.floating]) -> dict[str, float]:
    try:
        return _TOL[dtype]
    except KeyError:  # pragma: no cover - guards future dtype additions
        raise ValueError(
            f"no conformance tolerance defined for dtype {dtype}; add one to "
            "portable_attention.conformance._TOL."
        ) from None


@dataclass(frozen=True)
class ConformanceCase:
    """One attention scenario a backend must reproduce.

    Attributes:
        name: Stable identifier used in test ids and failure messages.
        make_inputs: Zero-argument factory returning the ``(query, key, value)``
            triple. It must be deterministic so failures reproduce; the built-in
            cases seed their RNG.
        kwargs: Keyword arguments forwarded to both the backend and the oracle
            (``scale``, ``is_causal``, ``attn_mask``, ``enable_gqa``, ...).
    """

    name: str
    make_inputs: Callable[[], tuple[Array, Array, Array]]
    kwargs: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ConformanceResult:
    """Outcome of running one :class:`ConformanceCase` against a backend."""

    name: str
    passed: bool
    detail: str = ""


def _inputs(
    shape_q: tuple[int, ...],
    shape_k: tuple[int, ...],
    shape_v: tuple[int, ...],
    dtype: type[np.floating],
    seed: int,
) -> Callable[[], tuple[Array, Array, Array]]:
    """Build a deterministic input factory for the given shapes and dtype."""

    def factory() -> tuple[Array, Array, Array]:
        rng = np.random.default_rng(seed)
        dt = np.dtype(dtype)
        q = rng.standard_normal(shape_q).astype(dt)
        k = rng.standard_normal(shape_k).astype(dt)
        v = rng.standard_normal(shape_v).astype(dt)
        return q, k, v

    return factory


# Canonical shape matrix: no-batch, single-batch, batch+head leading dims, plus
# non-square scores (L != S) and a differing value width (Ev != E).
_SHAPES: list[tuple[tuple[int, ...], int, int, int, int]] = [
    ((), 4, 4, 8, 8),
    ((3,), 5, 7, 8, 4),
    ((2, 3), 5, 7, 6, 4),
    ((1, 8), 16, 16, 64, 64),
]
_DTYPES: list[type[np.floating]] = [np.float64, np.float32, np.float16]


def conformance_cases() -> list[ConformanceCase]:
    """Return the canonical conformance matrix every backend must pass.

    The list is freshly built on each call (cases are cheap descriptors), so
    callers may filter or extend it without disturbing the shared definition.
    It spans the documented contract surface: leading dims, non-square scores,
    differing value widths, all supported dtypes, ``scale``, causal masking,
    boolean and additive masks, fully-masked rows (the exact-zero guarantee),
    and grouped-query attention including its causal composition.
    """
    cases: list[ConformanceCase] = []

    for dtype in _DTYPES:
        for lead, length, source, e, ev in _SHAPES:
            cases.append(
                ConformanceCase(
                    name=f"unmasked-{np.dtype(dtype).name}-{lead}-{length}x{source}",
                    make_inputs=_inputs(
                        (*lead, length, e),
                        (*lead, source, e),
                        (*lead, source, ev),
                        dtype,
                        seed=1,
                    ),
                )
            )

    for dtype in _DTYPES:
        dt = np.dtype(dtype)
        cases.append(
            ConformanceCase(
                name=f"scaled-{dt.name}",
                make_inputs=_inputs((2, 3, 5, 8), (2, 3, 7, 8), (2, 3, 7, 4), dtype, 2),
                kwargs={"scale": 0.3},
            )
        )
        cases.append(
            ConformanceCase(
                name=f"causal-{dt.name}",
                make_inputs=_inputs((2, 4, 6, 8), (2, 4, 6, 8), (2, 4, 6, 4), dtype, 3),
                kwargs={"is_causal": True},
            )
        )

    # Boolean mask (keep where True) broadcast over the batch axis.
    def _bool_mask_inputs() -> tuple[Array, Array, Array]:
        return _inputs((2, 5, 8), (2, 7, 8), (2, 7, 4), np.float32, 4)()

    bool_rng = np.random.default_rng(40)
    bool_mask = bool_rng.random((5, 7)) > 0.3
    cases.append(
        ConformanceCase(
            name="bool-mask",
            make_inputs=_bool_mask_inputs,
            kwargs={"attn_mask": bool_mask},
        )
    )

    # Additive (floating) mask added to the scores.
    additive_rng = np.random.default_rng(50)
    additive_mask = additive_rng.standard_normal((5, 7)).astype(np.float32)
    cases.append(
        ConformanceCase(
            name="additive-mask",
            make_inputs=_inputs((2, 5, 8), (2, 7, 8), (2, 7, 4), np.float32, 5),
            kwargs={"attn_mask": additive_mask},
        )
    )

    # Fully-masked rows: every key masked for the first batch element, exercising
    # the exact-zero contract the runner checks against the oracle.
    fully_masked = np.ones((3, 5, 7), dtype=bool)
    fully_masked[0] = False
    cases.append(
        ConformanceCase(
            name="fully-masked-rows",
            make_inputs=_inputs((3, 5, 8), (3, 7, 8), (3, 7, 4), np.float32, 6),
            kwargs={"attn_mask": fully_masked},
        )
    )

    # Grouped-query attention: key/value carry fewer heads than query.
    for q_heads, kv_heads in [(8, 2), (6, 3), (4, 1), (4, 4)]:
        cases.append(
            ConformanceCase(
                name=f"gqa-{q_heads}q-{kv_heads}kv",
                make_inputs=_inputs(
                    (2, q_heads, 5, 8),
                    (2, kv_heads, 7, 8),
                    (2, kv_heads, 7, 4),
                    np.float64,
                    7,
                ),
                kwargs={"enable_gqa": True},
            )
        )
    cases.append(
        ConformanceCase(
            name="gqa-causal",
            make_inputs=_inputs(
                (2, 8, 6, 8), (2, 2, 6, 8), (2, 2, 6, 4), np.float64, 8
            ),
            kwargs={"enable_gqa": True, "is_causal": True},
        )
    )

    return cases


def _describe(case: ConformanceCase, message: str) -> ConformanceResult:
    return ConformanceResult(name=case.name, passed=False, detail=message)


def check_case(backend: Backend, case: ConformanceCase) -> ConformanceResult:
    """Run one case against ``backend`` and return a structured result.

    The oracle output is computed from the CPU ``reference`` backend on the same
    inputs and keyword arguments. The backend passes the case when its output
    matches the oracle in shape, dtype, and value (to a dtype-appropriate
    tolerance), stays finite, and reproduces the oracle's exact zeros for
    fully-masked rows. This never raises for an ordinary mismatch; it records
    the reason in the result so a caller can report every failure at once.
    """
    query, key, value = case.make_inputs()
    kwargs = dict(case.kwargs)
    want = _oracle(query, key, value, **kwargs)
    got = backend(query, key, value, **kwargs)

    if got.shape != want.shape:
        return _describe(case, f"shape {got.shape} != oracle {want.shape}")
    if got.dtype != query.dtype:
        return _describe(
            case, f"dtype {got.dtype} != query dtype {query.dtype} (silent cast)"
        )
    if not np.all(np.isfinite(got)):
        return _describe(case, "output contains non-finite values")

    tol = _tol_for(query.dtype)
    # Compare in float64 so a low-precision output is measured against the
    # oracle without its own storage granularity skewing the tolerance.
    got64 = got.astype(np.float64)
    want64 = want.astype(np.float64)
    if not np.allclose(got64, want64, rtol=tol["rtol"], atol=tol["atol"]):
        max_abs = float(np.max(np.abs(got64 - want64)))
        return _describe(case, f"value mismatch vs oracle (max abs diff {max_abs:.3e})")

    # Exact-zero contract: any fully-zero oracle row (a fully-masked position)
    # must be exactly zero in the backend output too, not merely within tol.
    zero_rows = np.all(want == 0.0, axis=-1)
    if np.any(zero_rows) and not np.array_equal(got[zero_rows], want[zero_rows]):
        return _describe(case, "fully-masked rows are not exactly zero")

    return ConformanceResult(name=case.name, passed=True)


def check_backend(
    backend: Backend, cases: list[ConformanceCase] | None = None
) -> list[ConformanceResult]:
    """Run the full conformance matrix against ``backend``.

    Args:
        backend: The callable under test (e.g. from ``get_backend(name)``).
        cases: Cases to run; defaults to :func:`conformance_cases`.

    Returns:
        One :class:`ConformanceResult` per case, in order.
    """
    if cases is None:
        cases = conformance_cases()
    return [check_case(backend, case) for case in cases]


def assert_conforms(
    backend: Backend, cases: list[ConformanceCase] | None = None
) -> None:
    """Assert ``backend`` passes every conformance case.

    Raises:
        AssertionError: If any case fails, listing each failing case and reason.
    """
    failures = [r for r in check_backend(backend, cases) if not r.passed]
    if failures:
        report = "\n".join(f"  - {r.name}: {r.detail}" for r in failures)
        raise AssertionError(
            f"backend failed {len(failures)} conformance case(s):\n{report}"
        )
