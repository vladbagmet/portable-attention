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

The *backward* pass has its own half of the kit
(:func:`backward_conformance_cases`, :func:`check_backward`,
:func:`assert_backward_conforms`), checked the same way against
:func:`portable_attention.backward.scaled_dot_product_attention_backward`.
Backward support is optional, so only backends that advertise it are held to
that half::

    from portable_attention import get_backend, supports_backward
    from portable_attention.conformance import assert_backward_conforms

    backend = get_backend("fused")
    if supports_backward(backend):
        assert_backward_conforms(backend)
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable, cast

import numpy as np
from numpy.typing import NDArray

from .backward import scaled_dot_product_attention_backward as _reference_backward
from .reference import scaled_dot_product_attention as _reference

__all__ = [
    "ConformanceCase",
    "ConformanceResult",
    "assert_backward_conforms",
    "assert_conforms",
    "backward_conformance_cases",
    "check_backend",
    "check_backward",
    "check_backward_case",
    "check_case",
    "conformance_cases",
]

Array = NDArray[np.floating]

# A backend callable matching ``scaled_dot_product_attention``. Kept as a plain
# alias (not the runtime protocol) so the kit imports nothing from ``dispatch``
# and can validate any conforming callable, including bare functions.
Backend = Callable[..., Array]

# A backward callable matching
# ``portable_attention.backward.scaled_dot_product_attention_backward``, kept as
# a plain alias for the same reason as ``Backend``.
Backward = Callable[..., "tuple[Array, Array, Array]"]

# The oracles, viewed through the generic aliases so ``**kwargs`` (whose values
# are ``object``) type-check against their calls rather than the concrete
# strongly-typed reference signatures.
_oracle: Backend = _reference
_backward_oracle: Backward = _reference_backward

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
    fully-masked rows. This never raises for a nonconforming backend — an
    ordinary mismatch, or an exception raised by the backend on a valid case, is
    recorded as a failed result so a caller can report every failure at once.
    (Errors from the oracle itself still propagate: those are kit bugs.)
    """
    query, key, value = case.make_inputs()
    kwargs = dict(case.kwargs)
    want = _oracle(query, key, value, **kwargs)
    try:
        got = backend(query, key, value, **kwargs)
    except Exception as exc:  # noqa: BLE001 - any raise on a valid case fails it
        return _describe(case, f"backend raised {type(exc).__name__}: {exc}")

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


# --- backward half --------------------------------------------------------


def _backward_extra_cases() -> list[ConformanceCase]:
    """Cases that exercise gradient-specific machinery the forward matrix skips.

    The backward pass carries reductions the forward pass gets from NumPy for
    free: an input that was broadcast in the forward receives the *sum* of the
    gradients over every axis that was broadcast (a leading axis of length 1, or
    a leading axis the input does not have at all). These live in the backward
    matrix only — broadcasting inputs against each other is not part of what a
    device forward kernel is required to support.
    """
    return [
        ConformanceCase(
            name="broadcast-kv-batch",
            make_inputs=_inputs(
                (2, 3, 5, 8), (1, 3, 7, 8), (1, 3, 7, 4), np.float64, 11
            ),
        ),
        ConformanceCase(
            name="unbatched-kv",
            make_inputs=_inputs((2, 3, 5, 8), (7, 8), (7, 4), np.float64, 12),
        ),
        ConformanceCase(
            name="value-only-batch-axis",
            make_inputs=_inputs((5, 8), (7, 8), (2, 7, 4), np.float64, 13),
        ),
    ]


def backward_conformance_cases() -> list[ConformanceCase]:
    """Return the canonical matrix every *backward* implementation must pass.

    This is :func:`conformance_cases` — the same shapes, dtypes, scales, masks
    and grouped-query configurations, since a gradient has to cover the contract
    surface its forward pass does — plus a few broadcast cases that only the
    backward has machinery for (see :func:`_backward_extra_cases`).

    ``dropout_p`` never appears in the matrix: the forward contract supports
    only ``0.0``, so the backward signature has no counterpart for it.
    """
    return conformance_cases() + _backward_extra_cases()


def _grad_output_for(
    case: ConformanceCase, shape: tuple[int, ...], dtype: np.dtype[np.floating]
) -> Array:
    """Build the deterministic upstream gradient for ``case``.

    Seeded from the case name (via CRC32, which — unlike ``hash`` — is stable
    across interpreter runs) so a failure reproduces exactly, and so the same
    case always feeds every backend the same gradient.
    """
    rng = np.random.default_rng(zlib.crc32(case.name.encode("utf-8")))
    grad: Array = rng.standard_normal(shape).astype(dtype)
    return grad


def _resolve_backward(backward: object) -> Callable[..., object]:
    """Accept either a bare backward callable or a backend that carries one.

    The result is typed as returning ``object``, not ``(dq, dk, dv)``: the whole
    job of the runner is to check what an implementation *actually* returned,
    which a declared return type would assume away.
    """
    entry = getattr(backward, "backward", backward)
    if not callable(entry):
        raise TypeError(
            f"{backward!r} is neither a backward callable nor a backend with a "
            "backward attribute."
        )
    return entry


def check_backward_case(backward: object, case: ConformanceCase) -> ConformanceResult:
    """Run one case against a backward implementation and return the result.

    ``backward`` may be the gradient callable itself or a backend object that
    exposes one as a ``backward`` attribute (a
    :class:`portable_attention.TrainableSdpaBackend`), so both
    ``check_backward_case(backward_for(b), case)`` and
    ``check_backward_case(b, case)`` work.

    The upstream gradient is generated deterministically from the case name and
    shaped like the oracle's forward output. The implementation passes when its
    ``(dq, dk, dv)``:

    * is a three-element tuple of arrays;
    * matches each input's shape and dtype (including where the forward pass
      broadcast that input, which the gradient must be summed back onto);
    * is finite; and
    * agrees with the CPU backward oracle to the same dtype-appropriate
      tolerance the forward half uses, with exactly-zero query gradients for
      fully-masked rows.

    Like :func:`check_case`, a nonconforming implementation produces a failed
    result rather than an exception; errors from the oracle are kit bugs and
    still propagate.
    """
    entry = _resolve_backward(backward)
    query, key, value = case.make_inputs()
    kwargs = dict(case.kwargs)
    forward_out = _oracle(query, key, value, **kwargs)
    grad_output = _grad_output_for(case, forward_out.shape, query.dtype)
    want = _backward_oracle(grad_output, query, key, value, **kwargs)

    try:
        got = entry(grad_output, query, key, value, **kwargs)
    except Exception as exc:  # noqa: BLE001 - any raise on a valid case fails it
        return _describe(case, f"backward raised {type(exc).__name__}: {exc}")

    if not isinstance(got, tuple) or len(got) != 3:
        return _describe(case, f"expected a (dq, dk, dv) 3-tuple, got {type(got)}")

    tol = _tol_for(query.dtype)
    grads: list[Array] = []
    for label, grad, want_grad, source in zip(
        ("dq", "dk", "dv"), got, want, (query, key, value)
    ):
        if not isinstance(grad, np.ndarray):
            return _describe(case, f"{label} is {type(grad)}, not an ndarray")
        grads.append(cast(Array, grad))
        if grad.shape != source.shape:
            return _describe(
                case, f"{label} shape {grad.shape} != input shape {source.shape}"
            )
        if grad.dtype != source.dtype:
            return _describe(
                case,
                f"{label} dtype {grad.dtype} != input dtype {source.dtype} "
                "(silent cast)",
            )
        if not np.all(np.isfinite(grad)):
            return _describe(case, f"{label} contains non-finite values")
        got64 = grad.astype(np.float64)
        want64 = want_grad.astype(np.float64)
        if not np.allclose(got64, want64, rtol=tol["rtol"], atol=tol["atol"]):
            max_abs = float(np.max(np.abs(got64 - want64)))
            return _describe(
                case, f"{label} mismatch vs oracle (max abs diff {max_abs:.3e})"
            )

    # A fully-masked query row attends to nothing, so its query gradient is
    # exactly zero rather than merely small — the backward counterpart of the
    # forward exact-zero contract.
    want_dq = want[0]
    zero_rows = np.all(want_dq == 0.0, axis=-1)
    if np.any(zero_rows) and not np.array_equal(
        grads[0][zero_rows], want_dq[zero_rows]
    ):
        return _describe(case, "fully-masked rows have nonzero dq")

    return ConformanceResult(name=case.name, passed=True)


def check_backward(
    backward: object, cases: list[ConformanceCase] | None = None
) -> list[ConformanceResult]:
    """Run the full backward matrix against a backward implementation.

    Args:
        backward: The gradient callable under test, or a backend carrying one
            (see :func:`check_backward_case`).
        cases: Cases to run; defaults to :func:`backward_conformance_cases`.

    Returns:
        One :class:`ConformanceResult` per case, in order.
    """
    if cases is None:
        cases = backward_conformance_cases()
    return [check_backward_case(backward, case) for case in cases]


def assert_backward_conforms(
    backward: object, cases: list[ConformanceCase] | None = None
) -> None:
    """Assert a backward implementation passes every backward case.

    Raises:
        AssertionError: If any case fails, listing each failing case and reason.
    """
    failures = [r for r in check_backward(backward, cases) if not r.passed]
    if failures:
        report = "\n".join(f"  - {r.name}: {r.detail}" for r in failures)
        raise AssertionError(
            f"backward failed {len(failures)} conformance case(s):\n{report}"
        )
