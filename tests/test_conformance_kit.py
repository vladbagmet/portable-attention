"""Tests for the shared conformance kit itself, and every backend against it.

Two layers live here. First, the kit runs each registered backend through the
full canonical matrix — this is the developer-parity promise made executable, so
adding a backend to the registry automatically holds it to the same bar without
new bespoke tests. Second, the kit's own machinery (case construction, the
oracle comparison, and each failure mode of the runner) is unit-tested so a
green conformance run is trustworthy rather than vacuous.

Both layers are repeated for the gradient half of the kit at the bottom of the
file. Backward support is optional, so there the registry sweep runs only
against backends that advertise it.
"""

from __future__ import annotations

import numpy as np
import pytest

from portable_attention import (
    available_backends,
    backward_for,
    get_backend,
    scaled_dot_product_attention_backward,
    supports_backward,
)
from portable_attention.conformance import (
    ConformanceCase,
    ConformanceResult,
    assert_backward_conforms,
    assert_conforms,
    backward_conformance_cases,
    check_backend,
    check_backward,
    check_backward_case,
    check_case,
    conformance_cases,
)

_CASES = conformance_cases()


def test_conformance_cases_are_nonempty_and_uniquely_named() -> None:
    names = [c.name for c in _CASES]
    assert names
    assert len(names) == len(set(names)), "case names must be unique"


def test_cases_cover_the_documented_contract_surface() -> None:
    names = " ".join(c.name for c in _CASES)
    for token in (
        "unmasked",
        "scaled",
        "causal",
        "bool-mask",
        "additive-mask",
        "fully-masked",
        "gqa",
    ):
        assert token in names, f"missing coverage for {token!r}"
    # All supported dtypes appear somewhere in the matrix.
    for dt in ("float64", "float32", "float16"):
        assert dt in names


@pytest.mark.parametrize("backend_name", available_backends())
def test_registered_backend_conforms(backend_name: str) -> None:
    # Every backend in the registry must pass the shared kit. New backends join
    # this parametrization automatically once registered.
    assert_conforms(get_backend(backend_name))


@pytest.mark.parametrize("backend_name", available_backends())
def test_check_backend_reports_all_pass(backend_name: str) -> None:
    results = check_backend(get_backend(backend_name))
    assert len(results) == len(_CASES)
    assert all(isinstance(r, ConformanceResult) for r in results)
    assert all(r.passed for r in results)


def test_check_backend_uses_default_matrix_length() -> None:
    results = check_backend(get_backend("reference"))
    assert len(results) == len(conformance_cases())


def test_check_backend_accepts_explicit_cases() -> None:
    subset = _CASES[:2]
    results = check_backend(get_backend("fused"), subset)
    assert [r.name for r in results] == [c.name for c in subset]


def test_reference_is_its_own_oracle() -> None:
    # The oracle trivially conforms to itself; this guards the runner against a
    # bug that would make even the reference fail.
    for result in check_backend(get_backend("reference")):
        assert result.passed, result.detail


# --- runner failure modes -------------------------------------------------

_SIMPLE_CASE = ConformanceCase(
    name="probe",
    make_inputs=lambda: (
        np.random.default_rng(0).standard_normal((2, 4, 8)).astype(np.float64),
        np.random.default_rng(1).standard_normal((2, 4, 8)).astype(np.float64),
        np.random.default_rng(2).standard_normal((2, 4, 8)).astype(np.float64),
    ),
)


def test_runner_flags_shape_mismatch() -> None:
    reference = get_backend("reference")

    def wrong_shape(query, key, value, *args, **kwargs):  # type: ignore[no-untyped-def]
        return reference(query, key, value, *args, **kwargs)[..., :1]

    result = check_case(wrong_shape, _SIMPLE_CASE)
    assert not result.passed
    assert "shape" in result.detail


def test_runner_flags_silent_dtype_upcast() -> None:
    reference = get_backend("reference")

    def upcaster(query, key, value, *args, **kwargs):  # type: ignore[no-untyped-def]
        out = reference(query, key, value, *args, **kwargs)
        return out.astype(np.float32)  # query is float64 -> wrong dtype

    result = check_case(upcaster, _SIMPLE_CASE)
    assert not result.passed
    assert "dtype" in result.detail


def test_runner_flags_non_finite_output() -> None:
    reference = get_backend("reference")

    def poisoner(query, key, value, *args, **kwargs):  # type: ignore[no-untyped-def]
        out = np.array(reference(query, key, value, *args, **kwargs))
        out[0, 0, 0] = np.nan
        return out

    result = check_case(poisoner, _SIMPLE_CASE)
    assert not result.passed
    assert "non-finite" in result.detail


def test_runner_flags_backend_that_raises() -> None:
    # A backend that raises on a valid case is a conformance failure, not a
    # crash of the runner: it must be captured as a failed result.
    def exploder(query, key, value, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    result = check_case(exploder, _SIMPLE_CASE)
    assert not result.passed
    assert "backend raised RuntimeError" in result.detail
    assert "boom" in result.detail


def test_runner_flags_value_mismatch() -> None:
    reference = get_backend("reference")

    def perturber(query, key, value, *args, **kwargs):  # type: ignore[no-untyped-def]
        return reference(query, key, value, *args, **kwargs) + 1.0

    result = check_case(perturber, _SIMPLE_CASE)
    assert not result.passed
    assert "value mismatch" in result.detail


def test_runner_flags_non_exact_zero_rows() -> None:
    reference = get_backend("reference")
    mask = np.ones((3, 5, 7), dtype=bool)
    mask[0] = False  # first batch element fully masked -> exact-zero rows
    case = ConformanceCase(
        name="fully-masked-probe",
        make_inputs=lambda: (
            np.random.default_rng(0).standard_normal((3, 5, 8)).astype(np.float64),
            np.random.default_rng(1).standard_normal((3, 7, 8)).astype(np.float64),
            np.random.default_rng(2).standard_normal((3, 7, 4)).astype(np.float64),
        ),
        kwargs={"attn_mask": mask},
    )

    def tiny_nonzero(query, key, value, *args, **kwargs):  # type: ignore[no-untyped-def]
        out = np.array(reference(query, key, value, *args, **kwargs))
        out[0] += 1e-13  # within value tol, but no longer exactly zero
        return out

    result = check_case(tiny_nonzero, case)
    assert not result.passed
    assert "exactly zero" in result.detail


def test_assert_conforms_raises_with_case_names() -> None:
    def broken(query, key, value, *args, **kwargs):  # type: ignore[no-untyped-def]
        return get_backend("reference")(query, key, value, *args, **kwargs) + 1.0

    with pytest.raises(AssertionError, match="conformance case"):
        assert_conforms(broken)


def test_assert_conforms_passes_for_reference() -> None:
    # A conforming backend produces no assertion.
    assert_conforms(get_backend("reference"))


# --- backward half --------------------------------------------------------

_BACKWARD_CASES = backward_conformance_cases()


def test_backward_matrix_extends_the_forward_one() -> None:
    names = [c.name for c in _BACKWARD_CASES]
    assert len(names) == len(set(names)), "case names must be unique"
    # The gradient has to cover everything its forward pass does...
    assert set(c.name for c in _CASES) <= set(names)
    # ...plus the broadcast reductions only the backward has machinery for.
    for token in ("broadcast-kv-batch", "unbatched-kv", "value-only-batch-axis"):
        assert token in names


@pytest.mark.parametrize("backend_name", available_backends())
def test_registered_backend_backward_conforms(backend_name: str) -> None:
    # Backward support is optional, so a backend that does not advertise it is
    # skipped rather than failed; one that does is held to the full matrix.
    backend = get_backend(backend_name)
    if not supports_backward(backend):
        pytest.skip(f"{backend_name} is inference-only")
    assert_backward_conforms(backend)


def test_reference_backward_is_its_own_oracle() -> None:
    for result in check_backward(get_backend("reference")):
        assert result.passed, result.detail


def test_check_backward_reports_all_pass_for_fused() -> None:
    results = check_backward(get_backend("fused"))
    assert len(results) == len(_BACKWARD_CASES)
    assert all(isinstance(r, ConformanceResult) for r in results)
    assert all(r.passed for r in results)


def test_check_backward_accepts_explicit_cases() -> None:
    subset = _BACKWARD_CASES[:2]
    results = check_backward(get_backend("fused"), subset)
    assert [r.name for r in results] == [c.name for c in subset]


def test_backward_accepts_a_bare_callable_and_a_backend_object() -> None:
    backend = get_backend("fused")
    from_object = check_backward(backend)
    from_callable = check_backward(backward_for(backend))
    assert [r.passed for r in from_object] == [r.passed for r in from_callable]
    assert all(r.passed for r in from_object)


def test_backward_case_gradient_is_deterministic() -> None:
    # The same case must feed every implementation the same upstream gradient,
    # so a failure reproduces and two backends are compared on equal terms.
    seen: list[np.ndarray] = []

    def recorder(grad_output, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(np.array(grad_output))
        return scaled_dot_product_attention_backward(grad_output, *args, **kwargs)

    case = _BACKWARD_CASES[0]
    assert check_backward_case(recorder, case).passed
    assert check_backward_case(recorder, case).passed
    assert len(seen) == 2
    np.testing.assert_array_equal(seen[0], seen[1])


def test_resolve_rejects_something_that_is_neither() -> None:
    with pytest.raises(TypeError, match="neither a backward callable"):
        check_backward_case(object(), _BACKWARD_CASES[0])


# --- backward runner failure modes ---------------------------------------

_BACKWARD_PROBE = _BACKWARD_CASES[0]


def _oracle_backward(grad_output, query, key, value, *args, **kwargs):  # type: ignore[no-untyped-def]
    return scaled_dot_product_attention_backward(
        grad_output, query, key, value, *args, **kwargs
    )


def test_backward_runner_flags_an_implementation_that_raises() -> None:
    def exploder(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    result = check_backward_case(exploder, _BACKWARD_PROBE)
    assert not result.passed
    assert "backward raised RuntimeError" in result.detail


def test_backward_runner_flags_a_non_tuple_return() -> None:
    def single(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _oracle_backward(*args, **kwargs)[0]

    result = check_backward_case(single, _BACKWARD_PROBE)
    assert not result.passed
    assert "3-tuple" in result.detail


def test_backward_runner_flags_a_wrong_length_tuple() -> None:
    def pair(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _oracle_backward(*args, **kwargs)[:2]

    result = check_backward_case(pair, _BACKWARD_PROBE)
    assert not result.passed
    assert "3-tuple" in result.detail


def test_backward_runner_flags_a_non_array_element() -> None:
    def listy(*args, **kwargs):  # type: ignore[no-untyped-def]
        dq, dk, dv = _oracle_backward(*args, **kwargs)
        return dq.tolist(), dk, dv

    result = check_backward_case(listy, _BACKWARD_PROBE)
    assert not result.passed
    assert "not an ndarray" in result.detail


def test_backward_runner_flags_shape_mismatch() -> None:
    def truncator(*args, **kwargs):  # type: ignore[no-untyped-def]
        dq, dk, dv = _oracle_backward(*args, **kwargs)
        return dq, dk[..., :1], dv

    result = check_backward_case(truncator, _BACKWARD_PROBE)
    assert not result.passed
    assert result.detail.startswith("dk shape")


def test_backward_runner_flags_silent_dtype_upcast() -> None:
    def upcaster(*args, **kwargs):  # type: ignore[no-untyped-def]
        dq, dk, dv = _oracle_backward(*args, **kwargs)
        return dq, dk, dv.astype(np.float32)  # inputs are float64

    result = check_backward_case(upcaster, _BACKWARD_PROBE)
    assert not result.passed
    assert "dv dtype" in result.detail


def test_backward_runner_flags_non_finite_gradients() -> None:
    def poisoner(*args, **kwargs):  # type: ignore[no-untyped-def]
        dq, dk, dv = _oracle_backward(*args, **kwargs)
        dq = np.array(dq)
        dq.reshape(-1)[0] = np.inf
        return dq, dk, dv

    result = check_backward_case(poisoner, _BACKWARD_PROBE)
    assert not result.passed
    assert "dq contains non-finite" in result.detail


def test_backward_runner_flags_value_mismatch() -> None:
    def perturber(*args, **kwargs):  # type: ignore[no-untyped-def]
        dq, dk, dv = _oracle_backward(*args, **kwargs)
        return dq, dk, dv + 1.0

    result = check_backward_case(perturber, _BACKWARD_PROBE)
    assert not result.passed
    assert "dv mismatch vs oracle" in result.detail


def test_backward_runner_flags_nonzero_dq_on_fully_masked_rows() -> None:
    case = next(c for c in _BACKWARD_CASES if c.name == "fully-masked-rows")

    def tiny_nonzero(*args, **kwargs):  # type: ignore[no-untyped-def]
        dq, dk, dv = _oracle_backward(*args, **kwargs)
        dq = np.array(dq)
        dq[0] += 1e-9  # inside the value tolerance, but no longer exactly zero
        return dq, dk, dv

    # The case really does contain fully-masked rows, or the check below would
    # pass vacuously.
    reference_dq = check_backward_case(_oracle_backward, case)
    assert reference_dq.passed
    result = check_backward_case(tiny_nonzero, case)
    assert not result.passed
    assert "nonzero dq" in result.detail


def test_assert_backward_conforms_raises_with_case_names() -> None:
    def broken(*args, **kwargs):  # type: ignore[no-untyped-def]
        dq, dk, dv = _oracle_backward(*args, **kwargs)
        return dq + 1.0, dk, dv

    with pytest.raises(AssertionError, match="conformance case"):
        assert_backward_conforms(broken)
