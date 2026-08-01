"""Tests for the shared conformance kit itself, and every backend against it.

Two layers live here. First, the kit runs each registered backend through the
full canonical matrix — this is the developer-parity promise made executable, so
adding a backend to the registry automatically holds it to the same bar without
new bespoke tests. Second, the kit's own machinery (case construction, the
oracle comparison, and each failure mode of the runner) is unit-tested so a
green conformance run is trustworthy rather than vacuous.
"""

from __future__ import annotations

import numpy as np
import pytest

from portable_attention import available_backends, get_backend
from portable_attention.conformance import (
    ConformanceCase,
    ConformanceResult,
    assert_conforms,
    check_backend,
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
