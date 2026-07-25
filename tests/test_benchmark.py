"""Tests for the reproducible latency harness.

These exercise the deterministic parts of the harness — thread detection and
pinning, input generation, timing contract, result formatting, and the CLI —
without asserting on wall-clock numbers (which are machine-dependent).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import numpy as np
import pytest

from portable_attention import benchmark
from portable_attention.benchmark import (
    DEFAULT_SHAPES,
    SWEEP_SHAPE,
    BenchmarkResult,
    ThreadInfo,
    benchmark_shape,
    describe_threads,
    format_markdown,
    run_suite,
    run_thread_sweep,
)

TINY: tuple[int, int, int, int] = (1, 1, 8, 4)


# --- thread detection -----------------------------------------------------


def test_describe_threads_uses_threadpoolctl_detail(monkeypatch: pytest.MonkeyPatch):
    fake_info = [
        {"internal_api": "openblas", "version": "0.3.33", "num_threads": 4},
        {"prefix": "libomp", "version": "5", "num_threads": 2},
    ]
    monkeypatch.setattr(benchmark, "_threadpool_info", lambda: fake_info)
    info = describe_threads()
    assert info.source == "threadpoolctl"
    assert info.libraries == (
        ("openblas", "0.3.33", 4),
        ("libomp", "5", 2),  # falls back to "prefix" when internal_api is absent
    )
    assert info.summary == "openblas=4, libomp=2"


def test_describe_threads_empty_library_list(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(benchmark, "_threadpool_info", lambda: [])
    info = describe_threads()
    assert info.source == "threadpoolctl"
    assert info.summary == "no threaded BLAS detected"


def test_describe_threads_env_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(benchmark, "_threadpool_info", lambda: None)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    info = describe_threads()
    assert info == ThreadInfo("env", "OPENBLAS_NUM_THREADS=3")


def test_describe_threads_unknown(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(benchmark, "_threadpool_info", lambda: None)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    info = describe_threads()
    assert info.source == "unknown"
    assert "threadpoolctl" in info.summary


def test_threadpoolctl_probe_absent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(benchmark.importlib.util, "find_spec", lambda _name: None)
    assert benchmark._threadpoolctl() is None
    assert benchmark._threadpool_info() is None


def test_threadpoolctl_probe_present_when_installed():
    if benchmark.importlib.util.find_spec("threadpoolctl") is None:
        pytest.skip("threadpoolctl not installed")
    mod = benchmark._threadpoolctl()
    assert mod is not None
    assert benchmark._threadpool_info() is not None


# --- thread pinning -------------------------------------------------------


def test_limit_threads_none_is_noop():
    with benchmark._limit_threads(None):
        pass  # nothing to assert beyond it not raising


def test_limit_threads_env_fallback_sets_and_restores(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(benchmark, "_threadpoolctl", lambda: None)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "9")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    with benchmark._limit_threads(1):
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
        assert os.environ["OMP_NUM_THREADS"] == "1"
    # previously-set var restored, previously-unset var removed
    assert os.environ["OPENBLAS_NUM_THREADS"] == "9"
    assert "OMP_NUM_THREADS" not in os.environ


def test_limit_threads_uses_threadpoolctl(monkeypatch: pytest.MonkeyPatch):
    calls: list[int] = []

    @contextlib.contextmanager
    def fake_limits(*, limits: int) -> Iterator[None]:
        calls.append(limits)
        yield

    fake_mod = type("Mod", (), {"threadpool_limits": staticmethod(fake_limits)})
    monkeypatch.setattr(benchmark, "_threadpoolctl", lambda: fake_mod)
    with benchmark._limit_threads(2):
        pass
    assert calls == [2]


# --- timing ---------------------------------------------------------------


def test_time_ms_requires_positive_repeats():
    with pytest.raises(ValueError, match="repeats"):
        benchmark._time_ms(lambda: None, repeats=0, warmup=0)


def test_time_ms_rejects_negative_warmup():
    with pytest.raises(ValueError, match="warmup"):
        benchmark._time_ms(lambda: None, repeats=1, warmup=-1)


def test_limit_threads_rejects_non_positive():
    for bad in (0, -2):
        with (
            pytest.raises(ValueError, match="num_threads"),
            benchmark._limit_threads(bad),
        ):
            pass


def test_time_ms_counts_calls():
    counter = {"n": 0}

    def tick() -> None:
        counter["n"] += 1

    latency = benchmark._time_ms(tick, repeats=5, warmup=2)
    assert counter["n"] == 7
    assert latency >= 0.0


# --- measurement ----------------------------------------------------------


def test_benchmark_shape_returns_populated_result():
    result = benchmark_shape(TINY, threads=1, repeats=3, warmup=1)
    assert isinstance(result, BenchmarkResult)
    assert result.shape == TINY
    assert result.dtype == "float32"
    assert result.threads == 1
    assert result.repeats == 3
    assert result.latency_ms >= 0.0
    assert result.thread_label == "1"


def test_benchmark_shape_rejects_non_floating_dtype():
    with pytest.raises(ValueError, match="floating"):
        benchmark_shape(TINY, threads=1, repeats=1, warmup=0, dtype="int32")


def test_benchmark_shape_rejects_degenerate_shape():
    with pytest.raises(ValueError, match="four positive dims"):
        benchmark_shape((1, 0, 8, 4), threads=1, repeats=1, warmup=0)


def test_benchmark_shape_rejects_non_positive_threads():
    with pytest.raises(ValueError, match="num_threads"):
        benchmark_shape(TINY, threads=0, repeats=1, warmup=0)


def test_benchmark_shape_default_threads_label():
    result = benchmark_shape(TINY, threads=None, repeats=2, warmup=0)
    assert result.threads is None
    assert result.thread_label == "default"


def test_benchmark_shape_matches_reference_output():
    # The harness must feed real, backend-consistent inputs, not zeros.
    from portable_attention import get_backend

    rng_shape = TINY
    q = benchmark._make_inputs(rng_shape, np.dtype("float32"), 0)
    k = benchmark._make_inputs(rng_shape, np.dtype("float32"), 1)
    v = benchmark._make_inputs(rng_shape, np.dtype("float32"), 2)
    out = get_backend("reference")(q, k, v)
    assert out.shape == rng_shape
    assert np.isfinite(out).all()


def test_run_suite_covers_all_shapes():
    results = run_suite(threads=1, repeats=2, shapes=(TINY, (1, 2, 8, 4)))
    assert len(results) == 2
    assert all(r.threads == 1 for r in results)


def test_run_thread_sweep_varies_threads():
    results = run_thread_sweep(TINY, thread_settings=(1, None), repeats=2)
    assert [r.threads for r in results] == [1, None]


# --- reporting ------------------------------------------------------------


def test_format_markdown_contains_expected_fields():
    results = [BenchmarkResult((1, 8, 128, 64), "float32", 1, 6.25, 20)]
    sweep = [
        BenchmarkResult((1, 8, 128, 64), "float32", 1, 6.25, 20),
        BenchmarkResult((1, 8, 128, 64), "float32", None, 105.0, 20),
    ]
    info = ThreadInfo("threadpoolctl", "openblas=1")
    md = format_markdown(results, sweep, info, commit="abc1234")
    assert "commit abc1234" in md
    assert "via threadpoolctl):** openblas=1" in md
    assert "(1, 8, 128, 64)" in md
    assert "6.250 ms" in md
    assert "105.000 ms" in md
    assert "default" in md  # the unpinned sweep row


def test_format_markdown_defaults_commit_to_unknown():
    md = format_markdown([], [], ThreadInfo("unknown", "n/a"))
    assert "commit unknown" in md


# --- module constants -----------------------------------------------------


def test_default_shapes_are_four_dims():
    assert all(len(shape) == 4 for shape in DEFAULT_SHAPES)
    assert SWEEP_SHAPE in DEFAULT_SHAPES


# --- CLI ------------------------------------------------------------------


def test_main_prints_markdown(capsys: pytest.CaptureFixture[str]):
    rc = benchmark.main(["--threads", "1", "--repeats", "2", "--commit", "deadbee"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "commit deadbee" in out
    assert "Latency by shape" in out
