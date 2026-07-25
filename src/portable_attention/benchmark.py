"""Reproducible CPU latency harness for attention backends.

NumPy's batched ``matmul`` latency is extremely sensitive to the BLAS thread
count: multi-head attention issues many small per-slice GEMMs, and the default
OpenBLAS policy (one thread per core) can spend far more time synchronising
threads than computing. Left uncontrolled this distorts every measurement.

To keep headline numbers reproducible this harness does two things for every
measurement:

* **pins** the BLAS thread count (via ``threadpoolctl`` when available), and
* **records** the effective thread configuration alongside the results.

``threadpoolctl`` is an optional dependency. When it is installed the harness
detects the real BLAS library and thread count and can pin threads precisely;
without it, the harness falls back to reporting/pinning through the
``OPENBLAS_NUM_THREADS`` environment variable (best effort — a BLAS already
loaded in the process will not re-read it).

Run it as a module to emit a Markdown block for ``BENCHMARKS.md``::

    python -m portable_attention.benchmark --threads 1 --commit abc1234
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import platform
import statistics
import time
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from . import __version__
from .dispatch import get_backend

__all__ = [
    "BenchmarkResult",
    "DEFAULT_SHAPES",
    "SWEEP_SHAPE",
    "ThreadInfo",
    "benchmark_shape",
    "describe_threads",
    "format_markdown",
    "run_suite",
    "run_thread_sweep",
]

Array = NDArray[np.floating]
Shape = tuple[int, int, int, int]

# Canonical measurement shapes (batch, heads, seq, head_dim). The single-head
# shape anchors the low end; the multi-head shapes are where the OpenBLAS
# thread-sync cliff shows up.
DEFAULT_SHAPES: tuple[Shape, ...] = (
    (1, 1, 64, 32),
    (1, 8, 128, 64),
    (2, 8, 256, 64),
    (1, 12, 512, 64),
)

# One multi-head shape used for the 1-vs-default thread sweep so the cliff
# stays visible in every report.
SWEEP_SHAPE: Shape = (1, 8, 128, 64)


@dataclass(frozen=True)
class ThreadInfo:
    """The effective BLAS thread configuration at measurement time.

    Attributes:
        source: Where the information came from — ``"threadpoolctl"`` (the
            real, per-library thread count), ``"env"`` (the
            ``OPENBLAS_NUM_THREADS`` variable), or ``"unknown"``.
        summary: A short human-readable description for the report.
        libraries: Per-library ``(api, version, num_threads)`` detail when
            ``threadpoolctl`` is available; empty otherwise.
    """

    source: str
    summary: str
    libraries: tuple[tuple[str, str, int], ...] = ()


@dataclass(frozen=True)
class BenchmarkResult:
    """One measured latency point."""

    shape: Shape
    dtype: str
    threads: int | None
    latency_ms: float
    repeats: int

    @property
    def thread_label(self) -> str:
        """Human label for the pinned thread count (``"default"`` if unpinned)."""
        return "default" if self.threads is None else str(self.threads)


def _threadpoolctl() -> ModuleType | None:
    """Import the optional ``threadpoolctl`` module, or ``None`` if absent."""
    if importlib.util.find_spec("threadpoolctl") is None:
        return None
    import threadpoolctl

    return threadpoolctl


def _threadpool_info() -> list[dict[str, object]] | None:
    """Return ``threadpoolctl.threadpool_info()`` output, or ``None``."""
    mod = _threadpoolctl()
    if mod is None:
        return None
    return list(mod.threadpool_info())


def describe_threads() -> ThreadInfo:
    """Detect the effective BLAS thread configuration for the report."""
    info = _threadpool_info()
    if info is not None:
        libraries = tuple(
            (
                str(entry.get("internal_api", entry.get("prefix", "?"))),
                str(entry.get("version", "?")),
                int(entry.get("num_threads", 0)),  # type: ignore[arg-type]
            )
            for entry in info
        )
        summary = (
            ", ".join(f"{api}={threads}" for api, _version, threads in libraries)
            or "no threaded BLAS detected"
        )
        return ThreadInfo("threadpoolctl", summary, libraries)
    env = os.environ.get("OPENBLAS_NUM_THREADS")
    if env is not None:
        return ThreadInfo("env", f"OPENBLAS_NUM_THREADS={env}")
    return ThreadInfo(
        "unknown",
        "unknown (install threadpoolctl or set OPENBLAS_NUM_THREADS)",
    )


@contextlib.contextmanager
def _env_threads(num_threads: int) -> Generator[None, None, None]:
    """Best-effort thread pinning via environment variables."""
    keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
    saved = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ[key] = str(num_threads)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _limit_threads(num_threads: int | None) -> Generator[None, None, None]:
    """Pin the BLAS thread count for the duration of the context.

    ``None`` leaves the process default untouched (used to measure the default
    policy). Otherwise ``threadpoolctl`` is used when present; failing that, the
    environment-variable fallback is applied.
    """
    if num_threads is not None and num_threads < 1:
        raise ValueError(f"num_threads must be >= 1, got {num_threads}")
    if num_threads is None:
        yield
        return
    mod = _threadpoolctl()
    if mod is None:
        with _env_threads(num_threads):
            yield
        return
    with mod.threadpool_limits(limits=num_threads):
        yield


def _make_inputs(shape: Shape, dtype: np.dtype[np.floating], seed: int) -> Array:
    b, h, s, d = shape
    rng = np.random.default_rng(seed)
    return rng.standard_normal((b, h, s, d)).astype(dtype)


def _time_ms(fn: Callable[[], object], *, repeats: int, warmup: int) -> float:
    """Return the median wall-clock latency of ``fn`` in milliseconds."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def benchmark_shape(
    shape: Shape,
    *,
    threads: int | None = None,
    dtype: str = "float32",
    repeats: int = 20,
    warmup: int = 3,
    backend: str = "reference",
    seed: int = 0,
) -> BenchmarkResult:
    """Measure median latency for one shape with the BLAS thread count pinned.

    Args:
        shape: ``(batch, heads, seq, head_dim)``. Query, key, and value all use
            this shape (self-attention).
        threads: BLAS thread count to pin for the measurement, or ``None`` to
            leave the process default in place.
        dtype: Floating dtype of the inputs.
        repeats: Number of timed calls; the median is reported.
        warmup: Untimed calls before timing (lets BLAS settle).
        backend: Registered backend name to measure.
        seed: RNG seed for the synthetic inputs.
    """
    resolved = np.dtype(dtype)
    if not np.issubdtype(resolved, np.floating):
        raise ValueError(f"dtype must be a floating type, got {resolved.name!r}")
    dims = tuple(shape)
    if len(dims) != 4 or any(dim < 1 for dim in dims):
        raise ValueError(
            f"shape must be four positive dims (B, H, S, D), got {shape!r}"
        )
    fn = get_backend(backend)
    query = _make_inputs(shape, resolved, seed)
    key = _make_inputs(shape, resolved, seed + 1)
    value = _make_inputs(shape, resolved, seed + 2)
    with _limit_threads(threads):
        latency = _time_ms(
            lambda: fn(query, key, value), repeats=repeats, warmup=warmup
        )
    return BenchmarkResult(shape, resolved.name, threads, latency, repeats)


def run_suite(
    *,
    threads: int | None = 1,
    dtype: str = "float32",
    repeats: int = 20,
    shapes: Sequence[Shape] = DEFAULT_SHAPES,
    backend: str = "reference",
) -> list[BenchmarkResult]:
    """Measure every shape at a single pinned thread count (default: 1)."""
    return [
        benchmark_shape(
            shape,
            threads=threads,
            dtype=dtype,
            repeats=repeats,
            backend=backend,
        )
        for shape in shapes
    ]


def run_thread_sweep(
    shape: Shape = SWEEP_SHAPE,
    *,
    thread_settings: Sequence[int | None] = (1, None),
    dtype: str = "float32",
    repeats: int = 20,
    backend: str = "reference",
) -> list[BenchmarkResult]:
    """Measure one shape across thread counts to keep the cliff visible."""
    return [
        benchmark_shape(
            shape,
            threads=threads,
            dtype=dtype,
            repeats=repeats,
            backend=backend,
        )
        for threads in thread_settings
    ]


def _results_table(results: Sequence[BenchmarkResult]) -> str:
    header = (
        "| shape (B,H,S,D)   | dtype   | threads | latency (median) |\n"
        "|-------------------|---------|---------|-----------------:|"
    )
    rows = [
        f"| {str(tuple(r.shape)):<17} | {r.dtype:<7} | "
        f"{r.thread_label:<7} | {r.latency_ms:>13.3f} ms |"
        for r in results
    ]
    return "\n".join([header, *rows])


def format_markdown(
    results: Sequence[BenchmarkResult],
    sweep: Sequence[BenchmarkResult],
    thread_info: ThreadInfo,
    *,
    commit: str | None = None,
    timestamp: datetime | None = None,
) -> str:
    """Render a dated Markdown block ready to append to ``BENCHMARKS.md``."""
    when = (timestamp or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    commit_label = commit or "unknown"
    lines = [
        f"## {when} — commit {commit_label}",
        "",
        f"**Environment:** {platform.machine()}, "
        f"{os.cpu_count()} CPUs. Python {platform.python_version()}, "
        f"NumPy {np.__version__}. portable-attention {__version__}.",
        "",
        f"**BLAS threads (process default, via {thread_info.source}):** "
        f"{thread_info.summary}. Each row below is measured with the thread "
        f"count pinned to its `threads` value.",
        "",
        "### Latency by shape (pinned threads, median over repeats)",
        "",
        _results_table(results),
        "",
        "### Thread sweep (multi-head shape — the cliff stays visible)",
        "",
        _results_table(sweep),
    ]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m portable_attention.benchmark",
        description="Measure reference SDPA latency with pinned BLAS threads.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="BLAS thread count to pin for the suite (default: 1).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Timed calls per shape; the median is reported (default: 20).",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="Commit hash to stamp into the report header.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: print a dated Markdown benchmark block to stdout."""
    args = _build_parser().parse_args(argv)
    threads: int = args.threads
    repeats: int = args.repeats
    commit: str | None = args.commit
    results = run_suite(threads=threads, repeats=repeats)
    sweep = run_thread_sweep(repeats=repeats)
    print(format_markdown(results, sweep, describe_threads(), commit=commit))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
