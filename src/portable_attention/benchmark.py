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

Passing more than one backend measures the same shapes on each of them and
prints a comparison table with speedups against the first one, which is how the
cross-backend blocks in ``BENCHMARKS.md`` are produced::

    python -m portable_attention.benchmark \\
        --threads default --backends reference,fused,vulkan
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
    "format_comparison",
    "format_markdown",
    "run_comparison",
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
    backend: str = "reference"

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
    return BenchmarkResult(shape, resolved.name, threads, latency, repeats, backend)


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


def run_comparison(
    *,
    backends: Sequence[str],
    threads: int | None = None,
    dtype: str = "float32",
    repeats: int = 20,
    shapes: Sequence[Shape] = DEFAULT_SHAPES,
) -> list[BenchmarkResult]:
    """Measure every shape on every backend at one thread setting.

    Shapes are the outer loop so each backend meets a shape at roughly the same
    machine state; the returned list is flat and carries the backend name on
    every result, which is what :func:`format_comparison` groups by.

    Args:
        backends: Registered backend names. The first one is the baseline the
            speedup columns are computed against.
        threads: BLAS thread count pinned for every measurement, or ``None``
            (the default) to measure the process default policy.
        dtype: Floating dtype of the inputs.
        repeats: Timed calls per measurement; the median is reported.
        shapes: Shapes to measure.
    """
    if not backends:
        raise ValueError("backends must name at least one backend")
    return [
        benchmark_shape(
            shape,
            threads=threads,
            dtype=dtype,
            repeats=repeats,
            backend=backend,
        )
        for shape in shapes
        for backend in backends
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


def _markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    right_aligned: Sequence[bool],
) -> str:
    """Render a padded Markdown table; ``right_aligned`` marks numeric columns."""
    widths = [
        max([len(headers[col]), *(len(row[col]) for row in rows)])
        for col in range(len(headers))
    ]

    def line(cells: Sequence[str]) -> str:
        parts = [
            cell.rjust(widths[col]) if right_aligned[col] else cell.ljust(widths[col])
            for col, cell in enumerate(cells)
        ]
        return "| " + " | ".join(parts) + " |"

    rule = (
        "|"
        + "|".join(
            "-" * (widths[col] + 1) + (":" if right_aligned[col] else "-")
            for col in range(len(headers))
        )
        + "|"
    )
    return "\n".join([line(headers), rule, *(line(row) for row in rows)])


def _speedup(baseline_ms: float | None, candidate_ms: float | None) -> str:
    """Format ``baseline / candidate`` as a speedup, or ``n/a`` if unusable."""
    if baseline_ms is None or candidate_ms is None or candidate_ms <= 0.0:
        return "n/a"
    return f"{baseline_ms / candidate_ms:.2f}x"


def format_comparison(
    results: Sequence[BenchmarkResult],
    *,
    baseline: str | None = None,
) -> str:
    """Render one row per shape with a latency column for each backend.

    Every non-baseline backend also gets a speedup column against the baseline
    (the first backend measured unless ``baseline`` names another one). Missing
    measurements are printed as ``n/a`` rather than dropped, so a backend that
    could not run a shape stays visible in the table.

    Args:
        results: Measurements from :func:`run_comparison` (or any list of
            results carrying backend names).
        baseline: Backend the speedups are relative to.
    """
    if not results:
        return "_no measurements_"
    names: list[str] = []
    for result in results:
        if result.backend not in names:
            names.append(result.backend)
    reference_name = names[0] if baseline is None else baseline
    if reference_name not in names:
        raise ValueError(f"baseline {reference_name!r} was not measured; have {names}")
    grouped: dict[tuple[Shape, str, str], dict[str, float]] = {}
    for result in results:
        key = (result.shape, result.dtype, result.thread_label)
        grouped.setdefault(key, {})[result.backend] = result.latency_ms
    others = [name for name in names if name != reference_name]
    headers = [
        "shape (B,H,S,D)",
        "dtype",
        "threads",
        *names,
        *(f"{name} vs {reference_name}" for name in others),
    ]
    right_aligned = [False, False, False, *(True for _ in headers[3:])]
    rows: list[list[str]] = []
    for (shape, dtype, thread_label), latencies in grouped.items():
        base = latencies.get(reference_name)
        cells = [str(tuple(shape)), dtype, thread_label]
        for name in names:
            measured = latencies.get(name)
            cells.append("n/a" if measured is None else f"{measured:.3f} ms")
        cells.extend(_speedup(base, latencies.get(name)) for name in others)
        rows.append(cells)
    return _markdown_table(headers, rows, right_aligned)


def format_markdown(
    results: Sequence[BenchmarkResult],
    sweep: Sequence[BenchmarkResult],
    thread_info: ThreadInfo,
    *,
    commit: str | None = None,
    timestamp: datetime | None = None,
) -> str:
    """Render a dated Markdown block ready to append to ``BENCHMARKS.md``."""
    lines = [
        *_report_header(
            thread_info,
            commit=commit,
            timestamp=timestamp,
            threads_note=(
                "Each row below is measured with the thread count pinned to "
                "its `threads` value."
            ),
        ),
        "### Latency by shape (pinned threads, median over repeats)",
        "",
        _results_table(results),
        "",
        "### Thread sweep (multi-head shape — the cliff stays visible)",
        "",
        _results_table(sweep),
    ]
    return "\n".join(lines)


def _report_header(
    thread_info: ThreadInfo,
    *,
    commit: str | None,
    timestamp: datetime | None,
    threads_note: str,
) -> list[str]:
    """The dated heading, environment line and thread line shared by blocks."""
    when = (timestamp or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    return [
        f"## {when} — commit {commit or 'unknown'}",
        "",
        f"**Environment:** {platform.machine()}, "
        f"{os.cpu_count()} CPUs. Python {platform.python_version()}, "
        f"NumPy {np.__version__}. portable-attention {__version__}.",
        "",
        f"**BLAS threads (process default, via {thread_info.source}):** "
        f"{thread_info.summary}. {threads_note}",
        "",
    ]


def _parse_threads(value: str) -> int | None:
    """Parse a ``--threads`` argument: an integer, or ``default`` for unpinned."""
    if value.strip().lower() == "default":
        return None
    try:
        threads = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a thread count or 'default', got {value!r}"
        ) from None
    if threads < 1:
        raise argparse.ArgumentTypeError(f"thread count must be >= 1, got {threads}")
    return threads


def _parse_backends(value: str) -> list[str]:
    """Parse a comma-separated backend list, preserving order."""
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError("expected at least one backend name")
    return names


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m portable_attention.benchmark",
        description="Measure SDPA latency with a recorded BLAS thread policy.",
    )
    parser.add_argument(
        "--backends",
        type=_parse_backends,
        default=["reference"],
        help=(
            "Comma-separated backend names. One backend prints the pinned "
            "suite plus a thread sweep; several print a comparison table with "
            "speedups against the first (default: reference)."
        ),
    )
    parser.add_argument(
        "--threads",
        type=_parse_threads,
        default=1,
        help=(
            "BLAS thread count to pin for every measurement, or 'default' to "
            "measure the process default policy (default: 1)."
        ),
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
    threads: int | None = args.threads
    repeats: int = args.repeats
    commit: str | None = args.commit
    backends: list[str] = args.backends
    thread_info = describe_threads()
    if len(backends) == 1:
        results = run_suite(threads=threads, repeats=repeats, backend=backends[0])
        sweep = run_thread_sweep(repeats=repeats, backend=backends[0])
        print(format_markdown(results, sweep, thread_info, commit=commit))
        return 0
    comparison = run_comparison(backends=backends, threads=threads, repeats=repeats)
    label = "the process default" if threads is None else f"{threads}"
    print(
        "\n".join(
            [
                *_report_header(
                    thread_info,
                    commit=commit,
                    timestamp=None,
                    threads_note=(
                        f"Every row below is measured at {label} thread "
                        f"count, median over {repeats} repeats."
                    ),
                ),
                f"### Backend comparison (baseline: `{backends[0]}`)",
                "",
                format_comparison(comparison),
            ]
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
