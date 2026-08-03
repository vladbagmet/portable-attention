#!/usr/bin/env bash
# Metal sign-off report, run by a human on a Mac.
#
# CI has no Apple GPU, so a change to the Metal backend is only ever
# "compiles" until someone runs this on real hardware and pastes the output
# into the pull request. The report is deliberately plain text so it can go in
# a comment unedited.
#
#   uv venv && uv pip install -e ".[dev,metal]"
#   ./scripts/verify-metal.sh
#
# Set PYTHON to point at a different interpreter. Exit status is 0 only when a
# device was found, the runtime compiler worked, and conformance did not fail.
set -euo pipefail

PYTHON="${PYTHON:-python3}"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "verify-metal.sh runs on macOS only (this host: $(uname -s))." >&2
  exit 2
fi

exec "$PYTHON" - "$@" <<'PY'
import platform
import sys

REPORT = []


def line(text=""):
    REPORT.append(text)


def finish(status):
    print("\n".join(REPORT))
    sys.exit(status)


line("## Metal verification report")
line()
line(f"- host: macOS {platform.mac_ver()[0]} ({platform.machine()})")
line(f"- python: {platform.python_version()}")

try:
    import Metal  # pyobjc-framework-Metal
except ImportError as exc:
    line(f"- pyobjc-framework-Metal: MISSING ({exc})")
    line()
    line('Install the extra first: `uv pip install -e ".[metal]"`.')
    finish(2)

device = Metal.MTLCreateSystemDefaultDevice()
if device is None:
    line("- device: NONE (MTLCreateSystemDefaultDevice returned nil)")
    finish(1)

families = [
    name
    for name, value in (
        ("apple7", Metal.MTLGPUFamilyApple7),
        ("apple8", Metal.MTLGPUFamilyApple8),
        ("apple9", Metal.MTLGPUFamilyApple9),
        ("metal3", Metal.MTLGPUFamilyMetal3),
    )
    if device.supportsFamily_(value)
]
line(f"- device: {device.name()}")
line(f"- families: {', '.join(families) or 'none reported'}")
line()

# The kernel is compiled from source at runtime; that is the shipping path, so
# verify it here rather than trusting an offline `xcrun metal` build.
source = """
#include <metal_stdlib>
using namespace metal;
kernel void scale_by_two(device const float *in [[buffer(0)]],
                         device float *out [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    out[i] = in[i] * 2.0f;
}
"""
library, error = device.newLibraryWithSource_options_error_(source, None, None)
if library is None:
    line(f"- runtime shader compile: FAIL ({error})")
    finish(1)
pipeline, error = device.newComputePipelineStateWithFunction_error_(
    library.newFunctionWithName_("scale_by_two"), None
)
if pipeline is None:
    line(f"- pipeline state: FAIL ({error})")
    finish(1)
line("- runtime shader compile: ok (newLibraryWithSource, no Xcode needed)")
line()

max_threads = device.maxThreadsPerThreadgroup()
line("| Limit | value |")
line("| --- | --- |")
line(f"| shared / threadgroup memory | {device.maxThreadgroupMemoryLength()} B |")
line(f"| max invocations per workgroup | {max_threads.width} |")
line(f"| SIMD / subgroup width | {pipeline.threadExecutionWidth()} |")
line(f"| max buffer / storage range | {device.maxBufferLength()} B |")
line(f"| unified memory | {'yes' if device.hasUnifiedMemory() else 'no'} |")
line()

status = 0
try:
    from portable_attention import assert_conforms, available_backends, get_backend
    from portable_attention.benchmark import DEFAULT_SHAPES, benchmark_shape
except ImportError as exc:
    line(f"- conformance: SKIPPED (portable_attention not importable: {exc})")
    finish(2)

if "metal" not in available_backends():
    line("- conformance: SKIPPED (no `metal` backend in this checkout)")
    line(f"- backends seen: {', '.join(available_backends())}")
    finish(status)

try:
    assert_conforms(get_backend("metal"))
except AssertionError as exc:
    line("- conformance vs reference oracle: FAIL")
    line()
    line("```")
    line(str(exc))
    line("```")
    status = 1
else:
    line("- conformance vs reference oracle: PASS (full case matrix)")

line()
line("| shape (B,H,S,D) | backend | latency (median) |")
line("| --- | --- | ---: |")
for shape in DEFAULT_SHAPES:
    for backend in ("reference", "metal"):
        result = benchmark_shape(shape, backend=backend, threads=None)
        dims = ",".join(str(d) for d in shape)
        line(f"| ({dims}) | {backend} | {result.latency_ms:.3f} ms |")

finish(status)
PY
