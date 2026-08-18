# portable-attention

Portable, CUDA-independent attention kernels with pluggable backends —
CPU-first and correctness-obsessed, with a hard portability floor: every
release must run end-to-end on an $80 computer.

**Status: pre-MVP (M0 walking skeleton).** Direction lives in `VISION.md` and
`ROADMAP.md`; the grounding research (a verified gap analysis of the non-CUDA AI
compute landscape, 2026-07-19) is in `RESEARCH.md`. A CPU reference
`scaled_dot_product_attention` with a full correctness test harness is in place
and the public API is frozen; only NumPy is required.

## Install

```sh
pip install portable-attention
```

The only runtime dependency is NumPy, so it installs and runs anywhere NumPy
does — no GPU, no CUDA, no vendor toolchain.

## Quickstart

```python
import numpy as np
from portable_attention import scaled_dot_product_attention

# One attention head: 4 query positions, 6 keys/values, embedding dim 8.
rng = np.random.default_rng(0)
query = rng.standard_normal((4, 8))
key = rng.standard_normal((6, 8))
value = rng.standard_normal((6, 8))

out = scaled_dot_product_attention(query, key, value)
print(out.shape)  # (4, 8)

# Causal (autoregressive) masking and batching work the same way:
batched_q = rng.standard_normal((2, 4, 8))  # (batch, seq, dim)
batched_k = rng.standard_normal((2, 4, 8))
batched_v = rng.standard_normal((2, 4, 8))
causal_out = scaled_dot_product_attention(
    batched_q, batched_k, batched_v, is_causal=True
)
print(causal_out.shape)  # (2, 4, 8)
```

The signature matches
[`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
(`query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *,
scale=None, enable_gqa=False`), so it can act as a drop-in for the inference
path on hardware where the fast vendor path is missing. The CPU backends
implement the forward, non-dropout computation, including grouped-query
attention: pass `enable_gqa=True` when `key`/`value` carry fewer heads than
`query` (their heads are repeated to match). `dropout_p` is accepted for
signature compatibility but must be left at `0.0` (a non-zero value raises
`NotImplementedError` rather than being silently ignored). The public API is the
set of names re-exported from the top-level package (its `__all__`):
`scaled_dot_product_attention`, `__version__`, the backend-registry helpers
`get_backend`, `available_backends`, `register_backend`, and the `SdpaBackend`
protocol, the conformance kit `assert_conforms`, `check_backend`,
`conformance_cases`, `ConformanceCase`, and `ConformanceResult`, and the Vulkan
detection helpers `detect_vulkan`, `vulkan_available`, `VulkanCapability`, and
`VulkanDevice`, and the Vulkan backend `VulkanAttention` with its
`register_vulkan_backend` (see
[Backends](#backends)); everything else is internal and may change.

## Backends

Attention runs through a small pluggable backend registry. The CPU `reference`
implementation is the correctness oracle and is always available; every future
backend is validated against its output. The public
`scaled_dot_product_attention` dispatches to the `"auto"` backend, a shape-aware
policy that routes batched (multi-slice) inputs to the fast `fused` backend and
single-slice inputs to the `reference` oracle — so multi-head workloads get the
fast path automatically. To resolve a specific backend explicitly:

```python
from portable_attention import available_backends, get_backend

print(available_backends())  # ['fused', 'reference'] (+ 'vulkan', see below)
out = get_backend("reference")(query, key, value)
```

The `fused` backend computes the same forward attention as `reference` but in
the input's native precision, pinning BLAS to a single thread for batched
(multi-slice) inputs. It matches the reference to floating tolerance and is much
faster for multi-head workloads at default OpenBLAS threads (see [CPU
performance and BLAS threads](#cpu-performance-and-blas-threads)):

```python
out = get_backend("fused")(query, key, value)
```

A backend is any callable matching the `SdpaBackend` protocol (the same
signature as `scaled_dot_product_attention`); register your own with
`register_backend(name, backend)`.

### Vulkan device detection

The portable GPU path is built on **Vulkan (V3DV)**, and it starts with a
detector that reports what Vulkan hardware the host actually exposes — the same
check that decides whether the backend below registers:

```python
from portable_attention import detect_vulkan, vulkan_available

if vulkan_available():
    ...  # a Vulkan-backed backend could register here

cap = detect_vulkan()
print(cap.available, cap.loader, cap.reason)
for device in cap.devices:
    print(device.name, device.api_version, device.compute)
```

`detect_vulkan()` locates the Vulkan ICD loader (`libvulkan`), creates a
throwaway instance through it, and enumerates the physical devices, recording
each device's name, Vulkan API version, and whether any of its queue families
advertises `VK_QUEUE_COMPUTE_BIT`. `available` is `True` when at least one
device can run compute; otherwise `reason` says which step came up empty. For a
quick summary without walking `cap.devices`, use `cap.device_count` and
`cap.device_names`.
On a board with V3DV that looks like:

```text
True libvulkan.so.1 None
V3D 7.1.7.0 1.2.289 True
llvmpipe (LLVM 15.0.6, 128 bits) 1.3.289 True
```

The loader is called with `ctypes`, so detection adds no runtime dependency —
no Python Vulkan binding is needed or used. It opens no device and submits no
GPU work; a host without Vulkan reports `available=False` rather than failing.

### Opening a Vulkan device

Detection says a device is there; `VulkanContext` opens it. The context creates
a logical device with one compute queue and allocates storage buffers whose
memory stays mapped, so arrays move in and out as plain copies:

```python
import numpy as np

from portable_attention import VulkanContext

q = np.random.default_rng(0).standard_normal((4, 64), dtype=np.float32)

with VulkanContext.open() as ctx:
    print(ctx.device_name, ctx.queue_family_index)  # V3D 7.1.7.0 0
    print(ctx.tile_limits.shared_memory_bytes)  # 16384 on V3D
    with ctx.allocate(q.nbytes) as buf:
        buf.write(q)
        same = buf.read(q.dtype, q.shape)
```

`open()` takes the first compute-capable device by default; `device_index=`
selects one from the enumeration order that `detect_vulkan()` reports. Memory is
chosen from the types the buffer allows, requiring `HOST_VISIBLE | HOST_COHERENT`
and preferring `DEVICE_LOCAL` as well — on unified-memory parts such as V3D that
is the same heap, so no staging copy is needed. Buffers are created with
`STORAGE_BUFFER` usage, which is what a compute shader binds.

`tile_limits` reports what the device answers for
`maxComputeSharedMemorySize` and its workgroup-size limits, as the
`DeviceLimits` that `plan_tiles` takes; `subgroupSize` needs the Vulkan 1.1
properties query this module does not make yet, so the width is the
conservative `SUBGROUP_WIDTH_FALLBACK` and only affects which of two equally
large tiles is preferred.

The context owns what it creates: closing it frees buffers the caller forgot,
then destroys the device and the instance. Both the context and its buffers work
as context managers, and `close()`/`free()` are idempotent. Anything that fails
raises `VulkanError` naming the entry point and the `VkResult`.

### Running a compute shader

`compute_pipeline()` turns a SPIR-V module into something dispatchable. The
shader's storage buffers occupy bindings `0..n-1` of descriptor set 0, in the
order they are passed to `dispatch()`:

```python
import struct
from pathlib import Path

import numpy as np

from portable_attention import VulkanContext

data = np.arange(1000, dtype=np.float32)
spirv = Path("scale.spv").read_bytes()  # glslangValidator -V -S comp ...

with VulkanContext.open() as ctx:
    src = ctx.allocate(data.nbytes)
    dst = ctx.allocate(data.nbytes)
    src.write(data)
    with ctx.compute_pipeline(spirv, buffer_count=2, push_constant_bytes=8) as pipe:
        pipe.dispatch(
            [src, dst],
            groups=(data.size + 63) // 64,  # workgroups, not invocations
            push_constants=struct.pack("<If", data.size, 2.0),
        )
    print(dst.read(np.float32, data.shape))  # data * 2.0
```

`groups` is the number of workgroups — divide the problem size by the shader's
`local_size` yourself, as Vulkan does. Small parameters travel as push
constants, capped at the 128 bytes every Vulkan implementation guarantees so a
kernel stays portable. `dispatch()` is synchronous: it submits, waits on a fence
(`timeout_s` bounds the wait), and inserts a barrier making shader writes
visible to the host, so a buffer can be read immediately afterwards. The command
buffer and fence are created once and reused, so dispatching again in a loop
only re-records.

SPIR-V is checked for word alignment and its magic number before the driver sees
it, and a module compiled for the opposite byte order is named as such. A
pipeline is owned by its context like a buffer, and a failure part-way through
creating one destroys the objects already made.

Values that a shader needs at compile time — a workgroup size, a tile shape, the
length of a shared array — go in as specialization constants rather than push
constants:

```python
with ctx.compute_pipeline(
    spirv,
    buffer_count=2,
    push_constant_bytes=4,
    specialization={0: 32, 1: 64, 2: 0.125},  # layout(constant_id = N)
) as pipe:
    ...
```

The driver compiles the module with those values baked in, so one SPIR-V file
covers several tile shapes without a recompile of the GLSL. Each constant is a
`bool`, `int` or `float` and fills one 4-byte slot; ids the shader declares and
the mapping omits keep their GLSL defaults.

### Tile sizing

A blocked (flash-style) attention kernel streams the key/value sequence in
tiles so the full score matrix never has to exist. How large a tile may be is a
property of the device, not of the algorithm, so the policy lives in one place
and takes the device numbers as data:

```python
from portable_attention import plan_tiles
from portable_attention.tiling import V3D_LIMITS

plan = plan_tiles(head_dim=64, dtype_bytes=4, limits=V3D_LIMITS)
print(plan.block_q, plan.block_k, plan.threads_per_group)  # 16 16 256
print(plan.shared_memory_bytes, plan.k_tiles(4096))  # 13440 256
```

`DeviceLimits` carries the three numbers a backend queries from its API:
shared/threadgroup memory per workgroup, maximum invocations per workgroup, and
SIMD/subgroup width. `plan_tiles` returns the largest tile that fits, preferring
a full workgroup, then a wide key tile. Pass `seq_len_q` / `seq_len_k` when they
are known and the blocks are capped at the next power of two at or above the
length, so a short input stops drawing a tile it can never fill. The layout being
priced is documented on `shared_memory_bytes_for`; a kernel calls it too, so both
sides bill the same budget. The output accumulator is not in that budget — it is
register-resident, `plan.accumulators_per_invocation` values per invocation. The
Vulkan backend plans every dispatch through it.

### The blocked algorithm, written down

A device kernel that returns a wrong number does not say whether tiling,
masking, the online-softmax rescale or the accumulator layout was at fault.
`blocked_attention` is the same algorithm in NumPy, driven by a `TilePlan`, so a
kernel can be diffed against a host implementation making identical decisions —
padded trailing tiles, per-row running max and sum, causal tile skipping:

```python
from portable_attention import blocked_attention, plan_tiles
from portable_attention.tiling import V3D_LIMITS

plan = plan_tiles(head_dim=64, dtype_bytes=4, limits=V3D_LIMITS)
out = blocked_attention(q, k, v, plan, is_causal=True)  # (n, seq, 64) each
```

It is a specification to port from, not a backend to serve traffic with: it
walks tiles in Python and is slower than either CPU backend. Inputs are the flat
`(n, seq, head_dim)` stack a dispatch sees, so reshape `(*, H, S, E)` down to it
first; masks, GQA and dropout stay with the real backends.

### Vulkan attention backend

Where the host has a compute-capable Vulkan device, importing the package
registers a `"vulkan"` backend that runs the blocked-attention shader:

```python
from portable_attention import available_backends, get_backend

print(available_backends())  # ['fused', 'reference', 'vulkan']
out = get_backend("vulkan")(query, key, value, is_causal=True)
```

The kernel implements unmasked and causal float32 attention where query, key
and value share a head dimension. Every other call the contract allows — masks,
grouped-query attention, `float64`/`float16`, a value width that differs from
the key width, a head dimension no tile plan fits — is forwarded to the `fused`
CPU backend, so the backend passes the full conformance kit rather than only
the cases it accelerates. Invalid input is forwarded too, so the error you get
is the documented one.

The device is opened on the first call it can serve, and the context, one
pipeline per tile shape and four buffers (which grow with the shapes they see)
are reused across calls. Tiles are planned against the limits the open device
reports, so a part with room to spare gets larger tiles than the Vulkan
minimums would allow; `VulkanAttention(limits=...)` pins them instead, and
`backend.limits` answers the minimums until a device has been opened. If the
device fails mid-run
the backend warns once and serves the rest of the process from the CPU: a
wedged GPU costs throughput, not correctness.

`"auto"` does not route to the device yet — that selection needs benchmark
numbers, so for now the device is opt-in through `get_backend("vulkan")`.
Setting `PORTABLE_ATTENTION_DISABLE_VULKAN` to any non-empty value skips both
the detection probe and the registration.

### Conformance kit

The portability promise is *developer parity*: code written against
`scaled_dot_product_attention` behaves the same on every backend. The shared
conformance kit makes that executable — it checks a backend against the
`reference` oracle across the full contract matrix (leading dims, non-square
scores, all supported dtypes, `scale`, causal and boolean/additive masks,
fully-masked rows, and grouped-query attention), verifying matching shape,
preserved `query` dtype, finite output, and exact zeros for fully-masked rows.
Every registered backend is held to it in CI; run it against your own backend
before relying on it:

```python
from portable_attention import assert_conforms, check_backend

assert_conforms(my_backend)  # raises with the failing cases, or passes
results = check_backend(my_backend)  # structured per-case results to inspect
```

`conformance_cases()` returns the case list as data, so you can parametrize a
test suite over it (`ConformanceCase` / `ConformanceResult` are the public
types).

## CPU performance and BLAS threads

Multi-head attention runs many small per-slice matrix multiplies through NumPy's
batched `matmul`. Under the default OpenBLAS policy (one thread per core), the
thread-synchronization overhead of those tiny GEMMs can dominate and make
multi-head workloads **much** slower than with a single BLAS thread.

The `fused` backend handles this for you (and `"auto"` selects it automatically
for batched inputs): for batched (multi-slice) inputs it pins BLAS to one thread
for its compute region (via `threadpoolctl` when installed) and computes in
native precision, so multi-head attention does not
cliff at default OpenBLAS threads. Prefer it for performance:

```python
from portable_attention import get_backend

out = get_backend("fused")(query, key, value)
```

If you call the `reference` backend directly (the correctness oracle, which
upcasts to float64), cap the BLAS thread count yourself for multi-head work:

```sh
OPENBLAS_NUM_THREADS=1 python your_script.py
```

Dated, reproducible latency numbers (with the thread configuration recorded
next to each result) live in `BENCHMARKS.md`. Regenerate them with the built-in
harness:

```sh
python -m portable_attention.benchmark --threads 1 --commit "$(git rev-parse --short HEAD)"
```

Installing `threadpoolctl` lets the harness detect and pin the real BLAS thread
count precisely; without it, it falls back to `OPENBLAS_NUM_THREADS`.

## Development

```sh
uv venv && uv pip install -e ".[dev]"
./scripts/check.sh   # the full gate CI runs: lint, format, types, security, tests
```

Everything runs on CPU with no GPU required — that is the point. See
`CONTRIBUTING.md` for the gate details.

The package version has a single source of truth: `__version__` in
`src/portable_attention/__init__.py`. The build backend reads it from there, so
it is never duplicated in `pyproject.toml`.

## Why

Attention is where the CUDA moat is deepest: PyTorch on Apple Silicon still has
no FlashAttention-class training path, AMD's consumer RDNA cards get attention
kernels years late, and every fast path assumes one vendor. The bet: a small,
portable, drop-in attention layer with a clean backend contract — CPU reference
first, vendor backends attached one at a time — is the highest-leverage wedge a
small project can drive into that moat.

## License

Apache-2.0
