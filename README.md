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
`VulkanDevice` (see
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

print(available_backends())  # ['fused', 'reference']
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

The next backend on the roadmap (M2) is a portable GPU path built on **Vulkan
(V3DV)**, so the package ships a detector that reports what Vulkan hardware the
host actually exposes:

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

The context owns what it creates: closing it frees buffers the caller forgot,
then destroys the device and the instance. Both the context and its buffers work
as context managers, and `close()`/`free()` are idempotent. Anything that fails
raises `VulkanError` naming the entry point and the `VkResult`. No shader runs
yet — this is the allocation and transfer path the kernel will sit on.

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
sides bill the same budget. No backend consumes the policy yet.

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
