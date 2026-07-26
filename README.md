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
path on hardware where the fast vendor path is missing. The CPU reference
implements the forward, non-dropout computation; `dropout_p` and `enable_gqa`
are accepted for signature compatibility but must be left at their defaults (a
non-default value raises `NotImplementedError` rather than being silently
ignored). The public API is the set of names re-exported from the top-level
package — `scaled_dot_product_attention`, `__version__`, and the backend-registry
helpers `get_backend`, `available_backends`, `register_backend`, and the
`SdpaBackend` protocol (see [Backends](#backends)); everything else is internal
and may change.

## Backends

Attention runs through a small pluggable backend registry. The CPU `reference`
implementation is the correctness oracle and is always available; every future
backend is validated against its output. The public
`scaled_dot_product_attention` dispatches to the `"auto"` backend (today that is
always `reference`). To resolve a specific backend explicitly:

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

## CPU performance and BLAS threads

Multi-head attention runs many small per-slice matrix multiplies through NumPy's
batched `matmul`. Under the default OpenBLAS policy (one thread per core), the
thread-synchronization overhead of those tiny GEMMs can dominate and make
multi-head workloads **much** slower than with a single BLAS thread.

The `fused` backend handles this for you: for batched (multi-slice) inputs it
pins BLAS to one thread for its compute region (via `threadpoolctl` when
installed) and computes in native precision, so multi-head attention does not
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
