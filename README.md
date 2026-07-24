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

The signature mirrors
[`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
(`query, key, value, attn_mask=None, is_causal=False, scale=None`), so it can
act as a drop-in on hardware where the fast vendor path is missing. Only the
names re-exported from the top-level package — `scaled_dot_product_attention`
and `__version__` — are public; everything else is internal and may change.

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
