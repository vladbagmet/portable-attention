# portable-attention

Portable, CUDA-independent attention kernels with pluggable backends —
CPU-first and correctness-obsessed, with a hard portability floor: every
release must run end-to-end on an $80 computer.

**Status: pre-MVP (M0 walking skeleton).** Direction lives in `VISION.md` and
`ROADMAP.md`; the grounding research (a verified gap analysis of the non-CUDA AI
compute landscape, 2026-07-19) is in `RESEARCH.md`. A CPU reference
`scaled_dot_product_attention` and its test harness are in place; the public API
and packaging are still being frozen (see roadmap M0.3).

## Development

```sh
uv venv && uv pip install -e ".[dev]"
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

Everything runs on CPU with no GPU required — that is the point.

## Why

Attention is where the CUDA moat is deepest: PyTorch on Apple Silicon still has
no FlashAttention-class training path, AMD's consumer RDNA cards get attention
kernels years late, and every fast path assumes one vendor. The bet: a small,
portable, drop-in attention layer with a clean backend contract — CPU reference
first, vendor backends attached one at a time — is the highest-leverage wedge a
small project can drive into that moat.

## License

Apache-2.0
