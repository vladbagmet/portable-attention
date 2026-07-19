# portable-attention

Portable, CUDA-independent attention kernels with pluggable backends —
CPU-first, correctness-obsessed, tested end-to-end on a Raspberry Pi 5.

**Status: pre-MVP.** Vision and roadmap are being drafted; see `VISION.md` and
`ROADMAP.md` once they land. The grounding research (a verified gap analysis of
the non-CUDA AI compute landscape, 2026-07-19) is in `RESEARCH.md`.

## Why

Attention is where the CUDA moat is deepest: PyTorch on Apple Silicon still has
no FlashAttention-class training path, AMD's consumer RDNA cards get attention
kernels years late, and every fast path assumes one vendor. The bet: a small,
portable, drop-in attention layer with a clean backend contract — CPU reference
first, vendor backends attached one at a time — is the highest-leverage wedge a
small project can drive into that moat.

## Transparency

This project is developed largely by an autonomous coding agent operating under
the direction of Vlad Bagmet ([@vladbagmet](https://github.com/vladbagmet)),
with capacity-bounded nightly build shifts, journaled decisions, and CI-gated
merges. Issues and PRs from humans are very welcome and get human attention.

## License

Apache-2.0
