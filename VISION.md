# Vision

## Thesis

Attention is where the CUDA moat is deepest, and that moat is the single
biggest reason "AI only runs well on NVIDIA" is still true. **portable-attention**
is a small, drop-in attention/SDPA layer with a clean, pluggable backend
contract: one stable API, a correctness-obsessed CPU reference that runs
anywhere, and vendor backends attached one at a time exactly where the verified
gaps are (Apple Metal training-grade forward+backward, ROCm/Triton for consumer
RDNA, Vulkan for everything else).

The bet (grounded in `RESEARCH.md`, a verified 2026-07-19 gap analysis): the
highest-leverage wedge a small project can drive into the CUDA moat is not a new
framework but a *portable primitive with a testable contract* — something a real
engineer can `pip install` and benefit from today, on hardware they already own.

## Why now

- PyTorch on Apple Silicon still has **no FlashAttention-class training path**
  (no dedicated SDPA backward; MPS falls back to the slow math path).
- AMD consumer RDNA gets attention kernels a generation late, with per-SKU
  holes and cases where the "fast" backend is *slower* than the math fallback.
- Vulkan-class consumer GPUs and ARM/edge CPUs have **no first-class attention
  story at all**.
- Non-CUDA developer parity is itself an unclosed gap: contributors without an
  NVIDIA GPU often cannot even run the tests. A CPU-first project fixes that by
  construction.

## What we are (principles)

1. **Correctness first.** Kernels live or die on numerical accuracy. Every
   behavioral change is tested against a reference; shape/dtype edge cases are
   first-class.
2. **A portability floor, enforced.** Every release runs end-to-end on an $80
   computer (a Raspberry Pi 5 is the canonical low-power floor). CPU/NEON is the
   reference backend, never an afterthought.
3. **A minimal backend contract.** Backends are pluggable behind one stable API
   (compatible with `torch.nn.functional.scaled_dot_product_attention` where
   possible). Adding a vendor backend must not require touching the core.
4. **Honest benchmarks.** Real numbers, real hardware, real commit hashes — or
   nothing.
5. **MVP bias.** Something installable and useful beats a grand architecture.

## Non-goals

- **Not a training framework.** We provide a primitive, not a model zoo or a
  new autograd engine.
- **Not chasing peak NVIDIA FLOPS.** We compete on *portability and coverage*,
  not on beating cuDNN on an H100.
- **No vendor lock-in in the core.** No backend may leak into the public API.
- **No unverified claims.** Backend maturity we haven't measured is not
  advertised (see the caveats in `RESEARCH.md`).
- **Mojo** is allowed only after a documented spike proves its target support;
  until then the implementation languages are Python and/or Rust.
