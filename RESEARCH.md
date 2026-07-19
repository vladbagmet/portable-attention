# Research: the non-CUDA attention gap

Snapshot date: **2026-07-19.** This document motivates portable-attention with
a verified gap analysis of the non-CUDA AI compute landscape. Method: claims
were extracted from ~24 primary sources and the load-bearing ones passed
3-vote adversarial verification. Labels: **✅ verified** (survived that gate),
**◐ corroborated** (multiple current sources), **○ reported** (single source).
Counts and statuses drift quickly in this space — treat them as a snapshot.

## The problem

Attention is the performance-critical primitive of modern AI, and fast
attention (FlashAttention-class fused kernels) exists first — and often only —
for NVIDIA CUDA. Off NVIDIA, an engineer hits missing kernels, uneven per-SKU
coverage, and vendor-specific forks and flags. There is no portable layer.

## What's broken today

**Apple Silicon (PyTorch/MPS) — ✅ verified**

- No dedicated SDPA backward pass: gradients fall back to the slow composite
  "math" path, and the restriction disables the fast forward kernel during
  training too ([pytorch#179294](https://github.com/pytorch/pytorch/issues/179294));
  fix PRs sat unmerged for months. There is no FlashAttention-class training
  path on Apple GPUs.
- Operator coverage is still incomplete years after MPS launched
  ([tracking issue #141287](https://github.com/pytorch/pytorch/issues/141287)):
  dense linalg landed piecemeal (2.7–2.10), `linalg_qr` nightly-only at
  snapshot time, complex dtypes failing, no float64 ○.
- MPS SDPA rejects basic input shapes that work on CPU (2-D inputs raise
  IndexError; Hugging Face ships a 4-D unsqueeze workaround).
- The only price-matched study found end-to-end training ~3–4× slower than
  similarly-priced NVIDIA hardware
  ([arXiv:2501.14925](https://arxiv.org/abs/2501.14925), M2-era, medium
  confidence), with nuances (near-parity in one FP32 case; Apple wins when
  NVIDIA VRAM is insufficient).

**AMD consumer RDNA (ROCm) — ◐ corroborated**

- The default Composable-Kernel FlashAttention backend is datacenter-first:
  fails to compile on RDNA4 (Wave64 assumptions vs Wave32), no backward on
  RDNA3, RDNA4 backward only non-deterministic
  ([ROCm/flash-attention](https://github.com/ROCm/flash-attention)).
- Per-SKU holes inside the same generation (gfx1101 eager-only,
  [pytorch#159226](https://github.com/pytorch/pytorch/issues/159226)) and
  measured cases where the FLASH_ATTENTION backend is 2.5× slower than the
  math fallback on a 7900 XTX
  ([pytorch#152595](https://github.com/pytorch/pytorch/issues/152595)).
- History says the lag is structural: upstream SDPA-on-ROCm took ~13 months
  from request to close and shipped CDNA-only, with consumer support blocked
  on Triton compiler work
  ([pytorch#112997](https://github.com/pytorch/pytorch/issues/112997)).
- FlashAttention-3/4 remain NVIDIA-only — AMD runs a kernel generation behind.

**Everything else**

- Vulkan-class consumer GPUs and ARM/edge CPUs have no first-class attention
  story at all; OpenCL never got standardized tensor-core access (typically a
  5–10× loss vs CUDA ○).
- Even flagship multi-backend projects develop CUDA-first: vLLM's full test
  suite is CUDA-only and GPU-less contributors are told to rely on CI ✅ —
  developer-tooling parity is itself an unclosed gap.

## The gaps this project targets

1. **A portable, drop-in attention layer** — one stable API (compatible with
   `torch.nn.functional.scaled_dot_product_attention` where possible) with a
   minimal, testable backend contract.
2. **A correct CPU/NEON reference backend** that runs anywhere — the project's
   enforced portability floor — plus an honest cross-hardware benchmark
   harness.
3. **Vendor backends where the verified gaps are**: Metal (training-grade
   forward+backward), ROCm/Triton for consumer RDNA, Vulkan for everything
   else.
4. **Non-CUDA developer parity**: a test suite contributors can run without
   an NVIDIA GPU.

Validation that the niche is real: third-party efforts (e.g.
[Aule-Attention](https://github.com/AuleTechnologies/Aule-Attention)) emerged
in 2026 targeting exactly this space.

## Caveats

The AMD findings are well-sourced but did not pass the same adversarial gate
as the Apple findings. Intel oneAPI/SYCL, JAX/XLA/IREE, and Mojo/MAX backend
maturity were not assessed (missing evidence, not assessed weakness). One
tempting claim was actively refuted in verification and must not be repeated:
"FP16 GEMM acceleration is largely missing on Apple's stack" (0–3 votes).

## Key sources

[pytorch#179294](https://github.com/pytorch/pytorch/issues/179294) ·
[pytorch#141287](https://github.com/pytorch/pytorch/issues/141287) ·
[pytorch#112997](https://github.com/pytorch/pytorch/issues/112997) ·
[pytorch#152595](https://github.com/pytorch/pytorch/issues/152595) ·
[pytorch#159226](https://github.com/pytorch/pytorch/issues/159226) ·
[ROCm/flash-attention](https://github.com/ROCm/flash-attention) ·
[ROCm/aotriton#16](https://github.com/ROCm/aotriton/issues/16) ·
[arXiv:2501.14925](https://arxiv.org/abs/2501.14925) ·
[AMD Radeon ROCm docs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html) ·
[SemiAnalysis: MI300X training](https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training) ·
[vLLM contributing guide](https://docs.vllm.ai/en/latest/contributing/overview.html)
