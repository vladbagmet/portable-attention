# Open-Source Strategy Report: AI Contributions & the Post-CUDA Opportunity

**Prepared for:** Vlad (GitHub: vladbagmet) · **Date:** 2026-07-19 · **Hardware context:** macOS / Apple Silicon

**Method.** A 106-agent deep-research run (5 search angles → 24 sources fetched → 120 claims extracted → top 25 adversarially verified with 3 independent votes each → 23 confirmed, 2 refuted), merged with live GitHub API data pulled 2026-07-19 and a follow-up verification pass on the AMD/ROCm side. Confidence labels: **✅ verified** (survived 3-vote adversarial verification), **◐ corroborated** (multiple current sources, not adversarially voted), **○ reported** (single source — treat as a lead, not a fact). All counts are snapshots as of 2026-07-19 and will drift.

---

## Part 1 — Where to contribute: ranked

Ranking criteria: (a) genuinely welcoming to new contributors, (b) 2025–2026 momentum/upside, (c) overlap with ML infrastructure / compilers / hardware portability, so contributions build directly toward your Part 2 goal.

### Tier 1 — deep-verified ranking

**1. vLLM** — [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) · 86.6k ★ · 3,036 contributors · 27 open good-first-issues ✅
The most institutionalized onboarding of any project checked. The [official contributing guide](https://docs.vllm.ai/en/latest/contributing/overview.html) routes newcomers to the `good first issue` label, an org-level "Onboarding Tasks" board, and a `new-model` label (111 open). Maintainer responsiveness is *codified policy*, not folklore: reviewers commit to status updates every 2–3 days, you're told to ping after 7 days of silence, and there's an expedited-review email channel. It is the de facto standard open-source LLM inference engine — high visibility per merged PR. (Caveat: SLAs are documented commitments, not measured latencies.)

**2. PyTorch** — [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch) · 101.8k ★ · 6,736 contributors · 62 curated good-first-issues ✅
The deepest skill-building on-ramp toward your CUDA-challenger goal. The [/contribute page](https://github.com/pytorch/pytorch/contribute) listed 62 starter issues on check day, and **38 of 62 (61%) sit in compiler/infra subsystems** — `oncall: distributed` (19), `oncall: pt2` (13), `module: dynamo` (7), JIT (5), Inductor (2). Two mentorship labels are actively applied: `internal ramp-up task` ("high-touch guidance from senior PyTorch folks", 12 open) and `OSS contribution wanted` (12 open). Bonus: the MPS backend gaps documented in Part 2 are *PyTorch issues* — you can contribute fixes for the exact bottlenecks you care about, on hardware you own.

**3. ExecuTorch** — [github.com/pytorch/executorch](https://github.com/pytorch/executorch) · 30 open good-first-issues ✅
PyTorch's edge/on-device runtime with an actively curated [new-contributor program](https://docs.pytorch.org/executorch/stable/new-contributor-guide.html) (dedicated org project board; ~12 GFIs closed May–Jul 2026). Its backend ecosystem is overwhelmingly **non-CUDA**: XNNPACK, CoreML, MPS/Metal, Vulkan, Qualcomm, MediaTek, ARM Ethos-U, Samsung, NXP, OpenVINO — in production at Meta (Ray-Ban glasses, Quest 3, Instagram, WhatsApp). June-2026 starter issues were largely **MLX-backend op handlers** — Apple-Silicon work, ready-labeled. (Caveats: GFI additions are bursty; the legacy MPS backend is being deprecated for v1.4.0 in favor of Metal/MLX paths.)

**4. vllm-metal** — [github.com/vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal) · 1.5k ★ · 89.7% Python · Apache-2.0 ✅
Community-maintained hardware plugin running vLLM on Apple Silicon (MLX as primary compute backend + prebuilt Metal kernels), under the official vllm-project org and listed in vLLM's hardware-plugin docs. Released v0.3.0.dev **two days before this check** — it moves fast, it's small (~448 commits), and it's the most direct Python entry into non-CUDA serving infrastructure. GFI funnel exists but is small (2 open, 7 closed-as-completed); some labeled items are user bug reports rather than curated tasks (2–1 verification vote — the one soft spot in this ranking).

**5. tinygrad** — [github.com/tinygrad/tinygrad](https://github.com/tinygrad/tinygrad) · 33.3k ★ · 486 contributors · MIT ✅
The best working exemplar of a minimal hardware-portability layer: **8 in-tree backends** (OpenCL, CPU, Metal, CUDA, AMD, NV, QCOM, WebGPU) and a README claim that a new accelerator needs only **~25 low-level ops** (their simplification — a real port also needs renderer/runtime/allocator glue). Contribution is *monetized*: **$50k+ in cash bounties** paid to ~34 external contributors via a public spreadsheet, used as tiny corp's hiring pipeline. But the bar is deliberately hostile to low-effort work: AI-generated-looking PRs are closed without feedback (possibly with a ban), docs/whitespace PRs are rejected, speedups must be benchmarked. Best as a *second* project once you've built confidence — but the single best teacher of exactly the abstraction a CUDA challenger needs.

**6. bitsandbytes** — [github.com/bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) ✅
Highest leverage per PR in this list. The de facto standard 4/8-bit quantization library (underpins Hugging Face `load_in_4bit` and QLoRA) whose maintainers [explicitly stated](https://github.com/bitsandbytes-foundation/bitsandbytes/discussions/1340) they **cannot build Apple Silicon support themselves**: "We don't have enough resources to do this work ourselves, so we're dependent on community contributions." ~19 volunteers offered hardware; no lead implementer emerged for months; the first macOS wheels only shipped Dec 2025 (v0.49.0) with self-described *slow* MPS implementations, and Hugging Face still labels Apple Silicon support "experimental". Walk in with working Metal kernels and you become *the* person who brought QLoRA to Mac.

### Tier 2 — strong signals from live GitHub data (not deep-verified)

| Project | Stars | Contribs | Open GFIs | Why it's interesting |
|---|---|---|---|---|
| [SGLang](https://github.com/sgl-project/sglang) | 30.5k | 1,649 | **50** (most in dataset) | vLLM's fastest-growing rival; Python; pushed same-day |
| [Unsloth](https://github.com/unslothai/unsloth) | 68.4k | **246** | 8 | Huge stars-to-contributors ratio → wide-open field |
| [MLX](https://github.com/ml-explore/mlx) | 27.6k | 269 | 0 | Apple's own array framework; C++ core, Python API |
| [diffusers](https://github.com/huggingface/diffusers) | 34.1k | 1,140 | 7 | HF's classic welcoming culture; MPS support has gaps to fix |
| [Modular (Mojo/MAX)](https://github.com/modular/modular) | 26.6k | 434 | 16 | The Mojo language repo itself; mixed licensing (stdlib Apache-2.0-with-LLVM-exceptions; MAX under community license) |
| [Burn](https://github.com/tracel-ai/burn) | 15.6k | 292 | 8 | Rust DL framework, multi-backend incl. wgpu/Metal |
| [wgpu](https://github.com/gfx-rs/wgpu) | 17.6k | 667 | 10 | Rust WebGPU implementation — the portable-GPU substrate |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | 121k | 1,828 | 17 | C++; the most mature Metal inference stack |
| [JAX](https://github.com/jax-ml/jax) | 36k | 1,097 | 4 | Google-backed; compiler-heavy (XLA) |
| [Ray](https://github.com/ray-project/ray) | 43.3k | 1,649 | 0 | Distributed compute; Python |

*(Stars/contributors/GFI counts pulled live via GitHub API, 2026-07-19. Transformers (162.7k ★, 4,033 contribs) and Keras (64.2k ★) had 0 open `good first issue` items on check day — their starter issues get claimed fast; both remain welcoming.)*

**Bottom line for Part 1:** Start with **vLLM + vllm-metal** (structured funnel, your hardware, inference upside) and **PyTorch MPS-adjacent issues** (skills that compound toward Part 2). Consider **SGLang** if you want the least-crowded good-first-issue board, and **bitsandbytes** if you want maximum leverage from day one.

---

## Part 2 — Breaking the CUDA moat

### 2.1 Landscape snapshot

| Stack | State (2026-07) |
|---|---|
| **AMD ROCm/HIP** | Real datacenter momentum (Meta, OpenAI, Oracle MI300X clusters, Azure, Frontier/El Capitan ◐). ROCm 7 (Sept 2025) brought native Windows support + day-zero PyTorch; ROCm 7.2.x (Jan–Mar 2026) added official RDNA4 consumer support (RX 9070/9060 series) ◐. Consumer experience still far behind CUDA (§2.4). |
| **Apple MLX / Metal / MPS** | MLX is mature for local inference; PyTorch-MPS has verified structural gaps in training, ops, and quantization (§2.3) ✅. |
| **OpenAI Triton** | The de facto portability vehicle — AMD's PyTorch attention kernels are AOT-compiled Triton ✅. But tuning heuristics assume NVIDIA warp geometry (§2.4) ○. |
| **Mojo / Modular MAX** | Chris Lattner's "Democratizing AI Compute" thesis; 26.6k ★ repo, 16 GFIs. Deep-research pass produced no verified claims on its current backend maturity — evaluate directly before committing (open question). |
| **tinygrad** | Working proof that a ~25-op portability layer can drive 8 backends ✅. |
| **JAX/XLA/IREE, ONNX Runtime** | Not covered by verified claims in this run — absence of evidence, not assessed weakness. |
| **llama.cpp/ggml** | Mature Metal kernels pre-2026; the strongest existing non-CUDA inference path on Mac ✅ (per verification caveats). |
| **OpenCL** | Lattner: still no standardized tensor-core access → typically a 5–10x loss vs CUDA for GenAI ○. Effectively out of the race. |
| **WebGPU/wgpu, Vulkan compute** | Portable substrates; used by tinygrad (WEBGPU backend) and third-party attention libs (Vulkan) as consumer-GPU escape hatches ◐. |
| **CUDA-source translation** | AMD's `hipify` can't translate CUDA *library* calls (cuBLAS/cuDNN/TensorRT) ○; SCALE (Spectral Compute) compiles CUDA source for AMD but is **proprietary** (free non-commercial only) ◐. No healthy open-source equivalent. |

### 2.2 The gap list — what a small team (Mojo/Rust/Python) could actually close

1. **Training-grade attention on Apple GPU** ✅ — PyTorch MPS has **no dedicated SDPA backward pass**; gradients fall back to the slow composite "math" path, and the restriction disables the fast forward kernel during training too ([pytorch#179294](https://github.com/pytorch/pytorch/issues/179294)). Fix PRs (#182746, #181030) sat open/unmerged as of check day. No FlashAttention-class fwd+bwd kernels exist upstream for Metal. *Fit: Metal Shading Language core + Python bindings; Rust via metal-rs possible.*
2. **Fast low-bit quantization on Apple Silicon** ✅ — bitsandbytes' first macOS wheels (Dec 2025) are self-described slow; maintainers publicly dependent on community ([discussion #1340](https://github.com/bitsandbytes-foundation/bitsandbytes/discussions/1340)). QLoRA-on-Mac lacks its standard tooling. *Fit: Metal kernels + Python; or a standalone Rust lib with PyTorch/MLX bindings.*
3. **MPS operator completeness** ✅ — the official [coverage tracker #141287](https://github.com/pytorch/pytorch/issues/141287) is still open and active: 3D max-pooling only landed in 2.9 (Oct 2025); `linalg_solve`/`cholesky` arrived 2.7; `linalg_lu_solve` 2.10; `linalg_qr` still nightly-only; complex-dtype linalg still failing; no float64 at all ○. Demand is pre-quantified by per-op request counts. *Fit: Python/C++/Metal, one op per PR.*
4. **RDNA-native attention kernels** ◐ — AMD's default Composable-Kernel FA backend is Wave64/CDNA-first: it **fails to compile on RDNA4** (Wave32) with the Triton backend as the documented workaround; **no backward on RDNA3** (RDNA4 backward only with `deterministic=False`) per [ROCm/flash-attention](https://github.com/ROCm/flash-attention); coverage holes within gfx11 (gfx1101 eager-only, [pytorch#159226](https://github.com/pytorch/pytorch/issues/159226)); and shapes where the FLASH_ATTENTION backend is **2.5x slower than the math backend** on a 7900 XTX ([pytorch#152595](https://github.com/pytorch/pytorch/issues/152595)). A consumer-RDNA-first attention library is missing. *Fit: Triton (Python) or HIP.*
5. **Cross-vendor drop-in SDPA replacement** ◐ — one library that monkey-patches/registers into `torch.nn.functional.scaled_dot_product_attention` and dispatches the best available kernel per device (Metal / ROCm-Triton / Vulkan / Intel). [Aule-Attention](https://github.com/AuleTechnologies/Aule-Attention) just emerged in exactly this niche (Triton for MI300X/RDNA3, Vulkan for consumer GPUs) — validation the gap is real, and it's early enough to join or compete. *Fit: Python + Rust/Vulkan.*
6. **Triton autotuning for non-NVIDIA geometry** ○ — heuristics tuned for 32-thread warps under-saturate AMD's 64-wide wavefronts; on new silicon, compiled kernels can silently degrade 30–50% vs hand-written HIP. The RDNA3 FA delay was a *compiler* blocker (Navi WMMA support in Triton), not a kernel problem — AMD maintainers spent months on it ([pytorch#112997](https://github.com/pytorch/pytorch/issues/112997) history). Autotuning/benchmark infra for non-NVIDIA Triton targets is thin. *Fit: Python.*
7. **Non-CUDA developer parity in flagship projects** ✅ — even vLLM's dev/test dependency set is CUDA-only; the documented kernel-dev path pins CUDA 12.9 wheels; contributors without NVIDIA GPUs are told to rely on CI ([contributing guide](https://docs.vllm.ai/en/latest/contributing/overview.html)). Runnable Metal/ROCm test suites and CI paths are an unglamorous, uncontested niche that compounds every other gap. *Fit: Python/infra.*
8. **Unified-memory-aware serving** ✅/○ — vllm-metal's pre-v0.2.0 KVCache path "reported fake memory capacity", disabling continuous batching entirely; its replacement unified Metal kernel self-reports **83x TTFT / 3.6x throughput** gains (against its own broken v0.1.0 — don't generalize). Paged attention and KV-cache management designed for unified memory (Apple Silicon, AMD APUs) is nascent. *Fit: Python + Metal.*
9. **Open CUDA-source portability layer** ◐ — `hipify` can't translate library calls; SCALE is proprietary. A real open competitor is a large-compiler-project — low feasibility for a small team; listed for completeness, not recommended as a first project.
10. **Distributed training off-NVIDIA** ○ — MPS has no distributed-training support; SemiAnalysis attributes much of MI300X's training shortfall to RCCL collectives lagging NCCL. Big surface, hard to attack solo; medium-term option.

### 2.3 Bottlenecks you'll hit **today** on macOS (Apple Silicon) — all ✅ verified

- **Ops**: incomplete MPS coverage years after the May-2022 launch; the runtime error message literally points users to tracking issue [#141287](https://github.com/pytorch/pytorch/issues/141287); `PYTORCH_ENABLE_MPS_FALLBACK=1` (silent CPU fallback) is the standard workaround. Dense linalg lagged CUDA by ~4 years; QR still nightly-only; complex dtypes broken; float64 nonexistent ○.
- **Attention/training**: no SDPA backward kernel → training can't use the fast attention path *at all*; the dedicated forward hides under the generic "math" backend and only fires with dropout=0.0, no-grad, contiguous inputs (or macOS 15+); basic 2-D shapes that work on CPU raise IndexError on MPS (HF Transformers ships a workaround that unsqueezes to 4-D).
- **Quantization**: bitsandbytes reached macOS only Dec 2025, slow and "experimental" per HF docs. (MLX's own quantization is the current escape hatch for inference.)
- **Distributed**: MPS backend does not support distributed training ○.
- **Tooling**: `torch.compile` fusions often fall back to CPU or run as unfused generic Metal kernels ○; flagship projects' test suites assume CUDA ✅.
- **Performance delta**: the only systematic price-matched study ([arXiv:2501.14925](https://arxiv.org/abs/2501.14925), Jan 2025, M2-generation) found end-to-end training **~3–4x slower** than similarly-priced NVIDIA cards — with nuances: near-parity in one FP32 case (M2 Ultra vs A6000 on GPT2-large), and Apple *wins* when NVIDIA VRAM is insufficient and offload kicks in. Medium confidence (single preprint, pre-M4, pre-2026 improvements). ⚠️ A tempting explanation — "FP16 GEMM acceleration is largely missing on Apple's stack" — was **refuted 0–3** in verification; do not repeat it.

### 2.4 Bottlenecks you'll hit **today** on AMD (consumer RDNA) — ◐ corroborated

- **Official support arrived only recently and is thin**: PyTorch-on-Radeon (Windows + Linux) shipped as *preview* in ROCm 6.4.4 (Sept 2025); [ROCm 7.2.x](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html) (Jan–Mar 2026) made RDNA4 (RX 9070/9060 series) + select RDNA3 official ([Phoronix](https://www.phoronix.com/news/AMD-ROCm-7.2-Released), [TechPowerUp](https://www.techpowerup.com/341329/amd-enables-pytorch-on-radeon-rx-7000-9000-gpus-with-windows-and-linux-preview)). Community consensus: "officially supported" ≠ tested/optimized; **inference is far more mature than training**; fine-tuning on consumer RDNA is still unreliable.
- **Attention kernels**: CK backend is CDNA-first — fails to compile on RDNA4 (Wave64 asm vs Wave32); RDNA3 has **no backward pass**; fp16/bf16 only, head-dim ≤ 256; the Triton/aiter backend is the feature-complete path but sliding-window attention (Mistral/Gemma) was still WIP ○; per-SKU holes (gfx1101 eager-only); pathological shapes where flash is 2.5x slower than math.
- **History says the lag is structural**: upstream PyTorch SDPA-on-ROCm took **~13 months** from request (Nov 2023) to close (Dec 2024), landed CDNA-only, and RDNA3 waited on Triton compiler WMMA support; the backward kernel was reported "very low efficiency" even after landing ([pytorch#112997](https://github.com/pytorch/pytorch/issues/112997), [ROCm/aotriton#16](https://github.com/ROCm/aotriton/issues/16)).
- **Ecosystem friction**: RDNA4 initially demanded PyTorch nightlies (breaking torchaudio via C++ header drift) ○; ROCm library defaults target Instinct datacenter cards until you bypass enterprise settings ○; RDNA4's native FP8 goes unexploited by standard kernels (TunableOp autotuning reportedly ~60% speedup) ○; Windows: open triaged issue of degrading performance / suspected VRAM leak on 7900 XTX with no AMD reply as of Jan 2026 ([ROCm#5834](https://github.com/ROCm/ROCm/issues/5834)).
- **Missing NVIDIA-equivalents**: TensorRT-LLM and FlashAttention-3/4 have no ROCm counterpart (FA3 is Hopper-only, CUDA ≥ 12.3) — AMD runs a kernel generation behind on attention ◐.
- **Performance deltas (datacenter, for calibration)**: SemiAnalysis measured MI300X at 37–66% of H100/H200 training performance due to software immaturity (["CUDA Moat Still Alive"](https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training)); by 2026, MI355X reaches ~90–95% of H100 on standard PyTorch/vLLM paths but NVIDIA stays 20–40% ahead wherever CUDA-specific libraries apply ○.

### 2.5 Why the moat holds (so you aim at the right wall)

The moat is **not** silicon: it's (1) a 15-year kernel-library ecosystem alternatives must re-implement (hipify can't translate cuBLAS/cuDNN *calls*), (2) developer-tooling gravity — even multi-backend flagships develop and test CUDA-first ✅, (3) compiler infrastructure — Triton's NVIDIA-shaped assumptions delayed AMD consumer attention by a year ○, and (4) out-of-box polish: NVIDIA works on day one; every alternative starts with a debugging session. Each of those is attackable by software contributions — which is exactly the opportunity list above.

---

## Part 3 — What I'd build (impact × feasibility, for you specifically)

**The pick: a FlashAttention-class forward + backward kernel library for Apple Silicon, shipped as a drop-in PyTorch SDPA override (and MLX custom op).**
Why it wins: the gap is unanimous-verified and precisely scoped ([#179294](https://github.com/pytorch/pytorch/issues/179294)); upstream fix PRs stalled for months, so the window is open; it unlocks *training* (not just inference) on hardware millions of engineers already own; vllm-metal's 83x-from-one-kernel result shows the headroom is real; and you own the test hardware. Study prior art first: the Metal kernels vllm-metal ships, and philipturner/metal-flash-attention (○ inference-focused — verify current state).
*Stack: Metal Shading Language + C++/Objective-C core, Python/PyTorch binding (via `torch.library` override), optional Rust wrapper. Mojo is worth a spike only after you verify its Apple-GPU target maturity — that was an open question in this research.*

**Ranked directions:**
1. **Metal SDPA fwd+bwd library** (above) — highest impact, individually ownable.
2. **Apple Silicon 4/8-bit quantization backend** — bitsandbytes MPS backend or standalone; maintainers are asking for exactly this; QLoRA-on-Mac is the prize.
3. **MPS op-coverage PRs** (warm-up) — tracker-guided, demand pre-quantified per op, guaranteed relevance; the standard path to PyTorch commit access.
4. **vllm-metal contributions** — active GFI funnel, majority-Python, releases every few days.
5. **RDNA-first attention kernels** (if you buy AMD hardware) — Triton/aiter path, or join Aule-Attention; the consumer-AMD field is even emptier than Mac.
6. **tinygrad bounties** — paid portability education; expect a hostile bar.
7. **Non-CUDA CI/test parity for vLLM & friends** — unglamorous, uncontested, buys maintainer goodwill.

**A realistic 90-day arc:**
Weeks 1–2: one vLLM/vllm-metal GFI + read the MPS SDPA issue chain (#179294, #182746, #181030). Weeks 3–8: 2–3 MPS op PRs from tracker #141287 (linalg/complex-dtype ops are the open lane) + one bitsandbytes MPS kernel. Weeks 9–12: prototype the standalone Metal SDPA backward, benchmark vs the math path on your machine, publish numbers. Then decide: upstream the kernels into PyTorch, or grow the standalone library (the bitsandbytes precedent shows a standalone that frameworks adopt is a viable route to becoming infrastructure).

---

## Caveats & open questions

- AMD-section claims are corroborated (◐/○) but did not go through the 3-vote adversarial gate the Apple-section claims passed; treat magnitudes as approximate.
- The 3–4x Apple training delta is one M2-era preprint; M4-generation + 2026 MLX/MPS work may have narrowed it.
- Not assessed for lack of surviving evidence (absence ≠ weakness): Intel oneAPI/SYCL, JAX/XLA/IREE, ONNX Runtime, Mojo/MAX backend maturity, Triton's non-NVIDIA backends' current quality.
- Refuted in verification — do not cite: "FP16 GEMM largely missing on Apple stack" (0–3); "vllm-metal GFI tracker enumerates capability gaps" (1–2).
- How long the MPS-SDPA window stays open depends on when PyTorch merges #182746/#181030 — check before committing to the standalone-library route.

## Sources (primary)

[vLLM contributing guide](https://docs.vllm.ai/en/latest/contributing/overview.html) · [pytorch/contribute](https://github.com/pytorch/pytorch/contribute) · [ExecuTorch new-contributor guide](https://docs.pytorch.org/executorch/stable/new-contributor-guide.html) · [vllm-metal](https://github.com/vllm-project/vllm-metal) · [tinygrad](https://github.com/tinygrad/tinygrad) · [bitsandbytes discussion #1340](https://github.com/bitsandbytes-foundation/bitsandbytes/discussions/1340) · [pytorch#141287 (MPS op coverage)](https://github.com/pytorch/pytorch/issues/141287) · [pytorch#179294 (MPS SDPA backward)](https://github.com/pytorch/pytorch/issues/179294) · [arXiv:2501.14925 (price-matched benchmark)](https://arxiv.org/abs/2501.14925) · [pytorch#112997 (FA on ROCm)](https://github.com/pytorch/pytorch/issues/112997) · [ROCm/flash-attention](https://github.com/ROCm/flash-attention) · [ROCm/aotriton#16](https://github.com/ROCm/aotriton/issues/16) · [ROCm/ROCm#5834](https://github.com/ROCm/ROCm/issues/5834) · [pytorch#152595](https://github.com/pytorch/pytorch/issues/152595) · [pytorch#159226](https://github.com/pytorch/pytorch/issues/159226) · [AMD Radeon/Ryzen ROCm docs](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html) · [Phoronix: ROCm 7.2](https://www.phoronix.com/news/AMD-ROCm-7.2-Released) · [TechPowerUp: PyTorch on Radeon](https://www.techpowerup.com/341329/amd-enables-pytorch-on-radeon-rx-7000-9000-gpus-with-windows-and-linux-preview) · [SemiAnalysis: MI300X training](https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training) · [SemiAnalysis: AMD 2.0](https://semianalysis.com/2025/04/23/amd-2-0-new-sense-of-urgency-mi450x-chance-to-beat-nvidia-nvidias-new-moat/) · [Modular: Democratizing AI Compute](https://www.modular.com/democratizing-ai-compute) · [Aule-Attention](https://github.com/AuleTechnologies/Aule-Attention) · [Thunder Compute: ROCm vs CUDA](https://www.thundercompute.com/blog/rocm-vs-cuda-gpu-computing) · [sdxcentral: Beyond CUDA](https://www.sdxcentral.com/analysis/beyond-cuda-inside-the-push-to-loosen-nvidias-grip-on-ai-computing/)
