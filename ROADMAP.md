# Roadmap

Each milestone is *shippable*: it ends at a state a real engineer could install
and use. Direction is set by `VISION.md` and grounded in `RESEARCH.md`; this
file is regroomed on the weekly roadmap shift. Bricks are sized for one night
shift and tracked as step files under `workflow-state/`.

## M0 — Walking skeleton (installable CPU reference)

**Goal:** `pip install portable-attention`, call one SDPA function on CPU, get a
numerically correct result. Everything green on the Pi 5 floor.

- **M0.1 — Repo skeleton.** src-layout package, test harness (pytest), CI
  workflow (lint + typecheck + test), a minimal correct CPU reference SDPA, and
  one trivially-passing correctness test. *(this brick)*
- **M0.2 — Reference SDPA correctness suite.** Test against a golden/oracle
  (naive einsum softmax) across shapes, dtypes, scale, and `is_causal`; add an
  explicit `attn_mask` path. Property-based edge cases (batch/head dims).
- **M0.3 — Public API + packaging polish.** Freeze the `scaled_dot_product_attention`
  signature (torch-compatible surface), docstrings, `py.typed`, README quickstart,
  version story. Publishable sdist/wheel built in CI.

## M1 — Backend contract + honest benchmarks

**Goal:** a documented backend protocol with ≥2 interchangeable backends and a
reproducible benchmark harness.

- Backend `Protocol` + registry/dispatch (`backend="auto"|"reference"|...`),
  with the reference backend as the conformance oracle.
- A second CPU backend that exercises the contract (e.g. a blocked/streaming
  "flash-style" CPU kernel) — proves the seam is real, improves memory scaling.
- Benchmark harness: latency/throughput/peak-memory across shapes, dated
  results with hardware + commit hash appended to `BENCHMARKS.md`.
- A conformance test kit any backend must pass (the non-CUDA developer-parity
  promise made concrete).

## M2 — First vendor backend where the gap is verified

**Goal:** close one real, verified gap end-to-end.

- Path: **Vulkan (V3DV)** as the portable GPU backend, validated on low-power
  ARM hardware. CPU reference remains the correctness oracle.
- **Device detection** — enumerate physical devices through the loader and
  report which can run compute. *(done: `detect_vulkan`)*
- **Tile sizing policy** — one device-parameterized planner shared by every
  blocked backend. *(done: `plan_tiles`)*
- **Device open + host transfer** — a logical device with a compute queue and
  mapped storage buffers arrays travel through. *(done: `VulkanContext`)*
- **Shader dispatch** — compile a SPIR-V module, bind storage buffers and push
  constants, submit and wait. Specialization constants fix a kernel's workgroup
  size and tile shape when the pipeline is built, so one committed SPIR-V file
  serves every tile plan. *(done: `VulkanPipeline`)*
- **Blocked algorithm specification** — the tiled online-softmax attention a
  device kernel implements, written once in NumPy over a `TilePlan` and checked
  against the oracle, so the shader is a transcription with something to diff
  against. *(done: `blocked_attention`)*
- **Backend registration** gated on a compute-capable device, driving a minimal
  SPIR-V attention kernel, validated against the reference oracle through the
  conformance kit.
- **llvmpipe as the CI Vulkan target** — software Vulkan 1.3 needs no GPU, so
  hosted runners can execute the conformance kit on the Vulkan backend.
  *(done: `scripts/vulkan-conformance.sh`, run by the `vulkan (llvmpipe)` CI
  job)*
- Dated Vulkan numbers appended to `BENCHMARKS.md` alongside the CPU backends.
- Optional autograd hook (backward pass) so the layer becomes training-usable on
  the Vulkan path.

Kernel design has to fit the V3D (VideoCore VII) device limits measured on the
reference board — a flash-style tile must live within 16 KiB of shared memory at
no more than 256 invocations per workgroup:

| Limit | V3D 7.1.7.0 |
| --- | --- |
| `maxComputeSharedMemorySize` | 16384 (16 KiB) |
| `maxComputeWorkGroupInvocations` | 256 |
| `subgroupSize` (with `subgroupSizeControl`) | 16 |
| `maxPerStageDescriptorStorageBuffers` | 8 |
| `maxStorageBufferRange` | 1 GiB |
| `maxMemoryAllocationSize` | 1 GiB |

Those numbers belong to the device, not to the kernel. The tiling policy is
written once, parameterized by `(shared_memory_bytes, max_threads_per_group,
simd_width)`, and each backend supplies its own values — M2 pays for that seam
so M3 gets a tile size instead of a second hand-tuned kernel. That seam now
exists as `portable_attention.tiling`: on V3D at `head_dim=64` in float32 it
plans a 16x16 tile filling the 256-invocation workgroup in 13440 of the 16384
available bytes. The output accumulator stays out of that budget — a
`block_q x head_dim` block would cost more occupancy than it buys, so it is
register-resident, 4 values per invocation at that plan.

## M3 — Metal backend (Apple Silicon)

**Goal:** the same conformance kit passing on an Apple GPU, reached from a plain
`pip install` with no developer tooling on the machine.

- Forward SDPA compute kernel in MSL, embedded as source and compiled at first
  use through the Metal runtime compiler (`newLibraryWithSource:`), with the
  pipeline state cached. No precompiled `.metallib`, no Xcode requirement.
- Python reaches Metal through `pyobjc-framework-Metal`, declared as the
  `metal` optional extra and imported lazily; core install stays NumPy-only.
- Registration is gated on a real device, so `available_backends()` is unchanged
  on non-Apple hosts and the existing suite keeps running everywhere.
- Conformance-kit pass against the reference oracle, then dated benchmarks in
  `BENCHMARKS.md`. Backward/autograd is a later brick.
- A `macos-14` (arm64) CI job that runtime-compiles the MSL and runs the
  conformance kit when `MTLCreateSystemDefaultDevice()` returns a device.
  Whether hosted macOS runners expose a usable GPU is unknown, so the job is
  probed before anything depends on it; offline `xcrun metal` stays a
  non-blocking extra at most.

The measured limits on the reference Apple device sit roughly 2x above V3D on
every axis that shapes a tile, which is what the parameterized policy above is
for:

| Limit | V3D 7.1.7.0 | Apple M1 Pro (14-core, `apple7`/`metal3`) |
| --- | --- | --- |
| shared / threadgroup memory | 16 KiB | 32 KiB |
| max invocations per workgroup | 256 | 1024 |
| SIMD / subgroup width | 16 | 32 |
| max buffer / storage range | 1 GiB | 16 GiB |
| unified memory | yes | yes |

Apple hardware is not reachable from the machines that run CI, so a Metal brick
is signed off by a human running `scripts/verify-metal.sh` on a Mac and pasting
its report into the PR. Until such a report exists, a Metal change is described
as compiled-but-unverified — see CONTRIBUTING.md.

## Continuous (not milestone-gated)

- **e2e floor:** every release verified from a clean checkout on the Pi 5.
- **Docs honesty:** README/CONTRIBUTING track reality; breaking changes get a
  changelog entry.
