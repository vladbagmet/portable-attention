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
- **Backend registration** gated on a compute-capable device, driving a minimal
  SPIR-V attention kernel, validated against the reference oracle through the
  conformance kit.
- **llvmpipe as the CI Vulkan target** — software Vulkan 1.3 needs no GPU, so
  hosted runners can execute the conformance kit on the Vulkan backend.
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

- *Deferred (hardware-gated):* a Metal forward+backward backend for the verified
  Apple training gap is a separate future track, not part of this roadmap — it
  requires Apple hardware the current development environment does not have.

## Continuous (not milestone-gated)

- **e2e floor:** every release verified from a clean checkout on the Pi 5.
- **Docs honesty:** README/CONTRIBUTING track reality; breaking changes get a
  changelog entry.
