"""portable-attention: a portable, CUDA-independent attention/SDPA layer.

CPU-first and correctness-obsessed. The public surface is intentionally small
and tracks ``torch.nn.functional.scaled_dot_product_attention`` where possible,
so it can act as a drop-in on hardware where the fast vendor path is missing.

Only the names re-exported here are public API; everything else is internal and
may change without notice.

Backends are pluggable: the CPU ``reference`` implementation is the correctness
oracle and is always registered. Select a backend explicitly with
:func:`get_backend`; the public :func:`scaled_dot_product_attention` keeps a
torch-compatible signature and dispatches to the ``"auto"`` backend. Every
backend must pass the shared conformance kit (:func:`assert_conforms`,
:func:`check_backend`, :func:`conformance_cases`), which pins the
developer-parity promise: identical behaviour against the reference oracle
across the documented contract matrix.

Blocked GPU backends share one tile-sizing policy (:func:`plan_tiles`,
:class:`DeviceLimits`, :class:`TilePlan`) so a new device contributes its
shared-memory / workgroup / SIMD numbers instead of a second hand-tuned kernel.
The algorithm those kernels implement is written out once in NumPy as
:func:`blocked_attention`: a specification to port from and to diff a device
kernel against, not a backend to serve traffic with.

On hosts with Vulkan, :func:`detect_vulkan` reports what hardware is there and
:class:`VulkanContext` opens it: a logical device, a compute queue, and
host-visible :class:`VulkanBuffer` allocations arrays can be copied through,
which a :class:`VulkanPipeline` then runs SPIR-V compute shaders over. Where
such a device exists, importing this package registers a ``"vulkan"`` backend
that runs the blocked-attention shader and falls back to the CPU for calls the
kernel does not implement; select it with ``get_backend("vulkan")``. Set
``PORTABLE_ATTENTION_DISABLE_VULKAN`` to skip both the probe and the
registration.
"""

from __future__ import annotations

from .blocked import blocked_attention
from .conformance import (
    ConformanceCase,
    ConformanceResult,
    assert_conforms,
    check_backend,
    conformance_cases,
)
from .dispatch import (
    SdpaBackend,
    available_backends,
    get_backend,
    register_backend,
    scaled_dot_product_attention,
)
from .tiling import (
    DeviceLimits,
    TilePlan,
    TileSizingError,
    candidate_plans,
    plan_tiles,
    shared_memory_bytes_for,
    tile_plan_for,
)
from .vkbackend import VulkanAttention, register_vulkan_backend
from .vkcompute import VulkanBuffer, VulkanContext, VulkanError, VulkanPipeline
from .vulkan import VulkanCapability, VulkanDevice, detect_vulkan, vulkan_available

__all__ = [
    "ConformanceCase",
    "ConformanceResult",
    "DeviceLimits",
    "SdpaBackend",
    "TilePlan",
    "TileSizingError",
    "VulkanAttention",
    "VulkanBuffer",
    "VulkanCapability",
    "VulkanContext",
    "VulkanDevice",
    "VulkanError",
    "VulkanPipeline",
    "__version__",
    "assert_conforms",
    "available_backends",
    "blocked_attention",
    "candidate_plans",
    "check_backend",
    "conformance_cases",
    "detect_vulkan",
    "get_backend",
    "plan_tiles",
    "register_backend",
    "register_vulkan_backend",
    "scaled_dot_product_attention",
    "shared_memory_bytes_for",
    "tile_plan_for",
    "vulkan_available",
]

# Single source of truth for the package version. The build backend
# (hatchling) reads this string directly via [tool.hatch.version] in
# pyproject.toml, so it must never be duplicated there.
__version__ = "0.0.1"

# Registers only where the host has a compute-capable Vulkan device, so this is
# a no-op (one detection probe) on CPU-only machines. ``overwrite`` because the
# registry outlives a reload of this module, which would otherwise raise on the
# name it registered the first time.
register_vulkan_backend(overwrite=True)
