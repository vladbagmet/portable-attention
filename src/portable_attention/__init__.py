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
"""

from __future__ import annotations

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
    plan_tiles,
    shared_memory_bytes_for,
)
from .vulkan import VulkanCapability, VulkanDevice, detect_vulkan, vulkan_available

__all__ = [
    "ConformanceCase",
    "ConformanceResult",
    "DeviceLimits",
    "SdpaBackend",
    "TilePlan",
    "TileSizingError",
    "VulkanCapability",
    "VulkanDevice",
    "__version__",
    "assert_conforms",
    "available_backends",
    "check_backend",
    "conformance_cases",
    "detect_vulkan",
    "get_backend",
    "plan_tiles",
    "register_backend",
    "scaled_dot_product_attention",
    "shared_memory_bytes_for",
    "vulkan_available",
]

# Single source of truth for the package version. The build backend
# (hatchling) reads this string directly via [tool.hatch.version] in
# pyproject.toml, so it must never be duplicated there.
__version__ = "0.0.1"
