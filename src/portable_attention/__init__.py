"""portable-attention: a portable, CUDA-independent attention/SDPA layer.

CPU-first and correctness-obsessed. The public surface is intentionally small
and tracks ``torch.nn.functional.scaled_dot_product_attention`` where possible,
so it can act as a drop-in on hardware where the fast vendor path is missing.

Only the names re-exported here are public API; everything else is internal and
may change without notice.

Backends are pluggable: the CPU ``reference`` implementation is the correctness
oracle and is always registered. Select a backend explicitly with
:func:`get_backend`; the public :func:`scaled_dot_product_attention` keeps a
torch-compatible signature and dispatches to the ``"auto"`` backend.
"""

from __future__ import annotations

from .dispatch import (
    SdpaBackend,
    available_backends,
    get_backend,
    register_backend,
    scaled_dot_product_attention,
)

__all__ = [
    "SdpaBackend",
    "__version__",
    "available_backends",
    "get_backend",
    "register_backend",
    "scaled_dot_product_attention",
]

# Single source of truth for the package version. The build backend
# (hatchling) reads this string directly via [tool.hatch.version] in
# pyproject.toml, so it must never be duplicated there.
__version__ = "0.0.1"
