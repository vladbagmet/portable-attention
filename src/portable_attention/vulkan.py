"""Vulkan runtime capability detection.

The M2 goal is a portable GPU backend built on **Vulkan (V3DV)** — see
``ROADMAP.md``. Before any such backend can register itself, the package has to
answer one honest question on the host it is imported on: *is a usable Vulkan
runtime present at all?* This module answers exactly that, and nothing more. It
performs no GPU work and pulls in no new dependency; it only inspects what the
system already exposes.

Detection has two independent parts:

* the **Vulkan ICD loader** (the ``libvulkan`` shared library), located with
  :func:`ctypes.util.find_library`; without it there is no Vulkan on the host;
* a supported **Python binding** to drive Vulkan compute, detected by import
  probing (no import side effects).

Both must be present for :func:`detect_vulkan` to report the runtime available.
The probes are injectable so the logic is fully unit-testable on a host with no
Vulkan at all (such as the CPU-only development floor), where detection
correctly reports *unavailable* with a specific reason. A future V3DV backend
gates its own registration on :func:`vulkan_available`.
"""

from __future__ import annotations

import ctypes.util
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "VulkanCapability",
    "detect_vulkan",
    "vulkan_available",
]

# Python bindings that could drive a Vulkan compute backend, most-preferred
# first. Presence is checked by import probing only (no module is imported).
_KNOWN_BINDINGS: tuple[str, ...] = ("kp", "vulkan")

_NO_LOADER_REASON = (
    "no Vulkan ICD loader found (libvulkan); install a Vulkan runtime such as Mesa V3DV"
)


@dataclass(frozen=True)
class VulkanCapability:
    """Result of probing the host for a usable Vulkan runtime.

    Attributes:
        available: ``True`` only when both a Vulkan ICD loader and a supported
            Python binding are present. When ``False``, ``reason`` explains why.
        loader: The ICD loader library name reported by the system (e.g.
            ``"libvulkan.so.1"``), or ``None`` when no loader was found.
        binding: The name of the first supported, importable Python binding, or
            ``None`` when none is installed.
        reason: A human-readable explanation when ``available`` is ``False``;
            ``None`` when the runtime is available.
    """

    available: bool
    loader: str | None
    binding: str | None
    reason: str | None


def _default_find_loader() -> str | None:
    """Return the Vulkan ICD loader library name, or ``None`` if absent."""
    return ctypes.util.find_library("vulkan")


def _default_find_binding() -> str | None:
    """Return the first installed supported binding name, or ``None``.

    Uses :func:`importlib.util.find_spec`, which locates a module without
    importing it, so probing has no side effects.
    """
    for name in _KNOWN_BINDINGS:
        if importlib.util.find_spec(name) is not None:
            return name
    return None


def detect_vulkan(
    *,
    find_loader: Callable[[], str | None] = _default_find_loader,
    find_binding: Callable[[], str | None] = _default_find_binding,
) -> VulkanCapability:
    """Probe the host and describe its Vulkan runtime capability.

    The two probes are injectable so the detection logic can be exercised
    without a Vulkan runtime installed (and so a real backend can supply
    stricter device-level probes later). Defaults inspect the system Vulkan
    loader and the installed Python bindings.

    Args:
        find_loader: Callable returning the ICD loader library name, or ``None``
            when no Vulkan loader is present.
        find_binding: Callable returning a supported, importable binding name,
            or ``None`` when none is installed.

    Returns:
        A :class:`VulkanCapability` describing the result. ``available`` is
        ``True`` only when both probes succeed.
    """
    loader = find_loader()
    if loader is None:
        return VulkanCapability(
            available=False,
            loader=None,
            binding=find_binding(),
            reason=_NO_LOADER_REASON,
        )
    binding = find_binding()
    if binding is None:
        return VulkanCapability(
            available=False,
            loader=loader,
            binding=None,
            reason=(
                "Vulkan loader present but no supported Python binding "
                f"installed (tried: {', '.join(_KNOWN_BINDINGS)})"
            ),
        )
    return VulkanCapability(available=True, loader=loader, binding=binding, reason=None)


def vulkan_available() -> bool:
    """Return ``True`` when the host has a usable Vulkan runtime.

    Convenience wrapper over :func:`detect_vulkan` for the common yes/no gate
    (e.g. deciding whether to register a Vulkan-backed attention backend).
    """
    return detect_vulkan().available
