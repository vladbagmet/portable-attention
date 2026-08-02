"""Vulkan runtime capability detection.

The M2 goal is a portable GPU backend built on **Vulkan (V3DV)** — see
``ROADMAP.md``. This module answers a narrow, honest question on the host it is imported
on: *are the dependencies a Vulkan backend needs present at all?* It is a
**dependency preflight**, not a promise that a GPU will enumerate — confirming
an actual, working device is real Vulkan work that only a backend can do at
registration. This module performs no GPU work and pulls in no new dependency;
it only inspects what the system already exposes.

The preflight has two independent parts:

* the **Vulkan ICD loader** (the ``libvulkan`` shared library), located with
  :func:`ctypes.util.find_library`; without it there is no Vulkan on the host;
* a supported **Python binding** to drive Vulkan compute, detected by import
  probing (no import side effects).

Both must be present for :func:`detect_vulkan` to report ``available`` — i.e.
*worth attempting*. A backend still enumerates devices itself before it commits
to registering. The probes are injectable so the logic is fully unit-testable
on a host with no Vulkan at all (such as the CPU-only development floor), where
the preflight correctly reports *unavailable* with a specific reason.
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
    """Result of the Vulkan dependency preflight for a host.

    Attributes:
        available: ``True`` when the dependencies a Vulkan backend needs — both
            a Vulkan ICD loader and a supported Python binding — are present, so
            attempting the backend is worthwhile. It does **not** guarantee a
            usable GPU device; the backend confirms that itself. When
            ``False``, ``reason`` explains why.
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
        try:
            spec = importlib.util.find_spec(name)
        except ValueError:
            # A stale/partly-initialised entry in ``sys.modules`` (missing or
            # ``None`` ``__spec__``) makes ``find_spec`` raise; treat that
            # candidate as unavailable and keep probing the rest.
            continue
        if spec is not None:
            return name
    return None


def detect_vulkan(
    *,
    find_loader: Callable[[], str | None] = _default_find_loader,
    find_binding: Callable[[], str | None] = _default_find_binding,
) -> VulkanCapability:
    """Preflight the host for the dependencies a Vulkan backend needs.

    Reports whether the Vulkan ICD loader and a supported Python binding are
    present. This is a *necessary* precondition, not a guarantee that a GPU
    device will enumerate — a backend performs its own device probe at
    registration. The two probes are injectable so the logic can be exercised
    without a Vulkan runtime installed (and so a backend can supply stricter
    probes later). Defaults inspect the system Vulkan loader and the installed
    Python bindings.

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
    """Return ``True`` when the Vulkan backend dependencies are present.

    Convenience wrapper over :func:`detect_vulkan` for the common yes/no
    preflight (e.g. deciding whether it is worth a Vulkan-backed backend
    attempting device enumeration). ``True`` means the loader and a supported
    binding are installed, not that a GPU device is guaranteed usable.
    """
    return detect_vulkan().available
