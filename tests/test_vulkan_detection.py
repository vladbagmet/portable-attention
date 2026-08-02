"""Tests for Vulkan runtime capability detection."""

from __future__ import annotations

import importlib.util

import pytest

import portable_attention as pa
from portable_attention import vulkan as vk


def test_public_reexports() -> None:
    """Detection helpers are exported from the top-level package."""
    assert pa.detect_vulkan is vk.detect_vulkan
    assert pa.vulkan_available is vk.vulkan_available
    assert pa.VulkanCapability is vk.VulkanCapability
    for name in ("detect_vulkan", "vulkan_available", "VulkanCapability"):
        assert name in pa.__all__


def test_no_loader_reports_unavailable_with_reason() -> None:
    """Without an ICD loader the runtime is unavailable, binding still probed."""
    cap = vk.detect_vulkan(
        find_loader=lambda: None,
        find_binding=lambda: "kp",
    )
    assert cap.available is False
    assert cap.loader is None
    # The binding probe still runs, so its result is reported for diagnostics.
    assert cap.binding == "kp"
    assert cap.reason is not None
    assert "loader" in cap.reason


def test_loader_without_binding_reports_unavailable() -> None:
    """A loader but no Python binding is unavailable and names the tried set."""
    cap = vk.detect_vulkan(
        find_loader=lambda: "libvulkan.so.1",
        find_binding=lambda: None,
    )
    assert cap.available is False
    assert cap.loader == "libvulkan.so.1"
    assert cap.binding is None
    assert cap.reason is not None
    for name in vk._KNOWN_BINDINGS:
        assert name in cap.reason


def test_loader_and_binding_reports_available() -> None:
    """Both probes succeeding yields an available capability with no reason."""
    cap = vk.detect_vulkan(
        find_loader=lambda: "libvulkan.so.1",
        find_binding=lambda: "vulkan",
    )
    assert cap.available is True
    assert cap.loader == "libvulkan.so.1"
    assert cap.binding == "vulkan"
    assert cap.reason is None


def test_vulkan_available_matches_detect(monkeypatch: pytest.MonkeyPatch) -> None:
    """``vulkan_available`` is a thin bool wrapper over ``detect_vulkan``."""
    available = vk.VulkanCapability(
        available=True, loader="libvulkan.so.1", binding="kp", reason=None
    )
    monkeypatch.setattr(vk, "detect_vulkan", lambda: available)
    assert vk.vulkan_available() is True

    unavailable = vk.VulkanCapability(
        available=False, loader=None, binding=None, reason="none"
    )
    monkeypatch.setattr(vk, "detect_vulkan", lambda: unavailable)
    assert vk.vulkan_available() is False


def test_capability_is_frozen() -> None:
    """``VulkanCapability`` is an immutable value object."""
    cap = vk.VulkanCapability(
        available=True, loader="libvulkan.so.1", binding="kp", reason=None
    )
    with pytest.raises(AttributeError):
        cap.available = False  # type: ignore[misc]


def test_default_find_binding_hits_and_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default binding probe returns the first import-locatable name."""
    real_find_spec = importlib.util.find_spec

    def only_second(name: str) -> object:
        # Report the first known binding missing and the second present, so the
        # loop's continue-then-return path is exercised without importing it.
        if name == vk._KNOWN_BINDINGS[0]:
            return None
        if name == vk._KNOWN_BINDINGS[1]:
            return object()
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", only_second)
    assert vk._default_find_binding() == vk._KNOWN_BINDINGS[1]

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert vk._default_find_binding() is None


def test_default_find_loader_returns_str_or_none() -> None:
    """The default loader probe returns whatever the host reports, unmodified."""
    result = vk._default_find_loader()
    assert result is None or isinstance(result, str)


def test_detect_defaults_run_on_host() -> None:
    """Calling ``detect_vulkan`` with no injection exercises the real probes."""
    cap = vk.detect_vulkan()
    assert isinstance(cap, vk.VulkanCapability)
    assert isinstance(cap.available, bool)
    if not cap.available:
        assert cap.reason is not None
