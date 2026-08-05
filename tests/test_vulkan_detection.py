"""Tests for Vulkan device detection."""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Any

import pytest

import portable_attention as pa
from portable_attention import vulkan as vk

_INSTANCE_HANDLE = 0xBEEF


class _FakeLib:
    """Stand-in for ``libvulkan`` loaded through :class:`ctypes.CDLL`.

    Writes into the ctypes out-parameters exactly as the real entry points do,
    which lets the enumeration path be exercised on hosts without Vulkan (and
    lets failure modes be forced deterministically).
    """

    def __init__(
        self,
        devices: tuple[tuple[str, int, tuple[int, ...]], ...] = (),
        *,
        create_status: int = 0,
        create_handle: int = _INSTANCE_HANDLE,
        count_status: int = 0,
        fill_status: int = 0,
    ) -> None:
        # Each device is (name, packed api version, queue family flags).
        self.devices = devices
        self.create_status = create_status
        self.create_handle = create_handle
        self.count_status = count_status
        self.fill_status = fill_status
        self.destroyed = 0

    def vkCreateInstance(  # noqa: N802 - mirrors the Vulkan entry point name
        self, create_info: Any, allocator: Any, instance: Any
    ) -> int:
        assert create_info.contents.sType == 1
        assert allocator is None
        instance[0] = self.create_handle
        return self.create_status

    def vkEnumeratePhysicalDevices(  # noqa: N802
        self, instance: Any, count: Any, handles: Any
    ) -> int:
        assert instance.value == _INSTANCE_HANDLE
        if handles is None:
            count[0] = len(self.devices)
            return self.count_status
        for index in range(count[0]):
            # Handles are opaque; index+1 keeps them non-NULL and distinct.
            handles[index] = index + 1
        return self.fill_status

    def vkGetPhysicalDeviceProperties(self, device: Any, props: Any) -> None:  # noqa: N802
        name, api_version, _ = self.devices[device.value - 1]
        props.contents.deviceName = name.encode()
        props.contents.apiVersion = api_version

    def vkGetPhysicalDeviceQueueFamilyProperties(  # noqa: N802
        self, device: Any, count: Any, families: Any
    ) -> None:
        flags = self.devices[device.value - 1][2]
        if families is None:
            count[0] = len(flags)
            return
        for index, value in enumerate(flags):
            families[index].queueFlags = value

    def vkDestroyInstance(self, instance: Any, allocator: Any) -> None:  # noqa: N802
        assert allocator is None
        self.destroyed += 1


def _probe(devices: tuple[vk.VulkanDevice, ...]) -> vk.VulkanCapability:
    """Run detection against a fake loader exposing ``devices``."""
    return vk.detect_vulkan(
        find_loader=lambda: "libvulkan.so.1",
        probe_devices=lambda loader: devices,
    )


_COMPUTE = vk.VulkanDevice(name="V3D 7.1.7.0", api_version="1.2.289", compute=True)
_GRAPHICS_ONLY = vk.VulkanDevice(
    name="display-only", api_version="1.1.0", compute=False
)


def test_public_reexports() -> None:
    """Detection helpers are exported from the top-level package."""
    assert pa.detect_vulkan is vk.detect_vulkan
    assert pa.vulkan_available is vk.vulkan_available
    assert pa.VulkanCapability is vk.VulkanCapability
    assert pa.VulkanDevice is vk.VulkanDevice
    for name in (
        "detect_vulkan",
        "vulkan_available",
        "VulkanCapability",
        "VulkanDevice",
    ):
        assert name in pa.__all__


def test_no_loader_reports_unavailable_with_reason() -> None:
    """Without an ICD loader there is nothing to enumerate."""
    probed: list[str] = []

    def probe(loader: str) -> tuple[vk.VulkanDevice, ...]:
        probed.append(loader)
        return ()

    cap = vk.detect_vulkan(find_loader=lambda: None, probe_devices=probe)
    assert cap.available is False
    assert cap.loader is None
    assert cap.devices == ()
    assert cap.device_count == 0
    assert cap.reason is not None
    assert "loader" in cap.reason
    # Enumeration is skipped entirely when there is no library to enumerate with.
    assert probed == []


def test_loader_without_devices_reports_unavailable() -> None:
    """A loader that enumerates nothing (no ICD, no GPU) is unavailable."""
    cap = _probe(())
    assert cap.available is False
    assert cap.loader == "libvulkan.so.1"
    assert cap.devices == ()
    assert cap.reason is not None
    assert "no physical device" in cap.reason


def test_devices_without_compute_queue_report_unavailable() -> None:
    """Devices are reported even when none of them can run compute."""
    cap = _probe((_GRAPHICS_ONLY,))
    assert cap.available is False
    assert cap.devices == (_GRAPHICS_ONLY,)
    assert cap.device_names == ("display-only",)
    assert cap.reason is not None
    assert "compute queue" in cap.reason


def test_compute_capable_device_reports_available() -> None:
    """One compute-capable device is enough for the host to be available."""
    cap = _probe((_GRAPHICS_ONLY, _COMPUTE))
    assert cap.available is True
    assert cap.loader == "libvulkan.so.1"
    assert cap.device_count == 2
    assert cap.device_names == ("display-only", "V3D 7.1.7.0")
    assert cap.reason is None


def test_vulkan_available_matches_detect(monkeypatch: pytest.MonkeyPatch) -> None:
    """``vulkan_available`` is a thin bool wrapper over ``detect_vulkan``."""
    available = vk.VulkanCapability(
        available=True, loader="libvulkan.so.1", devices=(_COMPUTE,)
    )
    monkeypatch.setattr(vk, "detect_vulkan", lambda: available)
    assert vk.vulkan_available() is True

    monkeypatch.setattr(
        vk,
        "detect_vulkan",
        lambda: vk.VulkanCapability(available=False, loader=None, reason="none"),
    )
    assert vk.vulkan_available() is False


@pytest.mark.parametrize(
    "value",
    [
        vk.VulkanCapability(available=True, loader="libvulkan.so.1"),
        vk.VulkanDevice(name="V3D", api_version="1.2.289", compute=True),
    ],
)
def test_results_are_frozen(value: object) -> None:
    """Detection results are immutable value objects."""
    with pytest.raises(AttributeError):
        value.name = "mutated"  # type: ignore[misc]


def test_default_find_loader_probes_vulkan(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default loader probe queries ``find_library("vulkan")`` verbatim."""
    calls: list[str] = []

    def fake_find_library(name: str) -> str:
        calls.append(name)
        return "libvulkan.so.sentinel"

    monkeypatch.setattr(ctypes.util, "find_library", fake_find_library)
    assert vk._default_find_loader() == "libvulkan.so.sentinel"
    assert calls == ["vulkan"]


def _install_lib(monkeypatch: pytest.MonkeyPatch, lib: _FakeLib) -> _FakeLib:
    """Make ``ctypes.CDLL`` hand back ``lib`` for the duration of a test."""
    monkeypatch.setattr(ctypes, "CDLL", lambda name: lib)
    return lib


def test_probe_devices_reads_names_versions_and_queues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ctypes probe reports each device's name, version and compute flag."""
    lib = _install_lib(
        monkeypatch,
        _FakeLib(
            (
                # Packed VK_MAKE_VERSION(1, 2, 289) and VK_MAKE_VERSION(1, 3, 289).
                ("V3D 7.1.7.0", (1 << 22) | (2 << 12) | 289, (0x1 | 0x2 | 0x4,)),
                ("llvmpipe", (1 << 22) | (3 << 12) | 289, (0x1, 0x2)),
                ("display-only", (1 << 22) | (0 << 12) | 0, (0x1,)),
                ("headless", 1 << 22, ()),
            )
        ),
    )

    devices = vk._default_probe_devices("libvulkan.so.1")

    assert devices == (
        vk.VulkanDevice(name="V3D 7.1.7.0", api_version="1.2.289", compute=True),
        vk.VulkanDevice(name="llvmpipe", api_version="1.3.289", compute=True),
        vk.VulkanDevice(name="display-only", api_version="1.0.0", compute=False),
        vk.VulkanDevice(name="headless", api_version="1.0.0", compute=False),
    )
    # The throwaway instance is always released, including on the success path.
    assert lib.destroyed == 1


def test_probe_devices_handles_unloadable_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loader name that will not load reports no devices instead of raising."""

    def boom(name: str) -> object:
        raise OSError("cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", boom)
    assert vk._default_probe_devices("libvulkan.so.1") == ()


def test_probe_devices_handles_missing_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A library without ``vkCreateInstance`` is treated as having no devices."""
    monkeypatch.setattr(ctypes, "CDLL", lambda name: object())
    assert vk._default_probe_devices("libvulkan.so.1") == ()


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"create_status": -1}, "instance creation fails"),
        ({"create_handle": 0}, "instance handle comes back NULL"),
    ],
)
def test_probe_devices_handles_instance_failures(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, int], reason: str
) -> None:
    """No instance means no enumeration, whatever shape the failure takes."""
    lib = _install_lib(monkeypatch, _FakeLib((("V3D", 1 << 22, (0x2,)),), **kwargs))
    assert vk._default_probe_devices("libvulkan.so.1") == (), reason
    assert lib.destroyed == 0


@pytest.mark.parametrize("kwargs", [{"count_status": -1}, {"fill_status": -1}])
def test_probe_devices_handles_enumeration_failures(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, int]
) -> None:
    """A failing enumeration call yields no devices, and still frees the instance."""
    lib = _install_lib(monkeypatch, _FakeLib((("V3D", 1 << 22, (0x2,)),), **kwargs))
    assert vk._default_probe_devices("libvulkan.so.1") == ()
    assert lib.destroyed == 1


def test_probe_devices_handles_partial_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loader missing a symbol mid-enumeration still reports no devices."""

    class _PartialLib(_FakeLib):
        def __getattribute__(self, name: str) -> Any:
            if name == "vkGetPhysicalDeviceProperties":
                raise AttributeError(name)
            return super().__getattribute__(name)

    lib = _install_lib(monkeypatch, _PartialLib((("V3D", 1 << 22, (0x2,)),)))
    assert vk._default_probe_devices("libvulkan.so.1") == ()
    assert lib.destroyed == 1


def test_probe_devices_survives_failing_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Devices found are still returned when the instance cannot be destroyed."""

    class _UndestroyableLib(_FakeLib):
        def vkDestroyInstance(self, instance: Any, allocator: Any) -> None:  # noqa: N802
            raise AttributeError("vkDestroyInstance")

    _install_lib(monkeypatch, _UndestroyableLib((("V3D", 1 << 22, (0x2,)),)))
    assert vk._default_probe_devices("libvulkan.so.1") == (
        vk.VulkanDevice(name="V3D", api_version="1.0.0", compute=True),
    )


def test_probe_devices_handles_zero_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loader with no ICD enumerates successfully but reports nothing."""
    lib = _install_lib(monkeypatch, _FakeLib(()))
    assert vk._default_probe_devices("libvulkan.so.1") == ()
    assert lib.destroyed == 1


def test_detect_defaults_run_on_host() -> None:
    """Calling ``detect_vulkan`` with no injection exercises the real probes."""
    cap = vk.detect_vulkan()
    assert isinstance(cap, vk.VulkanCapability)
    assert isinstance(cap.available, bool)
    assert cap.device_count == len(cap.device_names)
    if cap.available:
        assert any(device.compute for device in cap.devices)
    else:
        assert cap.reason is not None
