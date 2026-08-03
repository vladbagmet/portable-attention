"""Vulkan device detection.

The M2 goal is a portable GPU backend built on **Vulkan (V3DV)** — see
``ROADMAP.md``. This module answers, on the host it is imported on, the question
that backend has to ask before it registers itself: *is there a Vulkan device
here that can run compute?* It enumerates real devices through the system
Vulkan loader with :mod:`ctypes`, so it adds no runtime dependency — in
particular no Python Vulkan binding, which a numpy-only package has no business
requiring.

Detection runs the minimal Vulkan sequence:

* locate the **ICD loader** (the ``libvulkan`` shared library) with
  :func:`ctypes.util.find_library` and load it;
* create a throwaway instance (``vkCreateInstance``) with no layers or
  extensions and enumerate physical devices;
* read each device's name and API version, and its queue families, recording
  whether any family advertises ``VK_QUEUE_COMPUTE_BIT``;
* destroy the instance.

A host is :attr:`~VulkanCapability.available` when at least one enumerated
device exposes a compute-capable queue family. Devices are still reported when
they cannot compute, so an unavailable result can say why. No device is opened
and no GPU work is submitted.

The loader and enumeration probes are injectable, so every branch is testable on
a host with no Vulkan at all (or with a device that has no compute queue).
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "VulkanCapability",
    "VulkanDevice",
    "detect_vulkan",
    "vulkan_available",
]

_VK_SUCCESS = 0
_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
_VK_QUEUE_COMPUTE_BIT = 0x00000002
_VK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256

_NO_LOADER_REASON = (
    "no Vulkan ICD loader found (libvulkan); install a Vulkan runtime such as Mesa V3DV"
)
_NO_DEVICE_REASON = "Vulkan loader present but no physical device enumerated"
_NO_COMPUTE_REASON = "Vulkan device(s) present but none exposes a compute queue family"


@dataclass(frozen=True)
class VulkanDevice:
    """A physical device reported by the Vulkan loader.

    Attributes:
        name: The device name string (``deviceName``), e.g. ``"V3D 7.1.7.0"``.
        api_version: The device's supported Vulkan version, formatted
            ``"major.minor.patch"`` (e.g. ``"1.2.289"``).
        compute: ``True`` when at least one of the device's queue families
            advertises ``VK_QUEUE_COMPUTE_BIT``.
    """

    name: str
    api_version: str
    compute: bool


@dataclass(frozen=True)
class VulkanCapability:
    """Result of Vulkan device detection for a host.

    Attributes:
        available: ``True`` when the loader was found and at least one
            enumerated device exposes a compute-capable queue family — i.e. a
            Vulkan compute backend has somewhere to run. When ``False``,
            ``reason`` explains why.
        loader: The ICD loader library name reported by the system (e.g.
            ``"libvulkan.so.1"``), or ``None`` when no loader was found.
        devices: Every physical device the loader enumerated, compute-capable
            or not, in enumeration order.
        reason: A human-readable explanation when ``available`` is ``False``;
            ``None`` otherwise.
    """

    available: bool
    loader: str | None
    devices: tuple[VulkanDevice, ...] = ()
    reason: str | None = None

    @property
    def device_count(self) -> int:
        """Number of physical devices enumerated (0 when none or no loader)."""
        return len(self.devices)

    @property
    def device_names(self) -> tuple[str, ...]:
        """Names of the enumerated devices, in enumeration order."""
        return tuple(device.name for device in self.devices)


class _VkInstanceCreateInfo(ctypes.Structure):
    """``VkInstanceCreateInfo`` with no layers, extensions or application info."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.c_void_p),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    )


class _VkPhysicalDeviceProperties(ctypes.Structure):
    """Head of ``VkPhysicalDeviceProperties`` plus room for the rest.

    Only ``apiVersion`` and ``deviceName`` are read here. The trailing
    ``limits``/``sparseProperties`` members are large, version-stable in layout
    but tedious to mirror, so they are reserved as opaque bytes; the reserve is
    deliberately larger than any published size of the struct so the driver
    always writes inside our allocation.
    """

    _fields_ = (
        ("apiVersion", ctypes.c_uint32),
        ("driverVersion", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
        ("deviceType", ctypes.c_uint32),
        ("deviceName", ctypes.c_char * _VK_MAX_PHYSICAL_DEVICE_NAME_SIZE),
        ("pipelineCacheUUID", ctypes.c_uint8 * 16),
        ("_reserved", ctypes.c_uint8 * 2048),
    )


class _VkQueueFamilyProperties(ctypes.Structure):
    """``VkQueueFamilyProperties`` (all members are 32-bit unsigned)."""

    _fields_ = (
        ("queueFlags", ctypes.c_uint32),
        ("queueCount", ctypes.c_uint32),
        ("timestampValidBits", ctypes.c_uint32),
        ("minImageTransferGranularityWidth", ctypes.c_uint32),
        ("minImageTransferGranularityHeight", ctypes.c_uint32),
        ("minImageTransferGranularityDepth", ctypes.c_uint32),
    )


def _format_api_version(packed: int) -> str:
    """Format a packed Vulkan ``VK_MAKE_VERSION`` integer as ``"x.y.z"``."""
    return f"{packed >> 22}.{(packed >> 12) & 0x3FF}.{packed & 0xFFF}"


def _default_find_loader() -> str | None:
    """Return the Vulkan ICD loader library name, or ``None`` if absent."""
    return ctypes.util.find_library("vulkan")


def _has_compute_queue(lib: ctypes.CDLL, device: ctypes.c_void_p) -> bool:
    """Return ``True`` when a queue family of ``device`` supports compute."""
    count = ctypes.c_uint32(0)
    lib.vkGetPhysicalDeviceQueueFamilyProperties(device, ctypes.pointer(count), None)
    if count.value == 0:
        return False
    families = (_VkQueueFamilyProperties * count.value)()
    lib.vkGetPhysicalDeviceQueueFamilyProperties(
        device, ctypes.pointer(count), families
    )
    return any(
        family.queueFlags & _VK_QUEUE_COMPUTE_BIT for family in families[: count.value]
    )


def _describe_device(lib: ctypes.CDLL, device: ctypes.c_void_p) -> VulkanDevice:
    """Read one physical device's name, API version and compute capability."""
    props = _VkPhysicalDeviceProperties()
    lib.vkGetPhysicalDeviceProperties(device, ctypes.pointer(props))
    return VulkanDevice(
        name=props.deviceName.decode("utf-8", "replace"),
        api_version=_format_api_version(props.apiVersion),
        compute=_has_compute_queue(lib, device),
    )


def _enumerate_devices(
    lib: ctypes.CDLL, instance: ctypes.c_void_p
) -> list[VulkanDevice]:
    """Describe every physical device visible to ``instance``."""
    count = ctypes.c_uint32(0)
    counted = lib.vkEnumeratePhysicalDevices(instance, ctypes.pointer(count), None)
    if counted != _VK_SUCCESS or count.value == 0:
        return []
    handles = (ctypes.c_void_p * count.value)()
    filled = lib.vkEnumeratePhysicalDevices(instance, ctypes.pointer(count), handles)
    if filled != _VK_SUCCESS:
        return []
    return [
        _describe_device(lib, ctypes.c_void_p(handle))
        for handle in handles[: count.value]
    ]


def _destroy_instance(lib: ctypes.CDLL, instance: ctypes.c_void_p) -> None:
    """Release the throwaway instance, tolerating a loader that cannot."""
    with contextlib.suppress(AttributeError, OSError):
        lib.vkDestroyInstance(instance, None)


def _default_probe_devices(loader: str) -> tuple[VulkanDevice, ...]:
    """Enumerate Vulkan devices through ``loader``; empty on any failure.

    Creates a temporary instance with no layers or extensions, describes each
    physical device it can see, and destroys the instance again. Every failure
    mode — an unloadable library, a missing entry point, a driver that refuses
    to create an instance — is reported as "no devices" rather than raised:
    detection must never take down an import on a broken install.
    """
    try:
        lib = ctypes.CDLL(loader)
    except OSError:
        return ()
    create_info = _VkInstanceCreateInfo(sType=_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
    instance = ctypes.c_void_p()
    try:
        status = lib.vkCreateInstance(
            ctypes.pointer(create_info), None, ctypes.pointer(instance)
        )
        if status != _VK_SUCCESS or not instance:
            return ()
        try:
            return tuple(_enumerate_devices(lib, instance))
        finally:
            _destroy_instance(lib, instance)
    except (AttributeError, OSError):
        return ()


def detect_vulkan(
    *,
    find_loader: Callable[[], str | None] = _default_find_loader,
    probe_devices: Callable[[str], tuple[VulkanDevice, ...]] = _default_probe_devices,
) -> VulkanCapability:
    """Detect whether this host has a Vulkan device that can run compute.

    Locates the Vulkan ICD loader and, if present, enumerates physical devices
    through it, recording each device's name, API version and whether it
    exposes a compute-capable queue family. Both steps are injectable so the
    logic can be exercised on hosts without Vulkan (and so a backend can supply
    a stricter probe later).

    Args:
        find_loader: Callable returning the ICD loader library name, or ``None``
            when no Vulkan loader is present.
        probe_devices: Callable taking the loader name and returning the devices
            it enumerates; an empty tuple means none were found.

    Returns:
        A :class:`VulkanCapability` describing the result. ``available`` is
        ``True`` only when some enumerated device reports ``compute``.
    """
    loader = find_loader()
    if loader is None:
        return VulkanCapability(available=False, loader=None, reason=_NO_LOADER_REASON)
    devices = probe_devices(loader)
    if not devices:
        return VulkanCapability(
            available=False, loader=loader, reason=_NO_DEVICE_REASON
        )
    if not any(device.compute for device in devices):
        return VulkanCapability(
            available=False, loader=loader, devices=devices, reason=_NO_COMPUTE_REASON
        )
    return VulkanCapability(available=True, loader=loader, devices=devices)


def vulkan_available() -> bool:
    """Return ``True`` when this host has a compute-capable Vulkan device.

    Convenience wrapper over :func:`detect_vulkan` for the common yes/no check
    (e.g. deciding whether a Vulkan-backed backend should register).
    """
    return detect_vulkan().available
