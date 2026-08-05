"""Shared ctypes bindings for the small slice of Vulkan this package uses.

Private module (hence the underscore): it holds the constants, ``ctypes``
structures and helpers that both :mod:`portable_attention.vulkan` (detection)
and :mod:`portable_attention.vkcompute` (device, queue and buffers) need, so the
struct layouts are declared exactly once. Nothing here is public API.

Only the entry points and members those two modules read are mirrored. Layouts
follow the Vulkan 1.0 headers and have been stable across every later version.
"""

from __future__ import annotations

import ctypes
import ctypes.util

VK_SUCCESS = 0
VK_QUEUE_COMPUTE_BIT = 0x00000002
VK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1


class VkInstanceCreateInfo(ctypes.Structure):
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


class VkPhysicalDeviceProperties(ctypes.Structure):
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
        ("deviceName", ctypes.c_char * VK_MAX_PHYSICAL_DEVICE_NAME_SIZE),
        ("pipelineCacheUUID", ctypes.c_uint8 * 16),
        ("_reserved", ctypes.c_uint8 * 2048),
    )


class VkQueueFamilyProperties(ctypes.Structure):
    """``VkQueueFamilyProperties`` (all members are 32-bit unsigned)."""

    _fields_ = (
        ("queueFlags", ctypes.c_uint32),
        ("queueCount", ctypes.c_uint32),
        ("timestampValidBits", ctypes.c_uint32),
        ("minImageTransferGranularityWidth", ctypes.c_uint32),
        ("minImageTransferGranularityHeight", ctypes.c_uint32),
        ("minImageTransferGranularityDepth", ctypes.c_uint32),
    )


def format_api_version(packed: int) -> str:
    """Format a packed Vulkan ``VK_MAKE_VERSION`` integer as ``"x.y.z"``."""
    return f"{packed >> 22}.{(packed >> 12) & 0x3FF}.{packed & 0xFFF}"


def find_loader_name() -> str | None:
    """Return the Vulkan ICD loader library name, or ``None`` if absent."""
    return ctypes.util.find_library("vulkan")
