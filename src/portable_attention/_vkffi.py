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


class VkPhysicalDeviceLimits(ctypes.Structure):
    """``VkPhysicalDeviceLimits`` up to the compute members, then opaque bytes.

    The struct has more than a hundred members and this package reads four of
    them, but ctypes computes offsets from the declaration, so every member in
    front of the compute block has to be spelled out in order. The members
    behind ``maxComputeWorkGroupSize`` are never read and are reserved as bytes.

    Two of the leading members are ``VkDeviceSize`` (64-bit), which is why this
    is its own structure rather than fields flattened into
    :class:`VkPhysicalDeviceProperties`: as a nested member it inherits the
    8-byte alignment the C layout gives it.
    """

    _fields_ = (
        ("maxImageDimension1D", ctypes.c_uint32),
        ("maxImageDimension2D", ctypes.c_uint32),
        ("maxImageDimension3D", ctypes.c_uint32),
        ("maxImageDimensionCube", ctypes.c_uint32),
        ("maxImageArrayLayers", ctypes.c_uint32),
        ("maxTexelBufferElements", ctypes.c_uint32),
        ("maxUniformBufferRange", ctypes.c_uint32),
        ("maxStorageBufferRange", ctypes.c_uint32),
        ("maxPushConstantsSize", ctypes.c_uint32),
        ("maxMemoryAllocationCount", ctypes.c_uint32),
        ("maxSamplerAllocationCount", ctypes.c_uint32),
        ("bufferImageGranularity", ctypes.c_uint64),
        ("sparseAddressSpaceSize", ctypes.c_uint64),
        ("maxBoundDescriptorSets", ctypes.c_uint32),
        ("maxPerStageDescriptorSamplers", ctypes.c_uint32),
        ("maxPerStageDescriptorUniformBuffers", ctypes.c_uint32),
        ("maxPerStageDescriptorStorageBuffers", ctypes.c_uint32),
        ("maxPerStageDescriptorSampledImages", ctypes.c_uint32),
        ("maxPerStageDescriptorStorageImages", ctypes.c_uint32),
        ("maxPerStageDescriptorInputAttachments", ctypes.c_uint32),
        ("maxPerStageResources", ctypes.c_uint32),
        ("maxDescriptorSetSamplers", ctypes.c_uint32),
        ("maxDescriptorSetUniformBuffers", ctypes.c_uint32),
        ("maxDescriptorSetUniformBuffersDynamic", ctypes.c_uint32),
        ("maxDescriptorSetStorageBuffers", ctypes.c_uint32),
        ("maxDescriptorSetStorageBuffersDynamic", ctypes.c_uint32),
        ("maxDescriptorSetSampledImages", ctypes.c_uint32),
        ("maxDescriptorSetStorageImages", ctypes.c_uint32),
        ("maxDescriptorSetInputAttachments", ctypes.c_uint32),
        ("maxVertexInputAttributes", ctypes.c_uint32),
        ("maxVertexInputBindings", ctypes.c_uint32),
        ("maxVertexInputAttributeOffset", ctypes.c_uint32),
        ("maxVertexInputBindingStride", ctypes.c_uint32),
        ("maxVertexOutputComponents", ctypes.c_uint32),
        ("maxTessellationGenerationLevel", ctypes.c_uint32),
        ("maxTessellationPatchSize", ctypes.c_uint32),
        ("maxTessellationControlPerVertexInputComponents", ctypes.c_uint32),
        ("maxTessellationControlPerVertexOutputComponents", ctypes.c_uint32),
        ("maxTessellationControlPerPatchOutputComponents", ctypes.c_uint32),
        ("maxTessellationControlTotalOutputComponents", ctypes.c_uint32),
        ("maxTessellationEvaluationInputComponents", ctypes.c_uint32),
        ("maxTessellationEvaluationOutputComponents", ctypes.c_uint32),
        ("maxGeometryShaderInvocations", ctypes.c_uint32),
        ("maxGeometryInputComponents", ctypes.c_uint32),
        ("maxGeometryOutputComponents", ctypes.c_uint32),
        ("maxGeometryOutputVertices", ctypes.c_uint32),
        ("maxGeometryTotalOutputComponents", ctypes.c_uint32),
        ("maxFragmentInputComponents", ctypes.c_uint32),
        ("maxFragmentOutputAttachments", ctypes.c_uint32),
        ("maxFragmentDualSrcAttachments", ctypes.c_uint32),
        ("maxFragmentCombinedOutputResources", ctypes.c_uint32),
        ("maxComputeSharedMemorySize", ctypes.c_uint32),
        ("maxComputeWorkGroupCount", ctypes.c_uint32 * 3),
        ("maxComputeWorkGroupInvocations", ctypes.c_uint32),
        ("maxComputeWorkGroupSize", ctypes.c_uint32 * 3),
        ("_reserved", ctypes.c_uint8 * 1024),
    )


#: Byte offset of ``limits`` inside ``VkPhysicalDeviceProperties`` per the
#: Vulkan headers: five ``uint32`` members, a 256-byte name, a 16-byte UUID,
#: then padding to the 8-byte alignment ``VkPhysicalDeviceLimits`` requires.
VK_PHYSICAL_DEVICE_LIMITS_OFFSET = 296


class VkPhysicalDeviceProperties(ctypes.Structure):
    """Head of ``VkPhysicalDeviceProperties`` plus room for the rest.

    ``apiVersion``, ``deviceName`` and the compute members of ``limits`` are
    read here. ``sparseProperties`` trails ``limits`` and is never read, so it
    is reserved as opaque bytes; the reserve is deliberately larger than any
    published size of the struct so the driver always writes inside our
    allocation.
    """

    _fields_ = (
        ("apiVersion", ctypes.c_uint32),
        ("driverVersion", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
        ("deviceType", ctypes.c_uint32),
        ("deviceName", ctypes.c_char * VK_MAX_PHYSICAL_DEVICE_NAME_SIZE),
        ("pipelineCacheUUID", ctypes.c_uint8 * 16),
        ("limits", VkPhysicalDeviceLimits),
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
