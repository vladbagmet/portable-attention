"""Vulkan compute context: device, queue, host-visible buffers and dispatch.

:mod:`portable_attention.vulkan` answers *is there a Vulkan device here?*. This
module takes the next step the M2 backend needs: it **opens** one of those
devices, moves bytes through it, and runs SPIR-V compute shaders on them. It
creates a logical device with a single compute queue, allocates storage buffers
backed by host-visible coherent memory, and keeps that memory mapped so numpy
arrays can be copied in and out without a staging buffer or a command
submission.

Everything goes through the system Vulkan loader with :mod:`ctypes`, so the
package still depends on nothing but numpy. Host-visible storage buffers are
the right shape for the kernel that follows: V3D (and llvmpipe) expose memory
that is both ``DEVICE_LOCAL`` and ``HOST_VISIBLE``, so Q/K/V/O can live in one
allocation that the shader reads and the caller writes.

Typical use::

    with VulkanContext.open() as ctx:
        with ctx.allocate(q.nbytes) as buf:
            buf.write(q)
            same = buf.read(q.dtype, q.shape)

    with VulkanContext.open() as ctx:
        pipeline = ctx.compute_pipeline(spirv, buffer_count=2)
        pipeline.dispatch([src, dst], groups=(groups_x,))

Failures raise :class:`VulkanError` rather than returning a sentinel: by the
time a caller opens a device, detection has already said one is there, so a
failure here is exceptional. Resources are released in reverse order by
:meth:`VulkanContext.close`, which also frees buffers the caller forgot.
"""

from __future__ import annotations

import ctypes
import struct
from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Any, NamedTuple, Union, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray

from ._vkffi import (
    VK_QUEUE_COMPUTE_BIT,
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    VK_SUCCESS,
    VkInstanceCreateInfo,
    VkPhysicalDeviceProperties,
    VkQueueFamilyProperties,
    find_loader_name,
    format_api_version,
)

__all__ = [
    "VulkanBuffer",
    "VulkanContext",
    "VulkanError",
    "VulkanPipeline",
]

_VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2
_VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3
_VK_STRUCTURE_TYPE_SUBMIT_INFO = 4
_VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5
_VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8
_VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12
_VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16
_VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18
_VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29
_VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30
_VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32
_VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33
_VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34
_VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35
_VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39
_VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40
_VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42
_VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46

_VK_TIMEOUT = 2
_VK_TRUE = 1
_VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7
_VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020
_VK_PIPELINE_BIND_POINT_COMPUTE = 1
_VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0
_VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002
_VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800
_VK_PIPELINE_STAGE_HOST_BIT = 0x00004000
_VK_ACCESS_SHADER_WRITE_BIT = 0x00000040
_VK_ACCESS_HOST_READ_BIT = 0x00002000
_VK_WHOLE_SIZE = 0xFFFFFFFFFFFFFFFF

# First word of a little-endian SPIR-V module; the reverse is the same module
# with the opposite byte order, which no Vulkan driver accepts.
_SPIRV_MAGIC = 0x07230203
_SPIRV_MAGIC_SWAPPED = 0x03022307

# Vulkan guarantees at least this much push-constant space on every
# implementation, so staying inside it keeps a kernel portable by construction.
_MIN_MAX_PUSH_CONSTANTS_SIZE = 128
# Likewise the guaranteed floor for maxComputeWorkGroupCount on each axis. It is
# also well inside the 32-bit range vkCmdDispatch takes, so a count that passes
# cannot wrap on the way to the driver.
_MIN_MAX_WORKGROUP_COUNT = 65535

# Specialization constants are 32-bit scalars, so ids and integer values are
# bounded by what a uint32 (or an int32, for negatives) can hold.
_MAX_UINT32 = 0xFFFFFFFF
_INT32_MIN_MAGNITUDE = 0x80000000
_SPECIALIZATION_VALUE_BYTES = 4

# What one ``layout(constant_id = N)`` slot may hold. ``bool`` is an ``int``
# subclass, so it is listed for the reader rather than for the type checker.
SpecializationValue = Union[bool, int, float]

_VK_SHARING_MODE_EXCLUSIVE = 0
_VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001
_VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002
_VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020
_BUFFER_USAGE = (
    _VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    | _VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    | _VK_BUFFER_USAGE_TRANSFER_DST_BIT
)

_VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001
_VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002
_VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004
# Mapped writes must be visible to the device without an explicit flush, so
# coherent host memory is required, not merely preferred.
_REQUIRED_MEMORY_PROPERTIES = (
    _VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | _VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
)
# On a unified-memory part the same heap is also device-local; prefer it so the
# shader reads the allocation at full speed instead of over the bus.
_PREFERRED_MEMORY_PROPERTIES = _VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT

_VK_MAX_MEMORY_TYPES = 32
_VK_MAX_MEMORY_HEAPS = 16

_RESULT_NAMES = {
    -1: "VK_ERROR_OUT_OF_HOST_MEMORY",
    -2: "VK_ERROR_OUT_OF_DEVICE_MEMORY",
    -3: "VK_ERROR_INITIALIZATION_FAILED",
    -4: "VK_ERROR_DEVICE_LOST",
    -5: "VK_ERROR_MEMORY_MAP_FAILED",
    -9: "VK_ERROR_INCOMPATIBLE_DRIVER",
    -10: "VK_ERROR_TOO_MANY_OBJECTS",
    -13: "VK_ERROR_UNKNOWN",
    _VK_TIMEOUT: "VK_TIMEOUT",
}


class VulkanError(RuntimeError):
    """A Vulkan entry point failed, or the host cannot satisfy a request."""


class _VkDeviceQueueCreateInfo(ctypes.Structure):
    """``VkDeviceQueueCreateInfo`` for a single queue of one family."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("queueFamilyIndex", ctypes.c_uint32),
        ("queueCount", ctypes.c_uint32),
        ("pQueuePriorities", ctypes.POINTER(ctypes.c_float)),
    )


class _VkDeviceCreateInfo(ctypes.Structure):
    """``VkDeviceCreateInfo`` with no layers, extensions or enabled features."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("queueCreateInfoCount", ctypes.c_uint32),
        ("pQueueCreateInfos", ctypes.c_void_p),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
        ("pEnabledFeatures", ctypes.c_void_p),
    )


class _VkBufferCreateInfo(ctypes.Structure):
    """``VkBufferCreateInfo`` for an exclusively owned storage buffer."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("size", ctypes.c_uint64),
        ("usage", ctypes.c_uint32),
        ("sharingMode", ctypes.c_uint32),
        ("queueFamilyIndexCount", ctypes.c_uint32),
        ("pQueueFamilyIndices", ctypes.c_void_p),
    )


class _VkMemoryRequirements(ctypes.Structure):
    """``VkMemoryRequirements`` as returned for a buffer."""

    _fields_ = (
        ("size", ctypes.c_uint64),
        ("alignment", ctypes.c_uint64),
        ("memoryTypeBits", ctypes.c_uint32),
    )


class _VkMemoryType(ctypes.Structure):
    """One entry of ``VkPhysicalDeviceMemoryProperties.memoryTypes``."""

    _fields_ = (
        ("propertyFlags", ctypes.c_uint32),
        ("heapIndex", ctypes.c_uint32),
    )


class _VkMemoryHeap(ctypes.Structure):
    """One entry of ``VkPhysicalDeviceMemoryProperties.memoryHeaps``."""

    _fields_ = (
        ("size", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
    )


class _VkPhysicalDeviceMemoryProperties(ctypes.Structure):
    """``VkPhysicalDeviceMemoryProperties`` (fixed-size type and heap arrays)."""

    _fields_ = (
        ("memoryTypeCount", ctypes.c_uint32),
        ("memoryTypes", _VkMemoryType * _VK_MAX_MEMORY_TYPES),
        ("memoryHeapCount", ctypes.c_uint32),
        ("memoryHeaps", _VkMemoryHeap * _VK_MAX_MEMORY_HEAPS),
    )


class _VkMemoryAllocateInfo(ctypes.Structure):
    """``VkMemoryAllocateInfo`` for one allocation of a chosen memory type."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("allocationSize", ctypes.c_uint64),
        ("memoryTypeIndex", ctypes.c_uint32),
    )


class _VkShaderModuleCreateInfo(ctypes.Structure):
    """``VkShaderModuleCreateInfo`` pointing at a SPIR-V word array."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("codeSize", ctypes.c_size_t),
        ("pCode", ctypes.c_void_p),
    )


class _VkDescriptorSetLayoutBinding(ctypes.Structure):
    """``VkDescriptorSetLayoutBinding`` for one storage buffer."""

    _fields_ = (
        ("binding", ctypes.c_uint32),
        ("descriptorType", ctypes.c_uint32),
        ("descriptorCount", ctypes.c_uint32),
        ("stageFlags", ctypes.c_uint32),
        ("pImmutableSamplers", ctypes.c_void_p),
    )


class _VkDescriptorSetLayoutCreateInfo(ctypes.Structure):
    """``VkDescriptorSetLayoutCreateInfo`` over an array of bindings."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("bindingCount", ctypes.c_uint32),
        ("pBindings", ctypes.c_void_p),
    )


class _VkDescriptorPoolSize(ctypes.Structure):
    """``VkDescriptorPoolSize``: how many descriptors of one type to reserve."""

    _fields_ = (
        ("type", ctypes.c_uint32),
        ("descriptorCount", ctypes.c_uint32),
    )


class _VkDescriptorPoolCreateInfo(ctypes.Structure):
    """``VkDescriptorPoolCreateInfo`` for a pool holding a single set."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("maxSets", ctypes.c_uint32),
        ("poolSizeCount", ctypes.c_uint32),
        ("pPoolSizes", ctypes.c_void_p),
    )


class _VkDescriptorSetAllocateInfo(ctypes.Structure):
    """``VkDescriptorSetAllocateInfo`` for one set of a known layout."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("descriptorPool", ctypes.c_uint64),
        ("descriptorSetCount", ctypes.c_uint32),
        ("pSetLayouts", ctypes.c_void_p),
    )


class _VkDescriptorBufferInfo(ctypes.Structure):
    """``VkDescriptorBufferInfo`` naming the whole range of one buffer."""

    _fields_ = (
        ("buffer", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
        ("range", ctypes.c_uint64),
    )


class _VkWriteDescriptorSet(ctypes.Structure):
    """``VkWriteDescriptorSet`` binding one buffer to one binding number."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("dstSet", ctypes.c_uint64),
        ("dstBinding", ctypes.c_uint32),
        ("dstArrayElement", ctypes.c_uint32),
        ("descriptorCount", ctypes.c_uint32),
        ("descriptorType", ctypes.c_uint32),
        ("pImageInfo", ctypes.c_void_p),
        ("pBufferInfo", ctypes.c_void_p),
        ("pTexelBufferView", ctypes.c_void_p),
    )


class _VkSpecializationMapEntry(ctypes.Structure):
    """``VkSpecializationMapEntry``: one constant's id, offset and size."""

    _fields_ = (
        ("constantID", ctypes.c_uint32),
        ("offset", ctypes.c_uint32),
        ("size", ctypes.c_size_t),
    )


class _VkSpecializationInfo(ctypes.Structure):
    """``VkSpecializationInfo``: the map entries plus their packed data blob."""

    _fields_ = (
        ("mapEntryCount", ctypes.c_uint32),
        ("pMapEntries", ctypes.c_void_p),
        ("dataSize", ctypes.c_size_t),
        ("pData", ctypes.c_void_p),
    )


class _VkPushConstantRange(ctypes.Structure):
    """``VkPushConstantRange`` for the compute stage."""

    _fields_ = (
        ("stageFlags", ctypes.c_uint32),
        ("offset", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
    )


class _VkPipelineLayoutCreateInfo(ctypes.Structure):
    """``VkPipelineLayoutCreateInfo`` with one set layout and one range."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("setLayoutCount", ctypes.c_uint32),
        ("pSetLayouts", ctypes.c_void_p),
        ("pushConstantRangeCount", ctypes.c_uint32),
        ("pPushConstantRanges", ctypes.c_void_p),
    )


class _VkPipelineShaderStageCreateInfo(ctypes.Structure):
    """``VkPipelineShaderStageCreateInfo`` for a compute entry point."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("stage", ctypes.c_uint32),
        ("module", ctypes.c_uint64),
        ("pName", ctypes.c_char_p),
        ("pSpecializationInfo", ctypes.c_void_p),
    )


class _VkComputePipelineCreateInfo(ctypes.Structure):
    """``VkComputePipelineCreateInfo`` (the stage is embedded by value)."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("stage", _VkPipelineShaderStageCreateInfo),
        ("layout", ctypes.c_uint64),
        ("basePipelineHandle", ctypes.c_uint64),
        ("basePipelineIndex", ctypes.c_int32),
    )


class _VkCommandPoolCreateInfo(ctypes.Structure):
    """``VkCommandPoolCreateInfo`` for the compute queue family."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("queueFamilyIndex", ctypes.c_uint32),
    )


class _VkCommandBufferAllocateInfo(ctypes.Structure):
    """``VkCommandBufferAllocateInfo`` for one primary command buffer."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("commandPool", ctypes.c_uint64),
        ("level", ctypes.c_uint32),
        ("commandBufferCount", ctypes.c_uint32),
    )


class _VkCommandBufferBeginInfo(ctypes.Structure):
    """``VkCommandBufferBeginInfo`` for a primary command buffer."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pInheritanceInfo", ctypes.c_void_p),
    )


class _VkMemoryBarrier(ctypes.Structure):
    """``VkMemoryBarrier`` for a global execution/memory dependency."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("srcAccessMask", ctypes.c_uint32),
        ("dstAccessMask", ctypes.c_uint32),
    )


class _VkSubmitInfo(ctypes.Structure):
    """``VkSubmitInfo`` for one command buffer and no semaphores."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("waitSemaphoreCount", ctypes.c_uint32),
        ("pWaitSemaphores", ctypes.c_void_p),
        ("pWaitDstStageMask", ctypes.c_void_p),
        ("commandBufferCount", ctypes.c_uint32),
        ("pCommandBuffers", ctypes.c_void_p),
        ("signalSemaphoreCount", ctypes.c_uint32),
        ("pSignalSemaphores", ctypes.c_void_p),
    )


class _VkFenceCreateInfo(ctypes.Structure):
    """``VkFenceCreateInfo`` for an unsignalled fence."""

    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
    )


def _require_positive_int(value: object) -> None:
    """Reject a buffer size that is not a positive integer.

    Sizes are byte counts handed straight to the driver, so a float would price
    a fractional allocation and a ``bool`` would slip through a bare
    ``isinstance(..., int)`` as 1 byte.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VulkanError(f"buffer size must be a positive integer, got {value!r}")


def _require_spirv(code: object) -> bytes:
    """Return ``code`` as SPIR-V bytes, rejecting anything a driver would refuse.

    A malformed module is a caller mistake that Vulkan is entitled to answer
    with undefined behaviour, so the header is checked here: 4-byte words and
    the little-endian magic number.
    """
    if not isinstance(code, (bytes, bytearray, memoryview)):
        raise VulkanError(f"SPIR-V code must be bytes-like, got {type(code).__name__}")
    data = bytes(code)
    if len(data) < 4 or len(data) % 4:
        raise VulkanError(
            f"SPIR-V code must be a whole number of 32-bit words, got {len(data)} bytes"
        )
    magic = int.from_bytes(data[:4], "little")
    if magic == _SPIRV_MAGIC_SWAPPED:
        raise VulkanError("SPIR-V module is byte-swapped; recompile for this host")
    if magic != _SPIRV_MAGIC:
        raise VulkanError(
            f"not a SPIR-V module: first word is 0x{magic:08x}, "
            f"expected 0x{_SPIRV_MAGIC:08x}"
        )
    return data


def _is_positive_int(value: object) -> bool:
    """Return ``True`` for a positive integer that is not a ``bool``.

    Counts here become 32-bit unsigned arguments to Vulkan, so a float would be
    truncated and ``True`` would silently mean 1.
    """
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _require_buffer_count(value: object) -> int:
    """Reject a descriptor count that is not a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VulkanError(f"buffer_count must be a positive integer, got {value!r}")
    return value


def _require_push_constant_bytes(value: object) -> int:
    """Validate a push-constant block size against the portable Vulkan floor.

    Vulkan requires push-constant sizes to be a multiple of 4 and guarantees
    only 128 bytes of space, so a larger block would work on some drivers and
    fail on others.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VulkanError(
            f"push_constant_bytes must be a non-negative integer, got {value!r}"
        )
    if value % 4:
        raise VulkanError(f"push_constant_bytes must be a multiple of 4, got {value}")
    if value > _MIN_MAX_PUSH_CONSTANTS_SIZE:
        raise VulkanError(
            f"push_constant_bytes {value} exceeds the {_MIN_MAX_PUSH_CONSTANTS_SIZE}"
            " bytes every Vulkan implementation guarantees"
        )
    return value


def _require_specialization(specialization: object) -> dict[int, SpecializationValue]:
    """Validate a specialization mapping and order it by constant id.

    Constant ids are the ``layout(constant_id = N)`` numbers in the shader.
    Values must be ``bool``, ``int`` or ``float``: each occupies one 4-byte
    slot, matching the ``bool``/``int``/``uint``/``float`` scalar constants
    GLSL can specialize. An id the shader does not declare is ignored by the
    driver, so this cannot check ids against the module.
    """
    if specialization is None:
        return {}
    if not isinstance(specialization, Mapping):
        raise VulkanError(
            "specialization must be a mapping of constant id to value, got "
            f"{type(specialization).__name__}"
        )
    items = cast("Mapping[object, object]", specialization)
    entries: dict[int, SpecializationValue] = {}
    for raw_id, value in items.items():
        constant_id = _require_constant_id(raw_id)
        entries[constant_id] = _require_specialization_value(constant_id, value)
    return {constant_id: entries[constant_id] for constant_id in sorted(entries)}


def _require_constant_id(value: object) -> int:
    """Reject a constant id that is not a uint32."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise VulkanError(
            f"specialization constant id must be an integer, got {value!r}"
        )
    if not 0 <= value <= _MAX_UINT32:
        raise VulkanError(
            f"specialization constant id must fit in a uint32, got {value}"
        )
    return value


def _require_specialization_value(
    constant_id: int, value: object
) -> SpecializationValue:
    """Reject a specialization value that has no 4-byte scalar encoding."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if not -_INT32_MIN_MAGNITUDE <= value <= _MAX_UINT32:
            raise VulkanError(
                f"specialization constant {constant_id} value {value} does not "
                "fit in a 32-bit int"
            )
        return value
    if isinstance(value, float):
        return _to_binary32(constant_id, value)
    raise VulkanError(
        f"specialization constant {constant_id} must be a bool, int or float, "
        f"got {type(value).__name__}"
    )


def _to_binary32(constant_id: int, value: float) -> float:
    """Round a Python float to the binary32 the shader will actually see.

    Python floats are binary64; the slot is binary32. Storing the rounded value
    keeps :attr:`VulkanPipeline.specialization` honest about what was compiled
    in, and a magnitude with no binary32 at all is rejected here rather than
    escaping as a raw ``OverflowError`` from the packing step.
    """
    try:
        rounded: float = struct.unpack("=f", struct.pack("=f", value))[0]
    except OverflowError:
        raise VulkanError(
            f"specialization constant {constant_id} value {value} is outside "
            "the range of a 32-bit float"
        ) from None
    return rounded


def _pack_specialization_value(value: SpecializationValue) -> bytes:
    """Encode one constant into its 4-byte SPIR-V representation.

    ``bool`` comes first: it is an ``int`` subclass, and a specialized GLSL
    ``bool`` is a 32-bit 0/1. Non-negative ints pack as ``uint32`` and negative
    ones as ``int32``; both fill the same slot, and the shader's declared type
    decides how the bits are read.

    The formats are native-endian (``=``): Vulkan reads specialization data as
    the host's own representation of the constant, not as a byte stream with a
    fixed order.
    """
    if isinstance(value, bool):
        return struct.pack("=I", 1 if value else 0)
    if isinstance(value, int):
        return struct.pack("=i" if value < 0 else "=I", value)
    return struct.pack("=f", value)


def _require_groups(groups: int | Sequence[int]) -> tuple[int, int, int]:
    """Normalise a workgroup count to a 3-tuple, padding missing axes with 1."""
    counts = (groups,) if isinstance(groups, int) else tuple(groups)
    if not 1 <= len(counts) <= 3:
        raise VulkanError(f"groups needs 1 to 3 dimensions, got {len(counts)}")
    for count in counts:
        if not _is_positive_int(count):
            raise VulkanError(
                f"workgroup counts must be positive integers, got {counts}"
            )
        if count > _MIN_MAX_WORKGROUP_COUNT:
            raise VulkanError(
                f"workgroup count {count} exceeds the {_MIN_MAX_WORKGROUP_COUNT} "
                "per axis every Vulkan implementation guarantees"
            )
    padded = counts + (1,) * (3 - len(counts))
    return padded[0], padded[1], padded[2]


def _check(status: int, what: str) -> None:
    """Raise :class:`VulkanError` unless ``status`` is ``VK_SUCCESS``."""
    if status != VK_SUCCESS:
        name = _RESULT_NAMES.get(status, f"VkResult {status}")
        raise VulkanError(f"{what} failed: {name}")


def _select_memory_type(
    property_flags: Sequence[int],
    type_bits: int,
    *,
    required: int = _REQUIRED_MEMORY_PROPERTIES,
    preferred: int = _PREFERRED_MEMORY_PROPERTIES,
) -> int:
    """Return the index of the best memory type a buffer can be backed by.

    Args:
        property_flags: ``propertyFlags`` of each reported memory type, in
            index order.
        type_bits: The buffer's ``memoryTypeBits`` mask — bit *i* set means
            memory type *i* is allowed for this buffer.
        required: Property bits a candidate must have.
        preferred: Property bits that break ties; a candidate carrying them
            wins over one that does not.

    Returns:
        The index of the chosen memory type: the first allowed candidate with
        every ``preferred`` bit, else the first allowed candidate at all.

    Raises:
        VulkanError: When no allowed memory type carries the required bits.
    """
    fallback: int | None = None
    for index, flags in enumerate(property_flags):
        if not type_bits >> index & 1:
            continue
        if flags & required != required:
            continue
        if flags & preferred == preferred:
            return index
        if fallback is None:
            fallback = index
    if fallback is None:
        raise VulkanError(
            "no host-visible coherent memory type accepts this buffer "
            f"(memoryTypeBits=0x{type_bits:x})"
        )
    return fallback


def _compute_queue_family(lib: ctypes.CDLL, physical_device: ctypes.c_void_p) -> int:
    """Return the first queue family index of ``physical_device`` doing compute.

    Raises:
        VulkanError: When the device exposes no compute-capable family.
    """
    count = ctypes.c_uint32(0)
    lib.vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device, ctypes.pointer(count), None
    )
    if count.value:
        families = (VkQueueFamilyProperties * count.value)()
        lib.vkGetPhysicalDeviceQueueFamilyProperties(
            physical_device, ctypes.pointer(count), families
        )
        for index, family in enumerate(families[: count.value]):
            if family.queueFlags & VK_QUEUE_COMPUTE_BIT:
                return index
    raise VulkanError("selected Vulkan device exposes no compute queue family")


def _default_load_library(name: str) -> ctypes.CDLL:
    """Load a shared library by name through :class:`ctypes.CDLL`."""
    return ctypes.CDLL(name)


def _load_library(
    find_loader: Callable[[], str | None],
    load_library: Callable[[str], ctypes.CDLL],
) -> ctypes.CDLL:
    """Locate and load the Vulkan ICD loader shared library.

    Raises:
        VulkanError: When no loader is installed or it cannot be loaded.
    """
    loader = find_loader()
    if loader is None:
        raise VulkanError(
            "no Vulkan ICD loader found (libvulkan); install a Vulkan runtime"
        )
    try:
        return load_library(loader)
    except OSError as exc:
        raise VulkanError(f"cannot load Vulkan loader {loader!r}: {exc}") from exc


class VulkanBuffer:
    """A storage buffer backed by persistently mapped host-coherent memory.

    Created by :meth:`VulkanContext.allocate`; not constructed directly. The
    mapping lives for the lifetime of the buffer, so :meth:`write` and
    :meth:`read` are plain memory copies. Usable as a context manager, and
    :meth:`free` is idempotent.
    """

    def __init__(
        self,
        lib: ctypes.CDLL,
        device: ctypes.c_void_p,
        nbytes: int,
        buffer: ctypes.c_uint64,
        memory: ctypes.c_uint64,
        mapped: int,
        release: Callable[[VulkanBuffer], None],
    ) -> None:
        self._lib = lib
        self._device = device
        self._nbytes = nbytes
        self._buffer = buffer
        self._memory = memory
        self._mapped = mapped
        self._release = release
        self._freed = False

    @property
    def nbytes(self) -> int:
        """Size of the buffer in bytes, as requested."""
        return self._nbytes

    @property
    def freed(self) -> bool:
        """``True`` once the buffer and its memory have been released."""
        return self._freed

    @property
    def handle(self) -> int:
        """The underlying ``VkBuffer`` handle, for binding into a descriptor set.

        Raises:
            VulkanError: When the buffer has already been freed.
        """
        if self._freed:
            raise VulkanError("buffer has been freed")
        return int(self._buffer.value)

    def _live_address(self) -> int:
        """Return the mapped address, refusing to touch a freed buffer."""
        if self._freed:
            raise VulkanError("buffer has been freed")
        return self._mapped

    def write(self, data: NDArray[Any]) -> None:
        """Copy a numpy array into the buffer.

        Args:
            data: A C-contiguous array whose bytes fit in the buffer. Non
                contiguous input is copied to a contiguous temporary first.

        Raises:
            VulkanError: When the buffer was freed or the array is too large.
        """
        address = self._live_address()
        source = np.ascontiguousarray(data)
        if source.nbytes > self._nbytes:
            raise VulkanError(
                f"array of {source.nbytes} bytes does not fit in a "
                f"{self._nbytes}-byte buffer"
            )
        ctypes.memmove(address, source.ctypes.data, source.nbytes)

    def read(
        self, dtype: DTypeLike, shape: tuple[int, ...] | None = None
    ) -> NDArray[Any]:
        """Copy bytes out of the buffer into a new numpy array.

        Args:
            dtype: Element type to interpret the bytes as.
            shape: Shape of the result. ``None`` reads the whole buffer as a
                1-D array, which requires the size to divide evenly.

        Returns:
            A new array owning its memory — the buffer can be freed after.

        Raises:
            VulkanError: When the buffer was freed, ``dtype`` has no fixed
                element size, ``shape`` has a negative dimension, or the
                requested view does not fit the allocation.
        """
        address = self._live_address()
        element = np.dtype(dtype)
        if element.itemsize <= 0:
            # Flexible dtypes ('U', 'S0', 'V0') have no fixed element size, so
            # neither the implicit length nor the byte budget below means anything.
            raise VulkanError(f"dtype {element.str!r} has no fixed element size")
        if shape is not None and any(dim < 0 for dim in shape):
            raise VulkanError(f"shape {shape} has a negative dimension")
        if shape is None:
            if self._nbytes % element.itemsize:
                raise VulkanError(
                    f"{self._nbytes} bytes is not a whole number of "
                    f"{element.name} elements"
                )
            shape = (self._nbytes // element.itemsize,)
        wanted = element.itemsize * int(np.prod(shape, dtype=np.int64))
        if wanted > self._nbytes:
            raise VulkanError(
                f"reading {wanted} bytes from a {self._nbytes}-byte buffer"
            )
        raw = (ctypes.c_ubyte * wanted).from_address(address)
        return np.frombuffer(bytes(raw), dtype=element).reshape(shape)

    def free(self) -> None:
        """Unmap and destroy the buffer and its memory. Safe to call twice."""
        if self._freed:
            return
        self._freed = True
        self._lib.vkUnmapMemory(self._device, self._memory)
        self._lib.vkFreeMemory(self._device, self._memory, None)
        self._lib.vkDestroyBuffer(self._device, self._buffer, None)
        self._release(self)

    def __enter__(self) -> VulkanBuffer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.free()


class _PipelineHandles(NamedTuple):
    """Every Vulkan object a dispatchable compute pipeline is made of."""

    shader_module: ctypes.c_uint64
    descriptor_set_layout: ctypes.c_uint64
    descriptor_pool: ctypes.c_uint64
    descriptor_set: ctypes.c_uint64
    pipeline_layout: ctypes.c_uint64
    pipeline: ctypes.c_uint64
    command_pool: ctypes.c_uint64
    command_buffer: ctypes.c_void_p
    fence: ctypes.c_uint64


class VulkanPipeline:
    """A compiled SPIR-V compute shader that can be dispatched over buffers.

    Created by :meth:`VulkanContext.compute_pipeline`; not constructed directly.
    The shader's storage buffers all live in descriptor set 0, bound in the
    order they are passed to :meth:`dispatch`: ``buffers[i]`` is ``binding = i``.
    One command buffer and one fence are created up front and reused, so a
    repeated dispatch records and submits without allocating.

    Values the shader needs at compile time (workgroup size, tile shape, the
    length of a shared array) are fixed here as specialization constants; see
    :attr:`specialization`.

    Dispatch is synchronous: it waits on the fence and inserts a barrier making
    shader writes visible to the host, so a buffer can be read straight after.
    Usable as a context manager, and :meth:`destroy` is idempotent.
    """

    def __init__(
        self,
        lib: ctypes.CDLL,
        device: ctypes.c_void_p,
        queue: ctypes.c_void_p,
        handles: _PipelineHandles,
        *,
        buffer_count: int,
        push_constant_bytes: int,
        release: Callable[[VulkanPipeline], None],
        specialization: Mapping[int, SpecializationValue] | None = None,
    ) -> None:
        self._lib = lib
        self._device = device
        self._queue = queue
        self._handles = handles
        self._buffer_count = buffer_count
        self._push_constant_bytes = push_constant_bytes
        self._specialization = dict(specialization or {})
        self._release = release
        self._destroyed = False
        self._pending = False
        self._dispatch_count = 0

    @property
    def buffer_count(self) -> int:
        """Number of storage buffers the shader expects, one per binding."""
        return self._buffer_count

    @property
    def push_constant_bytes(self) -> int:
        """Size of the push-constant block, in bytes (0 when the shader has none)."""
        return self._push_constant_bytes

    @property
    def specialization(self) -> dict[int, SpecializationValue]:
        """The specialization constants baked into this pipeline, by id.

        A copy: the values were consumed when the driver compiled the module
        and cannot be changed afterwards.
        """
        return dict(self._specialization)

    @property
    def dispatch_count(self) -> int:
        """How many dispatches have completed through this pipeline."""
        return self._dispatch_count

    @property
    def destroyed(self) -> bool:
        """``True`` once the pipeline's Vulkan objects have been released."""
        return self._destroyed

    def dispatch(
        self,
        buffers: Sequence[VulkanBuffer],
        *,
        groups: int | Sequence[int],
        push_constants: bytes | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        """Run the shader once and wait for it to finish.

        Args:
            buffers: One live buffer per binding, in binding order. Exactly
                :attr:`buffer_count` of them.
            groups: Workgroup counts, as an int or a 1-to-3 element sequence;
                missing axes default to 1. This is the number of *workgroups*,
                not invocations — divide by the shader's ``local_size``.
            push_constants: Exactly :attr:`push_constant_bytes` of packed data,
                or ``None`` when the shader declares no push constants.
            timeout_s: How long to wait for the queue before giving up. A
                timeout means the device is wedged, so it raises rather than
                returning quietly with a half-written buffer.

        Raises:
            VulkanError: When the pipeline was destroyed, an earlier dispatch
                never completed, the arguments do not match what the pipeline
                was built for, a buffer has been freed, or a Vulkan entry point
                fails or times out.
        """
        if self._destroyed:
            raise VulkanError("pipeline has been destroyed")
        if self._pending:
            raise VulkanError(
                "an earlier dispatch never completed; this pipeline cannot be reused"
            )
        if timeout_s <= 0:
            raise VulkanError(f"timeout_s must be positive, got {timeout_s!r}")
        bound = tuple(buffers)
        if len(bound) != self._buffer_count:
            raise VulkanError(
                f"pipeline binds {self._buffer_count} buffer(s), got {len(bound)}"
            )
        payload = self._push_constant_payload(push_constants)
        group_counts = _require_groups(groups)

        self._bind(bound)
        self._record(group_counts, payload)
        self._submit(timeout_s)
        self._dispatch_count += 1

    def _push_constant_payload(self, push_constants: bytes | None) -> bytes:
        """Validate the push-constant block against the pipeline's declared size."""
        payload = b"" if push_constants is None else bytes(push_constants)
        if len(payload) != self._push_constant_bytes:
            raise VulkanError(
                f"pipeline declares {self._push_constant_bytes} push-constant "
                f"byte(s), got {len(payload)}"
            )
        return payload

    def _bind(self, buffers: Sequence[VulkanBuffer]) -> None:
        """Point the descriptor set at ``buffers``, one buffer per binding."""
        infos = (_VkDescriptorBufferInfo * len(buffers))()
        writes = (_VkWriteDescriptorSet * len(buffers))()
        for index, buffer in enumerate(buffers):
            infos[index] = _VkDescriptorBufferInfo(
                buffer=buffer.handle, offset=0, range=_VK_WHOLE_SIZE
            )
            writes[index] = _VkWriteDescriptorSet(
                sType=_VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self._handles.descriptor_set,
                dstBinding=index,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=_VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=ctypes.cast(
                    ctypes.byref(infos, ctypes.sizeof(_VkDescriptorBufferInfo) * index),
                    ctypes.c_void_p,
                ),
            )
        self._lib.vkUpdateDescriptorSets(
            self._device,
            ctypes.c_uint32(len(buffers)),
            writes,
            ctypes.c_uint32(0),
            None,
        )

    def _record(self, groups: tuple[int, int, int], push_constants: bytes) -> None:
        """Re-record the command buffer for one dispatch of ``groups``."""
        lib = self._lib
        command_buffer = self._handles.command_buffer
        _check(
            lib.vkResetCommandBuffer(command_buffer, ctypes.c_uint32(0)),
            "vkResetCommandBuffer",
        )
        begin = _VkCommandBufferBeginInfo(
            sType=_VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        )
        _check(
            lib.vkBeginCommandBuffer(command_buffer, ctypes.pointer(begin)),
            "vkBeginCommandBuffer",
        )
        lib.vkCmdBindPipeline(
            command_buffer,
            ctypes.c_uint32(_VK_PIPELINE_BIND_POINT_COMPUTE),
            self._handles.pipeline,
        )
        descriptor_sets = (ctypes.c_uint64 * 1)(self._handles.descriptor_set)
        lib.vkCmdBindDescriptorSets(
            command_buffer,
            ctypes.c_uint32(_VK_PIPELINE_BIND_POINT_COMPUTE),
            self._handles.pipeline_layout,
            ctypes.c_uint32(0),
            ctypes.c_uint32(1),
            descriptor_sets,
            ctypes.c_uint32(0),
            None,
        )
        if push_constants:
            lib.vkCmdPushConstants(
                command_buffer,
                self._handles.pipeline_layout,
                ctypes.c_uint32(_VK_SHADER_STAGE_COMPUTE_BIT),
                ctypes.c_uint32(0),
                ctypes.c_uint32(len(push_constants)),
                ctypes.create_string_buffer(push_constants, len(push_constants)),
            )
        lib.vkCmdDispatch(command_buffer, *(ctypes.c_uint32(count) for count in groups))
        # Shader writes are not host-visible just because the memory is
        # coherent: the dependency has to be spelled out before the host reads.
        barrier = _VkMemoryBarrier(
            sType=_VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            srcAccessMask=_VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=_VK_ACCESS_HOST_READ_BIT,
        )
        lib.vkCmdPipelineBarrier(
            command_buffer,
            ctypes.c_uint32(_VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
            ctypes.c_uint32(_VK_PIPELINE_STAGE_HOST_BIT),
            ctypes.c_uint32(0),
            ctypes.c_uint32(1),
            ctypes.pointer(barrier),
            ctypes.c_uint32(0),
            None,
            ctypes.c_uint32(0),
            None,
        )
        _check(lib.vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer")

    def _submit(self, timeout_s: float) -> None:
        """Submit the recorded command buffer and block on its fence."""
        lib = self._lib
        fences = (ctypes.c_uint64 * 1)(self._handles.fence)
        _check(
            lib.vkResetFences(self._device, ctypes.c_uint32(1), fences),
            "vkResetFences",
        )
        command_buffers = (ctypes.c_void_p * 1)(self._handles.command_buffer)
        submit = _VkSubmitInfo(
            sType=_VK_STRUCTURE_TYPE_SUBMIT_INFO,
            commandBufferCount=1,
            pCommandBuffers=ctypes.cast(command_buffers, ctypes.c_void_p),
        )
        _check(
            lib.vkQueueSubmit(
                self._queue,
                ctypes.c_uint32(1),
                ctypes.pointer(submit),
                self._handles.fence,
            ),
            "vkQueueSubmit",
        )
        status = lib.vkWaitForFences(
            self._device,
            ctypes.c_uint32(1),
            fences,
            ctypes.c_uint32(_VK_TRUE),
            ctypes.c_uint64(int(timeout_s * 1e9)),
        )
        if status != VK_SUCCESS:
            # The submission is still in flight, so the command buffer and every
            # object it references must not be reset or destroyed from here.
            self._pending = True
        _check(status, f"vkWaitForFences (after {timeout_s}s)")

    def destroy(self) -> None:
        """Release every Vulkan object of the pipeline. Safe to call twice.

        After a dispatch that never completed, the device is drained first: an
        object referenced by a pending submission cannot legally be destroyed.
        """
        if self._destroyed:
            return
        self._destroyed = True
        lib, device, handles = self._lib, self._device, self._handles
        if self._pending:
            lib.vkDeviceWaitIdle(device)
            self._pending = False
        lib.vkDestroyFence(device, handles.fence, None)
        # Destroying the pools frees the command buffer and descriptor set.
        lib.vkDestroyCommandPool(device, handles.command_pool, None)
        lib.vkDestroyPipeline(device, handles.pipeline, None)
        lib.vkDestroyPipelineLayout(device, handles.pipeline_layout, None)
        lib.vkDestroyDescriptorPool(device, handles.descriptor_pool, None)
        lib.vkDestroyDescriptorSetLayout(device, handles.descriptor_set_layout, None)
        lib.vkDestroyShaderModule(device, handles.shader_module, None)
        self._release(self)

    def __enter__(self) -> VulkanPipeline:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.destroy()


class VulkanContext:
    """An open Vulkan device with one compute queue.

    Holds the instance, the logical device and the queue for as long as the
    caller needs them, and owns every buffer allocated through it. Use
    :meth:`open` to construct one; use it as a context manager (or call
    :meth:`close`) so the handles are destroyed in the right order.
    """

    def __init__(
        self,
        lib: ctypes.CDLL,
        instance: ctypes.c_void_p,
        physical_device: ctypes.c_void_p,
        device: ctypes.c_void_p,
        queue: ctypes.c_void_p,
        *,
        device_name: str,
        api_version: str,
        queue_family_index: int,
    ) -> None:
        self._lib = lib
        self._instance = instance
        self._physical_device = physical_device
        self._device = device
        self._queue = queue
        self._device_name = device_name
        self._api_version = api_version
        self._queue_family_index = queue_family_index
        self._buffers: list[VulkanBuffer] = []
        self._pipelines: list[VulkanPipeline] = []
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        device_index: int | None = None,
        find_loader: Callable[[], str | None] = find_loader_name,
        load_library: Callable[[str], ctypes.CDLL] = _default_load_library,
    ) -> VulkanContext:
        """Open a compute-capable Vulkan device.

        Args:
            device_index: Index into the loader's physical device enumeration.
                ``None`` (the default) picks the first device that exposes a
                compute queue family.
            find_loader: Callable returning the ICD loader library name, or
                ``None`` when no Vulkan runtime is installed.
            load_library: Callable loading that library; injectable so the
                sequence can be exercised without a Vulkan runtime.

        Returns:
            An open context whose device and queue are ready to use.

        Raises:
            VulkanError: When there is no loader, no device, no compute queue,
                or a Vulkan entry point fails.
        """
        lib = _load_library(find_loader, load_library)
        instance = _create_instance(lib)
        try:
            physical_device, properties = _select_physical_device(
                lib, instance, device_index
            )
            queue_family_index = _compute_queue_family(lib, physical_device)
            device, queue = _create_device(lib, physical_device, queue_family_index)
        except BaseException:
            lib.vkDestroyInstance(instance, None)
            raise
        return cls(
            lib,
            instance,
            physical_device,
            device,
            queue,
            device_name=properties.deviceName.decode("utf-8", "replace"),
            api_version=format_api_version(properties.apiVersion),
            queue_family_index=queue_family_index,
        )

    @property
    def device_name(self) -> str:
        """Name of the opened device, e.g. ``"V3D 7.1.7.0"``."""
        return self._device_name

    @property
    def api_version(self) -> str:
        """Vulkan version the opened device supports, as ``"major.minor.patch"``."""
        return self._api_version

    @property
    def queue_family_index(self) -> int:
        """Index of the queue family the compute queue was taken from."""
        return self._queue_family_index

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has run."""
        return self._closed

    @property
    def live_buffers(self) -> int:
        """Number of buffers allocated through this context and not yet freed."""
        return len(self._buffers)

    @property
    def live_pipelines(self) -> int:
        """Number of pipelines built through this context and not yet destroyed."""
        return len(self._pipelines)

    def compute_pipeline(
        self,
        spirv: bytes,
        *,
        buffer_count: int,
        push_constant_bytes: int = 0,
        entry_point: str = "main",
        specialization: Mapping[int, SpecializationValue] | None = None,
    ) -> VulkanPipeline:
        """Compile a SPIR-V compute shader into a dispatchable pipeline.

        Args:
            spirv: The compiled module, as produced by e.g. ``glslangValidator
                -V``. Bytes-like, a whole number of 32-bit words, starting with
                the SPIR-V magic number.
            buffer_count: How many storage buffers the shader reads or writes.
                They occupy bindings ``0..buffer_count - 1`` of descriptor set 0.
            push_constant_bytes: Size of the shader's push-constant block; 0
                when it has none. Must be a multiple of 4 and no larger than the
                128 bytes Vulkan guarantees everywhere.
            entry_point: Name of the shader's entry point.
            specialization: Values for the shader's ``layout(constant_id = N)``
                constants, keyed by id. Each must be a ``bool``, ``int`` or
                ``float`` and is baked in when the driver compiles the module,
                so one SPIR-V file can be built for several tile shapes or
                workgroup sizes. Ids the shader does not declare are ignored by
                the driver; ids it declares and this omits keep their default.

        Returns:
            A :class:`VulkanPipeline` owned by this context.

        Raises:
            VulkanError: When the context is closed, an argument is invalid, or
                a Vulkan entry point fails. Objects created before a failure are
                destroyed, so a rejected pipeline leaks nothing.
        """
        if self._closed:
            raise VulkanError("context is closed")
        code = _require_spirv(spirv)
        bindings = _require_buffer_count(buffer_count)
        push_bytes = _require_push_constant_bytes(push_constant_bytes)
        constants = _require_specialization(specialization)

        handles = _build_pipeline(
            self._lib,
            self._device,
            queue_family_index=self._queue_family_index,
            code=code,
            buffer_count=bindings,
            push_constant_bytes=push_bytes,
            entry_point=entry_point,
            specialization=constants,
        )
        pipeline = VulkanPipeline(
            self._lib,
            self._device,
            self._queue,
            handles,
            buffer_count=bindings,
            push_constant_bytes=push_bytes,
            specialization=constants,
            release=self._forget_pipeline,
        )
        self._pipelines.append(pipeline)
        return pipeline

    def allocate(self, nbytes: int) -> VulkanBuffer:
        """Allocate a host-visible storage buffer of ``nbytes`` bytes.

        Args:
            nbytes: Size of the buffer; must be a positive integer. The driver
                may back it with a larger allocation, but only ``nbytes`` are
                addressable through the returned object.

        Returns:
            A mapped :class:`VulkanBuffer` owned by this context.

        Raises:
            VulkanError: When the context is closed, the size is invalid, no
                suitable memory type exists, or an entry point fails.
        """
        if self._closed:
            raise VulkanError("context is closed")
        _require_positive_int(nbytes)

        info = _VkBufferCreateInfo(
            sType=_VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=nbytes,
            usage=_BUFFER_USAGE,
            sharingMode=_VK_SHARING_MODE_EXCLUSIVE,
        )
        buffer = ctypes.c_uint64(0)
        _check(
            self._lib.vkCreateBuffer(
                self._device, ctypes.pointer(info), None, ctypes.pointer(buffer)
            ),
            "vkCreateBuffer",
        )
        memory = ctypes.c_uint64(0)
        try:
            requirements = _VkMemoryRequirements()
            self._lib.vkGetBufferMemoryRequirements(
                self._device, buffer, ctypes.pointer(requirements)
            )
            allocate = _VkMemoryAllocateInfo(
                sType=_VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=max(requirements.size, nbytes),
                memoryTypeIndex=_select_memory_type(
                    self._memory_property_flags(), requirements.memoryTypeBits
                ),
            )
            _check(
                self._lib.vkAllocateMemory(
                    self._device,
                    ctypes.pointer(allocate),
                    None,
                    ctypes.pointer(memory),
                ),
                "vkAllocateMemory",
            )
            _check(
                self._lib.vkBindBufferMemory(
                    self._device, buffer, memory, ctypes.c_uint64(0)
                ),
                "vkBindBufferMemory",
            )
            mapped = ctypes.c_void_p(0)
            _check(
                self._lib.vkMapMemory(
                    self._device,
                    memory,
                    ctypes.c_uint64(0),
                    ctypes.c_uint64(nbytes),
                    ctypes.c_uint32(0),
                    ctypes.pointer(mapped),
                ),
                "vkMapMemory",
            )
            if not mapped.value:
                raise VulkanError("vkMapMemory returned a null pointer")
        except BaseException:
            if memory.value:
                self._lib.vkFreeMemory(self._device, memory, None)
            self._lib.vkDestroyBuffer(self._device, buffer, None)
            raise

        allocated = VulkanBuffer(
            self._lib,
            self._device,
            nbytes,
            buffer,
            memory,
            mapped.value,
            self._forget_buffer,
        )
        self._buffers.append(allocated)
        return allocated

    def _memory_property_flags(self) -> tuple[int, ...]:
        """Report ``propertyFlags`` of every memory type of the open device."""
        properties = _VkPhysicalDeviceMemoryProperties()
        self._lib.vkGetPhysicalDeviceMemoryProperties(
            self._physical_device, ctypes.pointer(properties)
        )
        count = min(properties.memoryTypeCount, _VK_MAX_MEMORY_TYPES)
        return tuple(
            int(properties.memoryTypes[index].propertyFlags) for index in range(count)
        )

    def _forget_buffer(self, buffer: VulkanBuffer) -> None:
        """Drop a freed buffer from the live set (called by its ``free``)."""
        self._buffers.remove(buffer)

    def _forget_pipeline(self, pipeline: VulkanPipeline) -> None:
        """Drop a destroyed pipeline from the live set (called by ``destroy``)."""
        self._pipelines.remove(pipeline)

    def close(self) -> None:
        """Destroy outstanding pipelines and buffers, then the device and instance.

        Idempotent. Objects created through this context are released first, so
        forgetting one is a leak the context still cleans up.
        """
        if self._closed:
            return
        for pipeline in list(self._pipelines):
            pipeline.destroy()
        for buffer in list(self._buffers):
            buffer.free()
        self._closed = True
        self._lib.vkDestroyDevice(self._device, None)
        self._lib.vkDestroyInstance(self._instance, None)

    def __enter__(self) -> VulkanContext:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _build_pipeline(
    lib: ctypes.CDLL,
    device: ctypes.c_void_p,
    *,
    queue_family_index: int,
    code: bytes,
    buffer_count: int,
    push_constant_bytes: int,
    entry_point: str,
    specialization: Mapping[int, SpecializationValue],
) -> _PipelineHandles:
    """Create every object a compute dispatch needs, rolling back on failure.

    The nine handles are created in dependency order; each one registers how to
    undo itself, so a failure anywhere destroys exactly what already exists and
    nothing else.
    """
    undo: list[Callable[[], None]] = []
    try:
        shader_module = _create_shader_module(lib, device, code)
        undo.append(lambda: lib.vkDestroyShaderModule(device, shader_module, None))

        set_layout = _create_descriptor_set_layout(lib, device, buffer_count)
        undo.append(lambda: lib.vkDestroyDescriptorSetLayout(device, set_layout, None))

        pool = _create_descriptor_pool(lib, device, buffer_count)
        undo.append(lambda: lib.vkDestroyDescriptorPool(device, pool, None))

        descriptor_set = _allocate_descriptor_set(lib, device, pool, set_layout)

        pipeline_layout = _create_pipeline_layout(
            lib, device, set_layout, push_constant_bytes
        )
        undo.append(lambda: lib.vkDestroyPipelineLayout(device, pipeline_layout, None))

        pipeline = _create_pipeline(
            lib,
            device,
            shader_module,
            pipeline_layout,
            entry_point,
            specialization,
        )
        undo.append(lambda: lib.vkDestroyPipeline(device, pipeline, None))

        command_pool = _create_command_pool(lib, device, queue_family_index)
        undo.append(lambda: lib.vkDestroyCommandPool(device, command_pool, None))

        command_buffer = _allocate_command_buffer(lib, device, command_pool)
        fence = _create_fence(lib, device)
    except BaseException:
        for step in reversed(undo):
            step()
        raise
    return _PipelineHandles(
        shader_module=shader_module,
        descriptor_set_layout=set_layout,
        descriptor_pool=pool,
        descriptor_set=descriptor_set,
        pipeline_layout=pipeline_layout,
        pipeline=pipeline,
        command_pool=command_pool,
        command_buffer=command_buffer,
        fence=fence,
    )


def _create_shader_module(
    lib: ctypes.CDLL, device: ctypes.c_void_p, code: bytes
) -> ctypes.c_uint64:
    """Hand a validated SPIR-V module to the driver."""
    words = (ctypes.c_uint32 * (len(code) // 4)).from_buffer_copy(code)
    info = _VkShaderModuleCreateInfo(
        sType=_VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        codeSize=len(code),
        pCode=ctypes.cast(words, ctypes.c_void_p),
    )
    module = ctypes.c_uint64(0)
    _check(
        lib.vkCreateShaderModule(
            device, ctypes.pointer(info), None, ctypes.pointer(module)
        ),
        "vkCreateShaderModule",
    )
    return module


def _create_descriptor_set_layout(
    lib: ctypes.CDLL, device: ctypes.c_void_p, buffer_count: int
) -> ctypes.c_uint64:
    """Describe ``buffer_count`` storage buffers at bindings 0..n-1 of set 0."""
    bindings = (_VkDescriptorSetLayoutBinding * buffer_count)(
        *(
            _VkDescriptorSetLayoutBinding(
                binding=index,
                descriptorType=_VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=_VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for index in range(buffer_count)
        )
    )
    info = _VkDescriptorSetLayoutCreateInfo(
        sType=_VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        bindingCount=buffer_count,
        pBindings=ctypes.cast(bindings, ctypes.c_void_p),
    )
    layout = ctypes.c_uint64(0)
    _check(
        lib.vkCreateDescriptorSetLayout(
            device, ctypes.pointer(info), None, ctypes.pointer(layout)
        ),
        "vkCreateDescriptorSetLayout",
    )
    return layout


def _create_descriptor_pool(
    lib: ctypes.CDLL, device: ctypes.c_void_p, buffer_count: int
) -> ctypes.c_uint64:
    """Reserve one descriptor set holding ``buffer_count`` storage buffers."""
    sizes = (_VkDescriptorPoolSize * 1)(
        _VkDescriptorPoolSize(
            type=_VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=buffer_count
        )
    )
    info = _VkDescriptorPoolCreateInfo(
        sType=_VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        maxSets=1,
        poolSizeCount=1,
        pPoolSizes=ctypes.cast(sizes, ctypes.c_void_p),
    )
    pool = ctypes.c_uint64(0)
    _check(
        lib.vkCreateDescriptorPool(
            device, ctypes.pointer(info), None, ctypes.pointer(pool)
        ),
        "vkCreateDescriptorPool",
    )
    return pool


def _allocate_descriptor_set(
    lib: ctypes.CDLL,
    device: ctypes.c_void_p,
    pool: ctypes.c_uint64,
    set_layout: ctypes.c_uint64,
) -> ctypes.c_uint64:
    """Allocate the single descriptor set; the pool owns it until destroyed."""
    layouts = (ctypes.c_uint64 * 1)(set_layout)
    info = _VkDescriptorSetAllocateInfo(
        sType=_VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        descriptorPool=pool,
        descriptorSetCount=1,
        pSetLayouts=ctypes.cast(layouts, ctypes.c_void_p),
    )
    sets = (ctypes.c_uint64 * 1)(0)
    _check(
        lib.vkAllocateDescriptorSets(device, ctypes.pointer(info), sets),
        "vkAllocateDescriptorSets",
    )
    return ctypes.c_uint64(sets[0])


def _create_pipeline_layout(
    lib: ctypes.CDLL,
    device: ctypes.c_void_p,
    set_layout: ctypes.c_uint64,
    push_constant_bytes: int,
) -> ctypes.c_uint64:
    """Combine the descriptor set layout with the push-constant range."""
    layouts = (ctypes.c_uint64 * 1)(set_layout)
    ranges = (_VkPushConstantRange * 1)(
        _VkPushConstantRange(
            stageFlags=_VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=push_constant_bytes,
        )
    )
    info = _VkPipelineLayoutCreateInfo(
        sType=_VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        setLayoutCount=1,
        pSetLayouts=ctypes.cast(layouts, ctypes.c_void_p),
        pushConstantRangeCount=1 if push_constant_bytes else 0,
        pPushConstantRanges=(
            ctypes.cast(ranges, ctypes.c_void_p) if push_constant_bytes else None
        ),
    )
    layout = ctypes.c_uint64(0)
    _check(
        lib.vkCreatePipelineLayout(
            device, ctypes.pointer(info), None, ctypes.pointer(layout)
        ),
        "vkCreatePipelineLayout",
    )
    return layout


def _specialization_arrays(
    specialization: Mapping[int, SpecializationValue],
) -> tuple[ctypes.Array[_VkSpecializationMapEntry], ctypes.Array[ctypes.c_char]]:
    """Lay the constants out as a map-entry table plus one packed data blob.

    Every value takes a 4-byte slot, so entry ``i`` sits at offset ``4 * i`` and
    the blob is as long as the table. Ids are already ordered by
    :func:`_require_specialization`, which keeps the layout reproducible.
    """
    count = len(specialization)
    entries = (_VkSpecializationMapEntry * count)()
    blob = b""
    for index, (constant_id, value) in enumerate(specialization.items()):
        entries[index] = _VkSpecializationMapEntry(
            constantID=constant_id,
            offset=index * _SPECIALIZATION_VALUE_BYTES,
            size=_SPECIALIZATION_VALUE_BYTES,
        )
        blob += _pack_specialization_value(value)
    return entries, (ctypes.c_char * len(blob)).from_buffer_copy(blob)


def _create_pipeline(
    lib: ctypes.CDLL,
    device: ctypes.c_void_p,
    shader_module: ctypes.c_uint64,
    pipeline_layout: ctypes.c_uint64,
    entry_point: str,
    specialization: Mapping[int, SpecializationValue],
) -> ctypes.c_uint64:
    """Compile the shader for the device against ``pipeline_layout``.

    Specialization constants are resolved here, at pipeline build time: the
    driver compiles the module with those values baked in, which is what lets a
    shader size a shared array or its workgroup from them.
    """
    name = entry_point.encode("utf-8")
    # entries/blob/spec must outlive the create call: the driver reads them
    # through the pointers below while it compiles.
    entries, blob = _specialization_arrays(specialization)
    spec_pointer: ctypes.c_void_p | None = None
    if specialization:
        spec = _VkSpecializationInfo(
            mapEntryCount=len(specialization),
            pMapEntries=ctypes.cast(entries, ctypes.c_void_p),
            dataSize=len(blob),
            pData=ctypes.cast(blob, ctypes.c_void_p),
        )
        spec_pointer = ctypes.cast(ctypes.byref(spec), ctypes.c_void_p)
    info = _VkComputePipelineCreateInfo(
        sType=_VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        stage=_VkPipelineShaderStageCreateInfo(
            sType=_VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=_VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader_module,
            pName=name,
            pSpecializationInfo=spec_pointer,
        ),
        layout=pipeline_layout,
    )
    pipeline = (ctypes.c_uint64 * 1)(0)
    _check(
        lib.vkCreateComputePipelines(
            device,
            ctypes.c_uint64(0),
            ctypes.c_uint32(1),
            ctypes.pointer(info),
            None,
            pipeline,
        ),
        "vkCreateComputePipelines",
    )
    return ctypes.c_uint64(pipeline[0])


def _create_command_pool(
    lib: ctypes.CDLL, device: ctypes.c_void_p, queue_family_index: int
) -> ctypes.c_uint64:
    """Create a command pool whose buffers can be reset and re-recorded."""
    info = _VkCommandPoolCreateInfo(
        sType=_VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        flags=_VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        queueFamilyIndex=queue_family_index,
    )
    pool = ctypes.c_uint64(0)
    _check(
        lib.vkCreateCommandPool(
            device, ctypes.pointer(info), None, ctypes.pointer(pool)
        ),
        "vkCreateCommandPool",
    )
    return pool


def _allocate_command_buffer(
    lib: ctypes.CDLL, device: ctypes.c_void_p, command_pool: ctypes.c_uint64
) -> ctypes.c_void_p:
    """Allocate the one primary command buffer every dispatch re-records."""
    info = _VkCommandBufferAllocateInfo(
        sType=_VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        commandPool=command_pool,
        level=_VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        commandBufferCount=1,
    )
    buffers = (ctypes.c_void_p * 1)(None)
    _check(
        lib.vkAllocateCommandBuffers(device, ctypes.pointer(info), buffers),
        "vkAllocateCommandBuffers",
    )
    return ctypes.c_void_p(buffers[0])


def _create_fence(lib: ctypes.CDLL, device: ctypes.c_void_p) -> ctypes.c_uint64:
    """Create the unsignalled fence each dispatch waits on."""
    info = _VkFenceCreateInfo(sType=_VK_STRUCTURE_TYPE_FENCE_CREATE_INFO)
    fence = ctypes.c_uint64(0)
    _check(
        lib.vkCreateFence(device, ctypes.pointer(info), None, ctypes.pointer(fence)),
        "vkCreateFence",
    )
    return fence


def _create_instance(lib: ctypes.CDLL) -> ctypes.c_void_p:
    """Create a Vulkan instance with no layers or extensions."""
    info = VkInstanceCreateInfo(sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
    instance = ctypes.c_void_p(0)
    _check(
        lib.vkCreateInstance(ctypes.pointer(info), None, ctypes.pointer(instance)),
        "vkCreateInstance",
    )
    if not instance.value:
        raise VulkanError("vkCreateInstance returned a null instance")
    return instance


def _select_physical_device(
    lib: ctypes.CDLL, instance: ctypes.c_void_p, device_index: int | None
) -> tuple[ctypes.c_void_p, VkPhysicalDeviceProperties]:
    """Choose a physical device and read its properties.

    ``device_index`` selects by enumeration order; ``None`` takes the first
    device with a compute queue family.
    """
    count = ctypes.c_uint32(0)
    _check(
        lib.vkEnumeratePhysicalDevices(instance, ctypes.pointer(count), None),
        "vkEnumeratePhysicalDevices",
    )
    if count.value == 0:
        raise VulkanError("Vulkan loader enumerated no physical devices")
    handles = (ctypes.c_void_p * count.value)()
    _check(
        lib.vkEnumeratePhysicalDevices(instance, ctypes.pointer(count), handles),
        "vkEnumeratePhysicalDevices",
    )
    devices = [ctypes.c_void_p(handle) for handle in handles[: count.value]]

    if device_index is not None:
        if not 0 <= device_index < len(devices):
            raise VulkanError(
                f"device_index {device_index} out of range: "
                f"{len(devices)} device(s) enumerated"
            )
        chosen = devices[device_index]
        return chosen, _device_properties(lib, chosen)

    for device in devices:
        properties = _device_properties(lib, device)
        if _has_compute_family(lib, device):
            return device, properties
    raise VulkanError("no enumerated Vulkan device exposes a compute queue family")


def _device_properties(
    lib: ctypes.CDLL, device: ctypes.c_void_p
) -> VkPhysicalDeviceProperties:
    """Read ``VkPhysicalDeviceProperties`` for one device."""
    properties = VkPhysicalDeviceProperties()
    lib.vkGetPhysicalDeviceProperties(device, ctypes.pointer(properties))
    return properties


def _has_compute_family(lib: ctypes.CDLL, device: ctypes.c_void_p) -> bool:
    """Return ``True`` when ``device`` has any compute-capable queue family."""
    try:
        _compute_queue_family(lib, device)
    except VulkanError:
        return False
    return True


def _create_device(
    lib: ctypes.CDLL, physical_device: ctypes.c_void_p, queue_family_index: int
) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
    """Create a logical device with one compute queue and fetch that queue."""
    priority = (ctypes.c_float * 1)(1.0)
    queue_info = _VkDeviceQueueCreateInfo(
        sType=_VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        queueFamilyIndex=queue_family_index,
        queueCount=1,
        pQueuePriorities=priority,
    )
    device_info = _VkDeviceCreateInfo(
        sType=_VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        queueCreateInfoCount=1,
        pQueueCreateInfos=ctypes.cast(ctypes.pointer(queue_info), ctypes.c_void_p),
    )
    device = ctypes.c_void_p(0)
    _check(
        lib.vkCreateDevice(
            physical_device, ctypes.pointer(device_info), None, ctypes.pointer(device)
        ),
        "vkCreateDevice",
    )
    queue = ctypes.c_void_p(0)
    lib.vkGetDeviceQueue(
        device,
        ctypes.c_uint32(queue_family_index),
        ctypes.c_uint32(0),
        ctypes.pointer(queue),
    )
    return device, queue
