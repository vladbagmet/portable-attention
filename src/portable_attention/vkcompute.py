"""Vulkan compute context: logical device, compute queue and host-visible buffers.

:mod:`portable_attention.vulkan` answers *is there a Vulkan device here?*. This
module takes the next step the M2 backend needs: it **opens** one of those
devices and moves bytes through it. It creates a logical device with a single
compute queue, allocates storage buffers backed by host-visible coherent
memory, and keeps that memory mapped so numpy arrays can be copied in and out
without a staging buffer or a command submission.

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

Failures raise :class:`VulkanError` rather than returning a sentinel: by the
time a caller opens a device, detection has already said one is there, so a
failure here is exceptional. Resources are released in reverse order by
:meth:`VulkanContext.close`, which also frees buffers the caller forgot.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Any

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
]

_VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2
_VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3
_VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5
_VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12

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


def _require_positive_int(value: object) -> None:
    """Reject a buffer size that is not a positive integer.

    Sizes are byte counts handed straight to the driver, so a float would price
    a fractional allocation and a ``bool`` would slip through a bare
    ``isinstance(..., int)`` as 1 byte.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VulkanError(f"buffer size must be a positive integer, got {value!r}")


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

    def close(self) -> None:
        """Free outstanding buffers and destroy the device and instance.

        Idempotent. Buffers allocated through this context are freed first, so
        forgetting one is a leak the context still cleans up.
        """
        if self._closed:
            return
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
