"""Tests for the Vulkan compute context (device, queue and mapped buffers)."""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
import pytest

import portable_attention as pa
from portable_attention import vkcompute as vc
from portable_attention.vulkan import vulkan_available

_INSTANCE = 0xBEEF
_DEVICE = 0xD3D1
_QUEUE = 0x0EE0

_COMPUTE_QUEUE = 0x2
_GRAPHICS_QUEUE = 0x1
_HOST_COHERENT = 0x2 | 0x4
_DEVICE_LOCAL_COHERENT = 0x1 | 0x2 | 0x4

_ONE_COMPUTE_DEVICE = (("V3D 7.1.7.0", (1 << 22) | (2 << 12) | 289, (_COMPUTE_QUEUE,)),)


class _FakeVulkan:
    """A loader stand-in that emulates the entry points the context calls.

    Memory allocations are real ctypes buffers, so mapping hands back a usable
    address and the read/write paths are exercised for real without a GPU.
    """

    def __init__(
        self,
        devices: tuple[tuple[str, int, tuple[int, ...]], ...] = _ONE_COMPUTE_DEVICE,
        *,
        memory_types: tuple[int, ...] = (_DEVICE_LOCAL_COHERENT,),
        memory_type_bits: int = 0xFFFFFFFF,
        reported_type_count: int | None = None,
        fail: dict[str, int] | None = None,
        map_null: bool = False,
        null_instance: bool = False,
    ) -> None:
        self.devices = devices
        self.memory_types = memory_types
        self.memory_type_bits = memory_type_bits
        self.reported_type_count = reported_type_count
        self.fail = fail or {}
        self.map_null = map_null
        self.null_instance = null_instance
        self.allocations: dict[int, Any] = {}
        self.buffer_sizes: dict[int, int] = {}
        self.allocation_sizes: dict[int, int] = {}
        self.bound: list[tuple[int, int]] = []
        self.mapped: set[int] = set()
        self.events: list[str] = []
        self._next_handle = 1

    def _status(self, name: str) -> int:
        return self.fail.get(name, 0)

    def _handle(self) -> int:
        self._next_handle += 1
        return self._next_handle

    # -- instance / device ------------------------------------------------
    def vkCreateInstance(self, info: Any, allocator: Any, instance: Any) -> int:  # noqa: N802
        assert info.contents.sType == 1
        instance[0] = 0 if self.null_instance else _INSTANCE
        return self._status("vkCreateInstance")

    def vkDestroyInstance(self, instance: Any, allocator: Any) -> None:  # noqa: N802
        self.events.append("destroy-instance")

    def vkEnumeratePhysicalDevices(  # noqa: N802
        self, instance: Any, count: Any, handles: Any
    ) -> int:
        if handles is None:
            count[0] = len(self.devices)
            return self._status("vkEnumeratePhysicalDevices")
        for index in range(count[0]):
            handles[index] = index + 1
        return self._status("vkEnumeratePhysicalDevicesFill")

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

    def vkCreateDevice(  # noqa: N802
        self, physical: Any, info: Any, allocator: Any, device: Any
    ) -> int:
        assert info.contents.queueCreateInfoCount == 1
        device[0] = _DEVICE
        return self._status("vkCreateDevice")

    def vkGetDeviceQueue(  # noqa: N802
        self, device: Any, family: Any, index: Any, queue: Any
    ) -> None:
        assert index.value == 0
        queue[0] = _QUEUE

    def vkDestroyDevice(self, device: Any, allocator: Any) -> None:  # noqa: N802
        self.events.append("destroy-device")

    # -- buffers / memory -------------------------------------------------
    def vkCreateBuffer(  # noqa: N802
        self, device: Any, info: Any, allocator: Any, buffer: Any
    ) -> int:
        status = self._status("vkCreateBuffer")
        if status != 0:
            # A failed create leaves the handle undefined; nothing to destroy.
            return status
        handle = self._handle()
        self.buffer_sizes[handle] = info.contents.size
        buffer[0] = handle
        return status

    def vkGetBufferMemoryRequirements(  # noqa: N802
        self, device: Any, buffer: Any, requirements: Any
    ) -> None:
        size = self.buffer_sizes[buffer.value]
        requirements.contents.size = (size + 255) // 256 * 256
        requirements.contents.alignment = 256
        requirements.contents.memoryTypeBits = self.memory_type_bits

    def vkGetPhysicalDeviceMemoryProperties(self, device: Any, props: Any) -> None:  # noqa: N802
        count = self.reported_type_count
        props.contents.memoryTypeCount = (
            len(self.memory_types) if count is None else count
        )
        for index, flags in enumerate(self.memory_types):
            props.contents.memoryTypes[index].propertyFlags = flags

    def vkAllocateMemory(  # noqa: N802
        self, device: Any, info: Any, allocator: Any, memory: Any
    ) -> int:
        self.last_memory_type = info.contents.memoryTypeIndex
        status = self._status("vkAllocateMemory")
        if status != 0:
            return status
        handle = self._handle()
        size = info.contents.allocationSize
        self.allocations[handle] = (ctypes.c_ubyte * size)()
        self.allocation_sizes[handle] = size
        memory[0] = handle
        return status

    def vkBindBufferMemory(  # noqa: N802
        self, device: Any, buffer: Any, memory: Any, offset: Any
    ) -> int:
        assert offset.value == 0
        self.bound.append((buffer.value, memory.value))
        return self._status("vkBindBufferMemory")

    def vkMapMemory(  # noqa: N802
        self, device: Any, memory: Any, offset: Any, size: Any, flags: Any, data: Any
    ) -> int:
        if self.map_null:
            data[0] = 0
        else:
            data[0] = ctypes.addressof(self.allocations[memory.value])
            self.mapped.add(memory.value)
        return self._status("vkMapMemory")

    def vkUnmapMemory(self, device: Any, memory: Any) -> None:  # noqa: N802
        self.mapped.discard(memory.value)
        self.events.append("unmap")

    def vkFreeMemory(self, device: Any, memory: Any, allocator: Any) -> None:  # noqa: N802
        self.allocations.pop(memory.value, None)
        self.events.append("free-memory")

    def vkDestroyBuffer(self, device: Any, buffer: Any, allocator: Any) -> None:  # noqa: N802
        self.buffer_sizes.pop(buffer.value, None)
        self.events.append("destroy-buffer")


def _open(lib: _FakeVulkan, **kwargs: Any) -> vc.VulkanContext:
    """Open a context against ``lib`` instead of the system loader."""
    return vc.VulkanContext.open(
        find_loader=lambda: "libvulkan.so.fake",
        load_library=lambda name: lib,
        **kwargs,
    )


# --------------------------------------------------------------------------
# memory type selection (pure)
# --------------------------------------------------------------------------


def test_public_reexports() -> None:
    """The context, buffer and error type are top-level exports."""
    for name in ("VulkanBuffer", "VulkanContext", "VulkanError"):
        assert getattr(pa, name) is getattr(vc, name)
        assert name in pa.__all__


def test_memory_type_prefers_device_local() -> None:
    """A device-local host-coherent type wins over a merely coherent one."""
    index = vc._select_memory_type((_HOST_COHERENT, _DEVICE_LOCAL_COHERENT), 0b11)
    assert index == 1


def test_memory_type_falls_back_to_first_allowed() -> None:
    """Without the preferred bits the first allowed candidate is taken."""
    index = vc._select_memory_type((0x1, _HOST_COHERENT, _HOST_COHERENT), 0b111)
    assert index == 1


def test_memory_type_respects_type_bits() -> None:
    """Types the buffer disallows are skipped even when they look ideal."""
    index = vc._select_memory_type(
        (_DEVICE_LOCAL_COHERENT, _HOST_COHERENT, _DEVICE_LOCAL_COHERENT), 0b110
    )
    assert index == 2


def test_memory_type_requires_host_coherent() -> None:
    """Device-local-only memory cannot back a mapped buffer."""
    with pytest.raises(vc.VulkanError, match="memoryTypeBits=0x3"):
        vc._select_memory_type((0x1, 0x1 | 0x2), 0b11)


# --------------------------------------------------------------------------
# opening a context
# --------------------------------------------------------------------------


def test_open_reports_device_identity() -> None:
    """The context exposes the device it opened and the queue family it used."""
    lib = _FakeVulkan()
    with _open(lib) as ctx:
        assert ctx.device_name == "V3D 7.1.7.0"
        assert ctx.api_version == "1.2.289"
        assert ctx.queue_family_index == 0
        assert ctx.closed is False
    assert ctx.closed is True
    assert lib.events == ["destroy-device", "destroy-instance"]


def test_open_skips_devices_without_compute() -> None:
    """Auto-selection walks past graphics-only devices."""
    lib = _FakeVulkan(
        (
            ("display-only", 1 << 22, (_GRAPHICS_QUEUE,)),
            ("llvmpipe", (1 << 22) | (3 << 12), (_GRAPHICS_QUEUE, _COMPUTE_QUEUE)),
        )
    )
    with _open(lib) as ctx:
        assert ctx.device_name == "llvmpipe"
        assert ctx.queue_family_index == 1


def test_open_by_index_selects_that_device() -> None:
    """``device_index`` picks by enumeration order, not by capability."""
    lib = _FakeVulkan(
        (
            ("first", 1 << 22, (_COMPUTE_QUEUE,)),
            ("second", 1 << 22, (_COMPUTE_QUEUE,)),
        )
    )
    with _open(lib, device_index=1) as ctx:
        assert ctx.device_name == "second"


def test_open_by_index_rejects_out_of_range() -> None:
    """An index past the enumeration is an error, not a silent fallback."""
    lib = _FakeVulkan()
    with pytest.raises(vc.VulkanError, match="out of range"):
        _open(lib, device_index=3)
    assert lib.events == ["destroy-instance"]


def test_open_by_index_rejects_device_without_compute() -> None:
    """Explicitly selecting a graphics-only device fails loudly."""
    lib = _FakeVulkan((("display-only", 1 << 22, (_GRAPHICS_QUEUE,)),))
    with pytest.raises(vc.VulkanError, match="no compute queue family"):
        _open(lib, device_index=0)


def test_open_without_compute_device_fails() -> None:
    """Auto-selection with nothing compute-capable reports why."""
    lib = _FakeVulkan((("display-only", 1 << 22, ()),))
    with pytest.raises(vc.VulkanError, match="no enumerated Vulkan device"):
        _open(lib)
    # The throwaway instance is still released on the failure path.
    assert lib.events == ["destroy-instance"]


def test_open_without_devices_fails() -> None:
    """A loader that enumerates nothing cannot be opened."""
    with pytest.raises(vc.VulkanError, match="no physical devices"):
        _open(_FakeVulkan(()))


@pytest.mark.parametrize(
    ("entry_point", "message"),
    [
        ("vkCreateInstance", "vkCreateInstance failed: VK_ERROR_INITIALIZATION_FAILED"),
        (
            "vkEnumeratePhysicalDevices",
            "vkEnumeratePhysicalDevices failed: VK_ERROR_INITIALIZATION_FAILED",
        ),
        (
            "vkEnumeratePhysicalDevicesFill",
            "vkEnumeratePhysicalDevices failed: VK_ERROR_INITIALIZATION_FAILED",
        ),
        ("vkCreateDevice", "vkCreateDevice failed: VK_ERROR_INITIALIZATION_FAILED"),
    ],
)
def test_open_propagates_entry_point_failures(entry_point: str, message: str) -> None:
    """Every step of opening a device reports the failing call and result."""
    lib = _FakeVulkan(fail={entry_point: -3})
    with pytest.raises(vc.VulkanError, match=message):
        _open(lib)


def test_open_rejects_null_instance() -> None:
    """A driver that returns success and a null handle is still a failure."""
    with pytest.raises(vc.VulkanError, match="null instance"):
        _open(_FakeVulkan(null_instance=True))


def test_open_reports_unknown_result_codes() -> None:
    """An unnamed VkResult is surfaced numerically rather than swallowed."""
    with pytest.raises(vc.VulkanError, match=r"VkResult -4242"):
        _open(_FakeVulkan(fail={"vkCreateInstance": -4242}))


def test_open_without_loader_fails() -> None:
    """No ICD loader means no context."""
    with pytest.raises(vc.VulkanError, match="no Vulkan ICD loader"):
        vc.VulkanContext.open(find_loader=lambda: None)


def test_open_with_unloadable_loader_fails() -> None:
    """A loader name that will not load is reported with the OS error."""

    def boom(name: str) -> ctypes.CDLL:
        raise OSError("cannot open shared object file")

    with pytest.raises(vc.VulkanError, match="cannot load Vulkan loader"):
        vc.VulkanContext.open(
            find_loader=lambda: "libvulkan.so.missing", load_library=boom
        )


def test_default_loader_uses_ctypes(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no injection the loader is opened through ``ctypes.CDLL``."""
    lib = _FakeVulkan()
    names: list[str] = []

    def fake_cdll(name: str) -> Any:
        names.append(name)
        return lib

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    with vc.VulkanContext.open(find_loader=lambda: "libvulkan.so.1") as ctx:
        assert ctx.device_name == "V3D 7.1.7.0"
    assert names == ["libvulkan.so.1"]


# --------------------------------------------------------------------------
# buffers
# --------------------------------------------------------------------------


def test_buffer_roundtrips_an_array() -> None:
    """Bytes written into a mapped buffer come back unchanged."""
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    with _open(_FakeVulkan()) as ctx, ctx.allocate(data.nbytes) as buf:
        assert buf.nbytes == data.nbytes
        buf.write(data)
        out = buf.read(data.dtype, data.shape)
        assert out.dtype == data.dtype
        assert np.array_equal(out, data)


def test_buffer_read_defaults_to_flat_view() -> None:
    """Without a shape the whole allocation is read as a 1-D array."""
    with _open(_FakeVulkan()) as ctx, ctx.allocate(16) as buf:
        buf.write(np.array([1, 2, 3, 4], dtype=np.int32))
        assert np.array_equal(buf.read(np.int32), [1, 2, 3, 4])


def test_buffer_read_survives_free() -> None:
    """The returned array owns its memory, so freeing the buffer is safe."""
    with _open(_FakeVulkan()) as ctx:
        buf = ctx.allocate(16)
        buf.write(np.array([5, 6, 7, 8], dtype=np.int32))
        out = buf.read(np.int32)
        buf.free()
        assert np.array_equal(out, [5, 6, 7, 8])


def test_buffer_write_accepts_non_contiguous_input() -> None:
    """A transposed view is made contiguous before the copy."""
    data = np.arange(12, dtype=np.float32).reshape(3, 4).T
    with _open(_FakeVulkan()) as ctx, ctx.allocate(data.nbytes) as buf:
        buf.write(data)
        assert np.array_equal(buf.read(np.float32, (4, 3)), data)


def test_buffer_write_rejects_oversized_array() -> None:
    """Writing more bytes than were allocated is refused, not truncated."""
    with (
        _open(_FakeVulkan()) as ctx,
        ctx.allocate(8) as buf,
        pytest.raises(vc.VulkanError, match="does not fit"),
    ):
        buf.write(np.zeros(4, dtype=np.float32))


def test_buffer_read_rejects_oversized_view() -> None:
    """A shape larger than the allocation is refused."""
    with (
        _open(_FakeVulkan()) as ctx,
        ctx.allocate(8) as buf,
        pytest.raises(vc.VulkanError, match="reading 16 bytes"),
    ):
        buf.read(np.float32, (4,))


def test_buffer_read_rejects_ragged_flat_view() -> None:
    """A flat read needs the size to be a whole number of elements."""
    with (
        _open(_FakeVulkan()) as ctx,
        ctx.allocate(6) as buf,
        pytest.raises(vc.VulkanError, match="whole number"),
    ):
        buf.read(np.float32)


@pytest.mark.parametrize("dtype", ["U", "S0", "V0"])
def test_buffer_read_rejects_sizeless_dtype(dtype: str) -> None:
    """A flexible dtype has no element size, so it cannot describe a view."""
    with (
        _open(_FakeVulkan()) as ctx,
        ctx.allocate(16) as buf,
        pytest.raises(vc.VulkanError, match="no fixed element size"),
    ):
        buf.read(dtype, (2,))


def test_buffer_read_rejects_negative_dimension() -> None:
    """A negative dimension would shrink the byte count instead of growing it."""
    with (
        _open(_FakeVulkan()) as ctx,
        ctx.allocate(16) as buf,
        pytest.raises(vc.VulkanError, match="negative dimension"),
    ):
        buf.read(np.float32, (-1, 4))


def test_freed_buffer_refuses_access() -> None:
    """Reads and writes after ``free`` raise instead of touching stale memory."""
    with _open(_FakeVulkan()) as ctx:
        buf = ctx.allocate(16)
        buf.free()
        assert buf.freed is True
        with pytest.raises(vc.VulkanError, match="has been freed"):
            buf.write(np.zeros(4, dtype=np.float32))
        with pytest.raises(vc.VulkanError, match="has been freed"):
            buf.read(np.float32)


def test_free_is_idempotent() -> None:
    """Freeing twice releases the Vulkan objects once."""
    lib = _FakeVulkan()
    with _open(lib) as ctx:
        buf = ctx.allocate(16)
        buf.free()
        buf.free()
    assert lib.events.count("free-memory") == 1
    assert lib.events.count("destroy-buffer") == 1


def test_allocation_rounds_up_to_requirements() -> None:
    """The driver's reported allocation size wins over the requested size."""
    lib = _FakeVulkan()
    with _open(lib) as ctx, ctx.allocate(100) as buf:
        assert buf.nbytes == 100
        assert list(lib.allocation_sizes.values()) == [256]


def test_allocation_uses_the_selected_memory_type() -> None:
    """The chosen memory type index reaches ``vkAllocateMemory``."""
    lib = _FakeVulkan(memory_types=(0x1, _HOST_COHERENT, _DEVICE_LOCAL_COHERENT))
    with _open(lib) as ctx, ctx.allocate(16):
        assert lib.last_memory_type == 2


def test_memory_type_count_is_clamped() -> None:
    """A driver reporting more types than Vulkan allows is clamped, not trusted."""
    lib = _FakeVulkan(
        memory_types=(_DEVICE_LOCAL_COHERENT,) * 32, reported_type_count=99
    )
    with _open(lib) as ctx:
        assert len(ctx._memory_property_flags()) == 32


@pytest.mark.parametrize(
    "entry_point",
    ["vkCreateBuffer", "vkAllocateMemory", "vkBindBufferMemory", "vkMapMemory"],
)
def test_allocation_failures_release_partial_state(entry_point: str) -> None:
    """A failed allocation leaves no buffer or memory behind."""
    lib = _FakeVulkan(fail={entry_point: -2})
    with _open(lib) as ctx:
        with pytest.raises(vc.VulkanError, match="VK_ERROR_OUT_OF_DEVICE_MEMORY"):
            ctx.allocate(64)
        assert ctx.live_buffers == 0
    assert lib.allocations == {}
    assert lib.buffer_sizes == {}


def test_null_mapping_is_an_error() -> None:
    """A successful map that yields a null pointer is rejected."""
    lib = _FakeVulkan(map_null=True)
    with _open(lib) as ctx, pytest.raises(vc.VulkanError, match="null pointer"):
        ctx.allocate(64)
    assert lib.allocations == {}


def test_allocation_without_usable_memory_type_fails() -> None:
    """No host-visible coherent type means no buffer, and no leak."""
    lib = _FakeVulkan(memory_types=(0x1,))
    with (
        _open(lib) as ctx,
        pytest.raises(vc.VulkanError, match="no host-visible coherent"),
    ):
        ctx.allocate(64)
    assert lib.buffer_sizes == {}


@pytest.mark.parametrize("nbytes", [0, -8, 4.0, True, "16"])
def test_allocation_rejects_invalid_sizes(nbytes: object) -> None:
    """Sizes that are not positive integers are rejected before any Vulkan call."""
    with (
        _open(_FakeVulkan()) as ctx,
        pytest.raises(vc.VulkanError, match="positive integer"),
    ):
        ctx.allocate(nbytes)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# lifetime
# --------------------------------------------------------------------------


def test_close_frees_outstanding_buffers() -> None:
    """Closing the context cleans up buffers the caller forgot."""
    lib = _FakeVulkan()
    ctx = _open(lib)
    first = ctx.allocate(16)
    second = ctx.allocate(32)
    assert ctx.live_buffers == 2
    ctx.close()
    assert (first.freed, second.freed) == (True, True)
    assert ctx.live_buffers == 0
    assert lib.allocations == {}
    assert lib.mapped == set()
    assert lib.events[-2:] == ["destroy-device", "destroy-instance"]


def test_close_is_idempotent() -> None:
    """A second close destroys nothing a second time."""
    lib = _FakeVulkan()
    ctx = _open(lib)
    ctx.close()
    ctx.close()
    assert lib.events.count("destroy-device") == 1
    assert lib.events.count("destroy-instance") == 1


def test_allocate_after_close_fails() -> None:
    """A closed context has no device to allocate from."""
    ctx = _open(_FakeVulkan())
    ctx.close()
    with pytest.raises(vc.VulkanError, match="context is closed"):
        ctx.allocate(16)


def test_context_manager_closes_on_exception() -> None:
    """An exception inside the ``with`` block still tears the context down."""
    lib = _FakeVulkan()
    with pytest.raises(ZeroDivisionError), _open(lib) as ctx:
        ctx.allocate(16)
        raise ZeroDivisionError
    assert ctx.closed is True
    assert lib.allocations == {}


# --------------------------------------------------------------------------
# real hardware (skipped where no Vulkan device is installed)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not vulkan_available(), reason="no Vulkan compute device")
def test_roundtrip_on_a_real_device() -> None:
    """On a host with Vulkan, a real device round-trips an array unchanged."""
    data = np.arange(256, dtype=np.float32).reshape(4, 64) / 3.0
    with vc.VulkanContext.open() as ctx:
        assert ctx.device_name
        assert ctx.queue_family_index >= 0
        with ctx.allocate(data.nbytes) as buf:
            buf.write(data)
            assert np.array_equal(buf.read(data.dtype, data.shape), data)
        assert ctx.live_buffers == 0
