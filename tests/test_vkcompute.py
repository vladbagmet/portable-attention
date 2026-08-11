"""Tests for the Vulkan compute context (device, buffers and shader dispatch)."""

from __future__ import annotations

import ctypes
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import portable_attention as pa
from portable_attention import vkcompute as vc
from portable_attention.vulkan import detect_vulkan, vulkan_available

_INSTANCE = 0xBEEF
_DEVICE = 0xD3D1
_QUEUE = 0x0EE0

# out[i] = in[i] * scale, compiled from tests/shaders/double.comp. Two storage
# buffers, an 8-byte push-constant block (uint count, float scale), 64 threads
# per workgroup.
_DOUBLE_SPV = Path(__file__).parent / "shaders" / "double.spv"
_DOUBLE_LOCAL_SIZE = 64

# dst[i] = NEGATE ? -(src[i] * FACTOR + OFFSET) : ..., compiled from
# tests/shaders/specialized.comp. Workgroup size (id 0) and the three constants
# (ids 1-3) are all specialization constants; the defaults are a local size of
# 1, OFFSET 0, FACTOR 1.0 and NEGATE false, i.e. a plain copy.
_SPECIALIZED_SPV = Path(__file__).parent / "shaders" / "specialized.spv"

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
        self.descriptors: list[tuple[int, int]] = []
        self.push_constants = b""
        self.specialization: list[tuple[int, bytes]] = []
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

    # -- pipeline objects -------------------------------------------------
    def _create(self, name: str, out: Any, event: str) -> int:
        """Emulate a ``vkCreate*`` that writes one handle through a pointer."""
        status = self._status(name)
        if status != 0:
            return status
        out[0] = self._handle()
        self.events.append(event)
        return status

    def vkCreateShaderModule(  # noqa: N802
        self, device: Any, info: Any, allocator: Any, module: Any
    ) -> int:
        self.shader_words = info.contents.codeSize // 4
        return self._create("vkCreateShaderModule", module, "create-shader")

    def vkDestroyShaderModule(self, device: Any, module: Any, allocator: Any) -> None:  # noqa: N802
        self.events.append("destroy-shader")

    def vkCreateDescriptorSetLayout(  # noqa: N802
        self, device: Any, info: Any, allocator: Any, layout: Any
    ) -> int:
        self.binding_count = info.contents.bindingCount
        return self._create("vkCreateDescriptorSetLayout", layout, "create-set-layout")

    def vkDestroyDescriptorSetLayout(  # noqa: N802
        self, device: Any, layout: Any, allocator: Any
    ) -> None:
        self.events.append("destroy-set-layout")

    def vkCreateDescriptorPool(  # noqa: N802
        self, device: Any, info: Any, allocator: Any, pool: Any
    ) -> int:
        assert info.contents.maxSets == 1
        return self._create("vkCreateDescriptorPool", pool, "create-descriptor-pool")

    def vkDestroyDescriptorPool(self, device: Any, pool: Any, allocator: Any) -> None:  # noqa: N802
        self.events.append("destroy-descriptor-pool")

    def vkAllocateDescriptorSets(self, device: Any, info: Any, sets: Any) -> int:  # noqa: N802
        return self._create("vkAllocateDescriptorSets", sets, "allocate-set")

    def vkCreatePipelineLayout(  # noqa: N802
        self, device: Any, info: Any, allocator: Any, layout: Any
    ) -> int:
        self.push_constant_ranges = info.contents.pushConstantRangeCount
        return self._create("vkCreatePipelineLayout", layout, "create-pipeline-layout")

    def vkDestroyPipelineLayout(self, device: Any, layout: Any, allocator: Any) -> None:  # noqa: N802
        self.events.append("destroy-pipeline-layout")

    def vkCreateComputePipelines(  # noqa: N802
        self,
        device: Any,
        cache: Any,
        count: Any,
        infos: Any,
        allocator: Any,
        pipelines: Any,
    ) -> int:
        assert count.value == 1
        self.entry_point = infos.contents.stage.pName
        self.specialization = _read_specialization(infos.contents.stage)
        return self._create("vkCreateComputePipelines", pipelines, "create-pipeline")

    def vkDestroyPipeline(self, device: Any, pipeline: Any, allocator: Any) -> None:  # noqa: N802
        self.events.append("destroy-pipeline")

    def vkCreateCommandPool(  # noqa: N802
        self, device: Any, info: Any, allocator: Any, pool: Any
    ) -> int:
        self.command_pool_family = info.contents.queueFamilyIndex
        return self._create("vkCreateCommandPool", pool, "create-command-pool")

    def vkDestroyCommandPool(self, device: Any, pool: Any, allocator: Any) -> None:  # noqa: N802
        self.events.append("destroy-command-pool")

    def vkAllocateCommandBuffers(self, device: Any, info: Any, buffers: Any) -> int:  # noqa: N802
        return self._create("vkAllocateCommandBuffers", buffers, "allocate-command")

    def vkCreateFence(  # noqa: N802
        self, device: Any, info: Any, allocator: Any, fence: Any
    ) -> int:
        return self._create("vkCreateFence", fence, "create-fence")

    def vkDestroyFence(self, device: Any, fence: Any, allocator: Any) -> None:  # noqa: N802
        self.events.append("destroy-fence")

    # -- recording / submission -------------------------------------------
    def vkUpdateDescriptorSets(  # noqa: N802
        self, device: Any, write_count: Any, writes: Any, copy_count: Any, copies: Any
    ) -> None:
        self.descriptors = [
            (
                writes[index].dstBinding,
                ctypes.cast(
                    writes[index].pBufferInfo,
                    ctypes.POINTER(vc._VkDescriptorBufferInfo),
                ).contents.buffer,
            )
            for index in range(write_count.value)
        ]
        self.events.append("update-descriptors")

    def vkResetCommandBuffer(self, command_buffer: Any, flags: Any) -> int:  # noqa: N802
        self.events.append("reset-command")
        return self._status("vkResetCommandBuffer")

    def vkBeginCommandBuffer(self, command_buffer: Any, info: Any) -> int:  # noqa: N802
        self.events.append("begin")
        return self._status("vkBeginCommandBuffer")

    def vkCmdBindPipeline(  # noqa: N802
        self, command_buffer: Any, bind_point: Any, pipeline: Any
    ) -> None:
        assert bind_point.value == 1
        self.events.append("bind-pipeline")

    def vkCmdBindDescriptorSets(  # noqa: N802
        self,
        command_buffer: Any,
        bind_point: Any,
        layout: Any,
        first_set: Any,
        set_count: Any,
        sets: Any,
        offset_count: Any,
        offsets: Any,
    ) -> None:
        self.events.append("bind-sets")

    def vkCmdPushConstants(  # noqa: N802
        self,
        command_buffer: Any,
        layout: Any,
        stages: Any,
        offset: Any,
        size: Any,
        values: Any,
    ) -> None:
        self.push_constants = bytes(values)[: size.value]
        self.events.append("push-constants")

    def vkCmdDispatch(self, command_buffer: Any, x: Any, y: Any, z: Any) -> None:  # noqa: N802
        self.groups = (x.value, y.value, z.value)
        self.events.append("dispatch")

    def vkCmdPipelineBarrier(  # noqa: N802
        self,
        command_buffer: Any,
        src_stage: Any,
        dst_stage: Any,
        flags: Any,
        memory_count: Any,
        memory: Any,
        buffer_count: Any,
        buffers: Any,
        image_count: Any,
        images: Any,
    ) -> None:
        self.barrier = (
            src_stage.value,
            dst_stage.value,
            memory.contents.srcAccessMask,
            memory.contents.dstAccessMask,
        )
        self.events.append("barrier")

    def vkEndCommandBuffer(self, command_buffer: Any) -> int:  # noqa: N802
        self.events.append("end")
        return self._status("vkEndCommandBuffer")

    def vkResetFences(self, device: Any, count: Any, fences: Any) -> int:  # noqa: N802
        self.events.append("reset-fences")
        return self._status("vkResetFences")

    def vkQueueSubmit(  # noqa: N802
        self, queue: Any, count: Any, submits: Any, fence: Any
    ) -> int:
        assert queue.value == _QUEUE
        assert submits.contents.commandBufferCount == 1
        self.events.append("submit")
        return self._status("vkQueueSubmit")

    def vkWaitForFences(  # noqa: N802
        self, device: Any, count: Any, fences: Any, wait_all: Any, timeout: Any
    ) -> int:
        self.wait_timeout_ns = timeout.value
        self.events.append("wait")
        return self._status("vkWaitForFences")

    def vkDeviceWaitIdle(self, device: Any) -> int:  # noqa: N802
        self.events.append("device-idle")
        return 0


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


def _read_specialization(
    stage: vc._VkPipelineShaderStageCreateInfo,
) -> list[tuple[int, bytes]]:
    """Decode ``pSpecializationInfo`` the way a driver would: id -> raw bytes."""
    if not stage.pSpecializationInfo:
        return []
    info = ctypes.cast(
        stage.pSpecializationInfo, ctypes.POINTER(vc._VkSpecializationInfo)
    ).contents
    blob = ctypes.string_at(info.pData, info.dataSize)
    entries = ctypes.cast(
        info.pMapEntries, ctypes.POINTER(vc._VkSpecializationMapEntry)
    )
    return [
        (
            entries[index].constantID,
            blob[entries[index].offset : entries[index].offset + entries[index].size],
        )
        for index in range(info.mapEntryCount)
    ]


def _spirv(words: list[int] | None = None) -> bytes:
    """Build a minimal well-formed SPIR-V header, or read the real kernel."""
    if words is None:
        return _DOUBLE_SPV.read_bytes()
    return b"".join(word.to_bytes(4, "little") for word in words)


_MINIMAL_SPIRV = _spirv([0x07230203, 0x00010000, 0, 1, 0])


def test_public_reexports() -> None:
    """The context, buffer and error type are top-level exports."""
    for name in ("VulkanBuffer", "VulkanContext", "VulkanError", "VulkanPipeline"):
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
# pipeline argument validation (pure)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("not bytes", "must be bytes-like"),
        (b"", "whole number of 32-bit words"),
        (b"\x03\x02#\x07\x00", "whole number of 32-bit words"),
        (b"\x07#\x02\x03", "byte-swapped"),
        (b"\x00\x00\x00\x00", "not a SPIR-V module"),
    ],
)
def test_spirv_header_is_validated(code: object, message: str) -> None:
    """A module the driver would reject (or crash on) is caught first."""
    with pytest.raises(vc.VulkanError, match=message):
        vc._require_spirv(code)


def test_spirv_accepts_bytes_like_input() -> None:
    """``bytearray`` and ``memoryview`` are copied, not refused."""
    assert vc._require_spirv(bytearray(_MINIMAL_SPIRV)) == _MINIMAL_SPIRV
    assert vc._require_spirv(memoryview(_MINIMAL_SPIRV)) == _MINIMAL_SPIRV


@pytest.mark.parametrize("count", [0, -1, True, 2.0, "2"])
def test_buffer_count_must_be_a_positive_int(count: object) -> None:
    """Binding counts index descriptor slots, so only positive ints make sense."""
    with pytest.raises(vc.VulkanError, match="buffer_count must be"):
        vc._require_buffer_count(count)


@pytest.mark.parametrize(
    ("size", "message"),
    [
        (-4, "non-negative integer"),
        (True, "non-negative integer"),
        (4.0, "non-negative integer"),
        (6, "multiple of 4"),
        (132, "exceeds the 128 bytes"),
    ],
)
def test_push_constant_size_is_validated(size: object, message: str) -> None:
    """Push-constant blocks must be word-sized and inside the portable floor."""
    with pytest.raises(vc.VulkanError, match=message):
        vc._require_push_constant_bytes(size)


@pytest.mark.parametrize("size", [0, 4, 128])
def test_push_constant_size_accepts_the_portable_range(size: int) -> None:
    """No block, one word, and the 128 bytes every implementation has all pass."""
    assert vc._require_push_constant_bytes(size) == size


@pytest.mark.parametrize(
    ("groups", "expected"),
    [(7, (7, 1, 1)), ((3,), (3, 1, 1)), ((3, 2), (3, 2, 1)), ([3, 2, 1], (3, 2, 1))],
)
def test_groups_pad_to_three_dimensions(
    groups: object, expected: tuple[int, int, int]
) -> None:
    """Workgroup counts accept 1-3 axes and default the rest to one."""
    assert vc._require_groups(groups) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ((), "1 to 3 dimensions"),
        ((1, 1, 1, 1), "1 to 3 dimensions"),
        ((1, 0), "must be positive integers"),
        ((1.5,), "must be positive integers"),
        (True, "must be positive integers"),
        ((1, 65536), "exceeds the 65535 per axis"),
    ],
)
def test_groups_reject_nonsense(groups: object, message: str) -> None:
    """An empty, non-positive or unportably large workgroup count is an error."""
    with pytest.raises(vc.VulkanError, match=message):
        vc._require_groups(groups)  # type: ignore[arg-type]


def test_groups_accept_the_portable_maximum() -> None:
    """The guaranteed 65535 per axis is allowed, one more is not."""
    assert vc._require_groups((65535, 65535, 65535)) == (65535, 65535, 65535)


# --------------------------------------------------------------------------
# building and dispatching a pipeline
# --------------------------------------------------------------------------


def test_pipeline_creates_every_object_it_needs() -> None:
    """Building a pipeline walks the full create sequence once."""
    lib = _FakeVulkan()
    with _open(lib) as ctx:
        pipeline = ctx.compute_pipeline(
            _MINIMAL_SPIRV, buffer_count=3, push_constant_bytes=8
        )
        assert (pipeline.buffer_count, pipeline.push_constant_bytes) == (3, 8)
        assert pipeline.dispatch_count == 0
        assert ctx.live_pipelines == 1
        assert lib.events == [
            "create-shader",
            "create-set-layout",
            "create-descriptor-pool",
            "allocate-set",
            "create-pipeline-layout",
            "create-pipeline",
            "create-command-pool",
            "allocate-command",
            "create-fence",
        ]
        assert lib.shader_words == len(_MINIMAL_SPIRV) // 4
        assert lib.binding_count == 3
        assert lib.push_constant_ranges == 1
        assert lib.entry_point == b"main"
        assert lib.command_pool_family == ctx.queue_family_index


def test_pipeline_without_push_constants_declares_no_range() -> None:
    """A shader with no push constants gets an empty range list."""
    lib = _FakeVulkan()
    with _open(lib) as ctx:
        ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1)
        assert lib.push_constant_ranges == 0


def test_pipeline_honours_a_custom_entry_point() -> None:
    """``entry_point`` reaches the shader stage as a NUL-terminated name."""
    lib = _FakeVulkan()
    with _open(lib) as ctx:
        ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1, entry_point="attention")
        assert lib.entry_point == b"attention"


def test_pipeline_without_specialization_passes_none() -> None:
    """Omitting the constants leaves ``pSpecializationInfo`` null."""
    lib = _FakeVulkan()
    with _open(lib) as ctx:
        pipeline = ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1)
        assert lib.specialization == []
        assert pipeline.specialization == {}


def test_specialization_constants_reach_the_shader_stage() -> None:
    """Each constant becomes a 4-byte slot, ordered by id whatever the input."""
    lib = _FakeVulkan()
    with _open(lib) as ctx:
        pipeline = ctx.compute_pipeline(
            _MINIMAL_SPIRV,
            buffer_count=1,
            specialization={7: 2.5, 0: 64, 3: True, 1: -2, 5: False},
        )
        assert lib.specialization == [
            (0, struct.pack("<I", 64)),
            (1, struct.pack("<i", -2)),
            (3, struct.pack("<I", 1)),
            (5, struct.pack("<I", 0)),
            (7, struct.pack("<f", 2.5)),
        ]
        assert pipeline.specialization == {7: 2.5, 0: 64, 3: True, 1: -2, 5: False}


def test_specialization_property_is_a_copy() -> None:
    """Mutating what the property returns cannot change the built pipeline."""
    lib = _FakeVulkan()
    with _open(lib) as ctx:
        pipeline = ctx.compute_pipeline(
            _MINIMAL_SPIRV, buffer_count=1, specialization={0: 16}
        )
        pipeline.specialization[0] = 32
        assert pipeline.specialization == {0: 16}


def test_empty_specialization_is_the_same_as_none() -> None:
    """An empty mapping means the shader keeps every default."""
    lib = _FakeVulkan()
    with _open(lib) as ctx:
        pipeline = ctx.compute_pipeline(
            _MINIMAL_SPIRV, buffer_count=1, specialization={}
        )
        assert lib.specialization == []
        assert pipeline.specialization == {}


@pytest.mark.parametrize(
    ("specialization", "message"),
    [
        ([(0, 1)], "must be a mapping"),
        ({"0": 1}, "constant id must be an integer"),
        ({True: 1}, "constant id must be an integer"),
        ({-1: 1}, "constant id must fit in a uint32"),
        ({1 << 32: 1}, "constant id must fit in a uint32"),
        ({0: "64"}, "must be a bool, int or float"),
        ({0: None}, "must be a bool, int or float"),
        ({0: 1 << 32}, "does not fit in a 32-bit int"),
        ({0: -(1 << 31) - 1}, "does not fit in a 32-bit int"),
    ],
)
def test_specialization_rejects_invalid_input(
    specialization: Any, message: str
) -> None:
    """Bad ids and unencodable values are refused before the driver sees them."""
    lib = _FakeVulkan()
    with _open(lib) as ctx, pytest.raises(vc.VulkanError, match=message):
        ctx.compute_pipeline(
            _MINIMAL_SPIRV, buffer_count=1, specialization=specialization
        )
        assert ctx.live_pipelines == 0


@pytest.mark.parametrize("value", [0, 0xFFFFFFFF, -(1 << 31)])
def test_specialization_accepts_the_32_bit_boundaries(value: int) -> None:
    """The widest ints a 4-byte slot can hold are packed, not rejected."""
    lib = _FakeVulkan()
    with _open(lib) as ctx:
        ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1, specialization={0: value})
        expected = struct.pack("<i" if value < 0 else "<I", value)
        assert lib.specialization == [(0, expected)]


def test_dispatch_records_bindings_constants_and_barrier() -> None:
    """One dispatch binds each buffer in order and ends with a host barrier."""
    lib = _FakeVulkan()
    with _open(lib) as ctx:
        src = ctx.allocate(64)
        dst = ctx.allocate(64)
        pipeline = ctx.compute_pipeline(
            _MINIMAL_SPIRV, buffer_count=2, push_constant_bytes=8
        )
        payload = struct.pack("<If", 16, 2.0)
        lib.events.clear()
        pipeline.dispatch([src, dst], groups=(4, 2), push_constants=payload)

        assert lib.events == [
            "update-descriptors",
            "reset-command",
            "begin",
            "bind-pipeline",
            "bind-sets",
            "push-constants",
            "dispatch",
            "barrier",
            "end",
            "reset-fences",
            "submit",
            "wait",
        ]
        assert lib.descriptors == [(0, src.handle), (1, dst.handle)]
        assert lib.push_constants == payload
        assert lib.groups == (4, 2, 1)
        # compute-shader writes -> host reads
        assert lib.barrier == (0x800, 0x4000, 0x40, 0x2000)
        assert pipeline.dispatch_count == 1


def test_dispatch_skips_push_constants_when_there_are_none() -> None:
    """A pipeline with no push-constant block records no push command."""
    lib = _FakeVulkan()
    with _open(lib) as ctx:
        buf = ctx.allocate(16)
        pipeline = ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1)
        lib.events.clear()
        pipeline.dispatch([buf], groups=1)
        assert "push-constants" not in lib.events


def test_dispatch_reuses_the_command_buffer() -> None:
    """A second dispatch re-records rather than allocating new objects."""
    lib = _FakeVulkan()
    with _open(lib) as ctx:
        buf = ctx.allocate(16)
        pipeline = ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1)
        pipeline.dispatch([buf], groups=1)
        pipeline.dispatch([buf], groups=2)
        assert pipeline.dispatch_count == 2
        assert lib.groups == (2, 1, 1)
        assert lib.events.count("allocate-command") == 1
        assert lib.events.count("reset-command") == 2


def test_dispatch_converts_the_timeout_to_nanoseconds() -> None:
    """``timeout_s`` reaches ``vkWaitForFences`` in its own unit."""
    lib = _FakeVulkan()
    with _open(lib) as ctx:
        buf = ctx.allocate(16)
        pipeline = ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1)
        pipeline.dispatch([buf], groups=1, timeout_s=0.25)
        assert lib.wait_timeout_ns == 250_000_000


def test_dispatch_rejects_the_wrong_number_of_buffers() -> None:
    """The buffer list has to match the descriptor set the pipeline was built for."""
    with _open(_FakeVulkan()) as ctx:
        buf = ctx.allocate(16)
        pipeline = ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=2)
        with pytest.raises(vc.VulkanError, match=r"binds 2 buffer\(s\), got 1"):
            pipeline.dispatch([buf], groups=1)


def test_dispatch_rejects_a_mismatched_push_constant_block() -> None:
    """Push-constant payloads must be exactly the declared size."""
    with _open(_FakeVulkan()) as ctx:
        buf = ctx.allocate(16)
        pipeline = ctx.compute_pipeline(
            _MINIMAL_SPIRV, buffer_count=1, push_constant_bytes=8
        )
        with pytest.raises(vc.VulkanError, match="declares 8 push-constant"):
            pipeline.dispatch([buf], groups=1, push_constants=b"\x00\x00\x00\x00")
        with pytest.raises(vc.VulkanError, match="declares 8 push-constant"):
            pipeline.dispatch([buf], groups=1)


@pytest.mark.parametrize("timeout", [0.0, -1.0])
def test_dispatch_rejects_a_non_positive_timeout(timeout: float) -> None:
    """Waiting for zero time would report a healthy device as wedged."""
    with _open(_FakeVulkan()) as ctx:
        buf = ctx.allocate(16)
        pipeline = ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1)
        with pytest.raises(vc.VulkanError, match="timeout_s must be positive"):
            pipeline.dispatch([buf], groups=1, timeout_s=timeout)


def test_dispatch_rejects_a_freed_buffer() -> None:
    """Binding a freed buffer is caught instead of passing a stale handle."""
    with _open(_FakeVulkan()) as ctx:
        buf = ctx.allocate(16)
        pipeline = ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1)
        buf.free()
        with pytest.raises(vc.VulkanError, match="has been freed"):
            pipeline.dispatch([buf], groups=1)


def test_pipeline_after_close_fails() -> None:
    """A closed context has no device to compile a shader for."""
    ctx = _open(_FakeVulkan())
    ctx.close()
    with pytest.raises(vc.VulkanError, match="context is closed"):
        ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1)


@pytest.mark.parametrize(
    ("entry_point", "expected_teardown"),
    [
        ("vkCreateShaderModule", []),
        ("vkCreateDescriptorSetLayout", ["destroy-shader"]),
        (
            "vkCreateDescriptorPool",
            ["destroy-set-layout", "destroy-shader"],
        ),
        (
            "vkAllocateDescriptorSets",
            ["destroy-descriptor-pool", "destroy-set-layout", "destroy-shader"],
        ),
        (
            "vkCreatePipelineLayout",
            ["destroy-descriptor-pool", "destroy-set-layout", "destroy-shader"],
        ),
        (
            "vkCreateComputePipelines",
            [
                "destroy-pipeline-layout",
                "destroy-descriptor-pool",
                "destroy-set-layout",
                "destroy-shader",
            ],
        ),
        (
            "vkCreateCommandPool",
            [
                "destroy-pipeline",
                "destroy-pipeline-layout",
                "destroy-descriptor-pool",
                "destroy-set-layout",
                "destroy-shader",
            ],
        ),
        (
            "vkAllocateCommandBuffers",
            [
                "destroy-command-pool",
                "destroy-pipeline",
                "destroy-pipeline-layout",
                "destroy-descriptor-pool",
                "destroy-set-layout",
                "destroy-shader",
            ],
        ),
        (
            "vkCreateFence",
            [
                "destroy-command-pool",
                "destroy-pipeline",
                "destroy-pipeline-layout",
                "destroy-descriptor-pool",
                "destroy-set-layout",
                "destroy-shader",
            ],
        ),
    ],
)
def test_pipeline_build_failure_unwinds(
    entry_point: str, expected_teardown: list[str]
) -> None:
    """A failure part-way through creation destroys what already exists."""
    lib = _FakeVulkan(fail={entry_point: -1})
    with _open(lib) as ctx:
        with pytest.raises(vc.VulkanError, match=f"{entry_point} failed"):
            ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=2)
        assert ctx.live_pipelines == 0
        teardown = [event for event in lib.events if event.startswith("destroy-")]
        assert teardown == expected_teardown


@pytest.mark.parametrize(
    "entry_point",
    [
        "vkResetCommandBuffer",
        "vkBeginCommandBuffer",
        "vkEndCommandBuffer",
        "vkResetFences",
        "vkQueueSubmit",
    ],
)
def test_dispatch_propagates_entry_point_failures(entry_point: str) -> None:
    """Every recording and submission step reports the call that failed."""
    lib = _FakeVulkan(fail={entry_point: -4})
    with _open(lib) as ctx:
        buf = ctx.allocate(16)
        pipeline = ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1)
        with pytest.raises(
            vc.VulkanError, match=f"{entry_point} failed: VK_ERROR_DEVICE_LOST"
        ):
            pipeline.dispatch([buf], groups=1)
        assert pipeline.dispatch_count == 0


def test_dispatch_reports_a_fence_timeout() -> None:
    """A device that never signals is an error, not a silently short read."""
    lib = _FakeVulkan(fail={"vkWaitForFences": 2})
    with _open(lib) as ctx:
        buf = ctx.allocate(16)
        pipeline = ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1)
        with pytest.raises(vc.VulkanError, match=r"after 0.5s\) failed: VK_TIMEOUT"):
            pipeline.dispatch([buf], groups=1, timeout_s=0.5)


def test_a_timed_out_pipeline_refuses_further_dispatch() -> None:
    """The submission is still in flight, so nothing may be re-recorded."""
    lib = _FakeVulkan(fail={"vkWaitForFences": 2})
    with _open(lib) as ctx:
        buf = ctx.allocate(16)
        pipeline = ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1)
        with pytest.raises(vc.VulkanError, match="VK_TIMEOUT"):
            pipeline.dispatch([buf], groups=1)
        with pytest.raises(vc.VulkanError, match="earlier dispatch never completed"):
            pipeline.dispatch([buf], groups=1)
        assert lib.events.count("reset-command") == 1


def test_a_timed_out_pipeline_drains_the_device_before_destroy() -> None:
    """Objects a pending submission references are only destroyed after idle."""
    lib = _FakeVulkan(fail={"vkWaitForFences": 2})
    with _open(lib) as ctx:
        buf = ctx.allocate(16)
        pipeline = ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1)
        with pytest.raises(vc.VulkanError, match="VK_TIMEOUT"):
            pipeline.dispatch([buf], groups=1)
        pipeline.destroy()
    assert lib.events.index("device-idle") < lib.events.index("destroy-command-pool")


def test_pipeline_destroy_is_idempotent() -> None:
    """Destroying twice releases each Vulkan object once."""
    lib = _FakeVulkan()
    with _open(lib) as ctx:
        pipeline = ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1)
        pipeline.destroy()
        pipeline.destroy()
        assert pipeline.destroyed is True
        assert ctx.live_pipelines == 0
    assert lib.events.count("destroy-pipeline") == 1
    assert lib.events.count("destroy-fence") == 1


def test_destroyed_pipeline_refuses_dispatch() -> None:
    """Dispatching after destroy raises instead of using dead handles."""
    with _open(_FakeVulkan()) as ctx:
        buf = ctx.allocate(16)
        pipeline = ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1)
        pipeline.destroy()
        with pytest.raises(vc.VulkanError, match="pipeline has been destroyed"):
            pipeline.dispatch([buf], groups=1)


def test_pipeline_is_a_context_manager() -> None:
    """Leaving the ``with`` block destroys the pipeline."""
    lib = _FakeVulkan()
    with (
        _open(lib) as ctx,
        ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1) as pipeline,
    ):
        assert pipeline.destroyed is False
    assert pipeline.destroyed is True


def test_close_destroys_outstanding_pipelines_before_the_device() -> None:
    """The context cleans up pipelines the caller forgot, in the right order."""
    lib = _FakeVulkan()
    ctx = _open(lib)
    pipeline = ctx.compute_pipeline(_MINIMAL_SPIRV, buffer_count=1)
    ctx.close()
    assert pipeline.destroyed is True
    assert ctx.live_pipelines == 0
    assert lib.events.index("destroy-pipeline") < lib.events.index("destroy-device")


def test_freed_buffer_has_no_handle() -> None:
    """The descriptor handle of a freed buffer is refused, not stale."""
    with _open(_FakeVulkan()) as ctx:
        buf = ctx.allocate(16)
        assert buf.handle > 0
        buf.free()
        with pytest.raises(vc.VulkanError, match="has been freed"):
            _ = buf.handle


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


def _compute_device_indices() -> list[int]:
    """Indices of every enumerated device that can run compute, or ``[]``."""
    capability = detect_vulkan()
    return [index for index, device in enumerate(capability.devices) if device.compute]


@pytest.mark.skipif(not vulkan_available(), reason="no Vulkan compute device")
@pytest.mark.parametrize("device_index", _compute_device_indices())
def test_dispatch_on_a_real_device(device_index: int) -> None:
    """Every compute device on this host runs the kernel and returns its result.

    Scales 1000 floats by a push constant, which exercises the whole submission
    path: descriptor writes, push constants, dispatch, barrier and fence wait.
    Runs once per compute-capable device, so a host with both a GPU and a
    software implementation checks both.
    """
    data = np.arange(1000, dtype=np.float32)
    groups = (data.size + _DOUBLE_LOCAL_SIZE - 1) // _DOUBLE_LOCAL_SIZE
    with vc.VulkanContext.open(device_index=device_index) as ctx:
        src = ctx.allocate(data.nbytes)
        dst = ctx.allocate(data.nbytes)
        src.write(data)
        with ctx.compute_pipeline(
            _spirv(), buffer_count=2, push_constant_bytes=8
        ) as pipeline:
            pipeline.dispatch(
                [src, dst],
                groups=groups,
                push_constants=struct.pack("<If", data.size, 2.0),
            )
            assert np.array_equal(dst.read(np.float32, data.shape), data * 2.0)

            # The same pipeline re-dispatched with new constants.
            pipeline.dispatch(
                [src, dst],
                groups=groups,
                push_constants=struct.pack("<If", data.size, -0.5),
            )
            assert np.array_equal(dst.read(np.float32, data.shape), data * -0.5)
            assert pipeline.dispatch_count == 2
        assert ctx.live_pipelines == 0


@pytest.mark.skipif(not vulkan_available(), reason="no Vulkan compute device")
@pytest.mark.parametrize("device_index", _compute_device_indices())
def test_specialization_on_a_real_device(device_index: int) -> None:
    """A driver compiles the same module twice with different constants.

    The workgroup size is itself specialized, so a run whose constants were
    dropped would also cover the wrong slice of the buffer, not merely compute
    the wrong value.
    """
    data = np.arange(1000, dtype=np.float32)
    count = struct.pack("<I", data.size)
    spirv = _SPECIALIZED_SPV.read_bytes()
    local_size = 32
    with vc.VulkanContext.open(device_index=device_index) as ctx:
        src = ctx.allocate(data.nbytes)
        dst = ctx.allocate(data.nbytes)
        src.write(data)
        with ctx.compute_pipeline(
            spirv,
            buffer_count=2,
            push_constant_bytes=4,
            specialization={0: local_size, 1: -3, 2: 2.5, 3: True},
        ) as pipeline:
            pipeline.dispatch(
                [src, dst],
                groups=(data.size + local_size - 1) // local_size,
                push_constants=count,
            )
            expected = -(data * 2.5 - 3.0)
            assert np.array_equal(dst.read(np.float32, data.shape), expected)

        # Unspecialized, the same SPIR-V keeps its defaults: local size 1,
        # FACTOR 1.0, OFFSET 0, NEGATE false, so the kernel is a copy.
        with ctx.compute_pipeline(
            spirv, buffer_count=2, push_constant_bytes=4
        ) as pipeline:
            pipeline.dispatch([src, dst], groups=data.size, push_constants=count)
            assert np.array_equal(dst.read(np.float32, data.shape), data)
