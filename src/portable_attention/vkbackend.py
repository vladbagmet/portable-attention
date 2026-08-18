"""The ``vulkan`` backend: the attention shader behind the SDPA contract.

``shaders/attention.comp`` computes unmasked or causal float32 attention over a
flat ``(n, seq, head_dim)`` stack. The public contract is wider than that — it
has masks, grouped-query attention, four dtypes and a value width that may
differ from the key width — so this module is the adapter between the two:

* it decides, per call, whether the kernel covers the request (see
  :func:`unsupported_reason`) and hands everything else to the CPU ``fused``
  backend, so the backend is conformant, not merely fast on the easy cases;
* it reshapes ``(*, H, L, E)`` down to the stack the kernel dispatches over and
  restores the leading axes on the way out;
* it owns the device: one :class:`~portable_attention.vkcompute.VulkanContext`
  opened on first use, one pipeline per tile shape, and four buffers that grow
  as shapes demand rather than being reallocated per call.

The backend registers under ``"vulkan"`` only where a compute-capable device
exists (:func:`register_vulkan_backend`), so importing the package on a CPU-only
host leaves the registry exactly as it was. It is not part of ``"auto"``: until
there are benchmark numbers saying when the device wins, selecting it is the
caller's decision.

Tiles are sized against the limits the opened device reports
(``VkPhysicalDeviceLimits``), so the kernel uses what the hardware has rather
than the minima the specification guarantees everywhere. Those minima remain as
:data:`VULKAN_FLOOR_LIMITS`, which is what :attr:`VulkanAttention.limits`
answers before a device has been opened.
"""

from __future__ import annotations

import contextlib
import os
import threading
import warnings
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from .dispatch import SdpaBackend, register_backend
from .fused import scaled_dot_product_attention as _fused_sdpa
from .tiling import DeviceLimits, TileSizingError, plan_tiles
from .vkattention import (
    BUFFER_COUNT,
    KERNEL_DTYPE_BYTES,
    PUSH_CONSTANT_BYTES,
    AttentionLaunch,
    attention_launch,
    attention_spirv,
)
from .vkcompute import VulkanBuffer, VulkanContext, VulkanError, VulkanPipeline
from .vulkan import VulkanCapability, detect_vulkan

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "DISABLE_ENV_VAR",
    "VULKAN_FLOOR_LIMITS",
    "VulkanAttention",
    "register_vulkan_backend",
    "unsupported_reason",
]

Array = NDArray[np.floating]

#: Setting this environment variable to a non-empty value keeps the backend
#: from registering, which also skips the device probe an import would run.
DISABLE_ENV_VAR = "PORTABLE_ATTENTION_DISABLE_VULKAN"

#: The tile-sizing limits every Vulkan implementation is required to meet:
#: 16 KiB of workgroup-shared memory and 128 invocations per workgroup. A plan
#: that fits these runs anywhere. It is what the backend reports before it has
#: opened a device, and what a caller can pass as an override to hold a kernel
#: to portable tile shapes. ``simd_width`` only orders the policy's
#: preferences, and 16 is the narrowest width the targeted devices have.
VULKAN_FLOOR_LIMITS = DeviceLimits(
    shared_memory_bytes=16384,
    max_threads_per_group=128,
    simd_width=16,
    name="Vulkan guaranteed minimums",
)

#: Element type the kernel reads and writes. Other dtypes take the CPU path.
KERNEL_DTYPE = np.float32

#: Buffer bindings, in the order the shader declares them.
_QUERY, _KEY, _VALUE, _OUTPUT = range(BUFFER_COUNT)


class _OpenContext(Protocol):
    """How a device is opened; injectable so the host half can be tested."""

    def __call__(self, *, device_index: int | None) -> VulkanContext: ...


def unsupported_reason(
    query: Array,
    key: Array,
    value: Array,
    attn_mask: NDArray[np.floating] | NDArray[np.bool_] | None,
    dropout_p: float,
    *,
    scale: float | None,
    enable_gqa: bool,
) -> str | None:
    """Return why the kernel cannot serve this call, or ``None`` if it can.

    The list is written from what ``attention.comp`` actually implements: one
    float32 stack of ``(seq, head_dim)`` slices, the same head dimension for
    query, key and value, no mask beyond the causal flag, no dropout, no
    grouped-query expansion. Anything else is not an error — it is a call the
    CPU backend takes — so the reason is returned rather than raised.

    Invalid inputs (mismatched shapes, ``is_causal`` together with a mask) also
    land here as "unsupported", which sends them to the CPU backend and lets it
    raise the error message the contract documents. There is exactly one place
    that decides what a bad shape means.
    """
    if dropout_p != 0.0:
        return "dropout is not implemented by the kernel"
    if attn_mask is not None:
        return "attn_mask is not implemented by the kernel"
    if enable_gqa:
        return "grouped-query attention is not implemented by the kernel"
    if scale is not None and not np.isfinite(scale):
        return f"scale {scale!r} is not finite"
    for name, array in (("query", query), ("key", key), ("value", value)):
        if array.dtype != KERNEL_DTYPE:
            return f"{name} is {array.dtype}, and the kernel is float32"
        if array.ndim < 2:
            return f"{name} has {array.ndim} dims; the kernel needs (*, seq, head_dim)"
    if query.shape[:-2] != key.shape[:-2] or query.shape[:-2] != value.shape[:-2]:
        return "query, key and value must share their leading dimensions"
    head_dim = query.shape[-1]
    if key.shape[-1] != head_dim or value.shape[-1] != head_dim:
        return (
            f"the kernel writes a {head_dim}-wide output, so key "
            f"({key.shape[-1]}) and value ({value.shape[-1]}) must match query"
        )
    if key.shape[-2] != value.shape[-2]:
        return "key and value must have the same sequence length"
    if min(query.shape[-2], key.shape[-2], head_dim) == 0 or query.size == 0:
        return "the kernel needs a non-empty stack and non-empty sequences"
    return None


class VulkanAttention:
    """SDPA backend that runs the attention shader, falling back to CPU.

    One instance owns one device. Calls the kernel covers are dispatched to it;
    everything else is forwarded unchanged to the ``fused`` CPU backend, which
    also owns the error messages for invalid input. If the device fails at
    runtime the instance stops using it (once, with a warning) and serves the
    rest of the process from the CPU — a wedged GPU degrades throughput, not
    correctness.

    Instances are usable from several threads: the device is serialised by a
    lock, since one context, one pipeline per shape and one set of buffers are
    shared between calls.
    """

    def __init__(
        self,
        *,
        limits: DeviceLimits | None = None,
        device_index: int | None = None,
        open_context: _OpenContext = VulkanContext.open,
        fallback: SdpaBackend | None = None,
    ) -> None:
        """Build a backend; the device is opened lazily on the first call.

        Args:
            limits: Device limits tile sizing plans against, overriding what
                the device reports. ``None`` (the default) asks the device
                once it is open, and reports :data:`VULKAN_FLOOR_LIMITS` until
                then.
            device_index: Index into the loader's device enumeration, or
                ``None`` for the first compute-capable device.
            open_context: How to open the device. Injectable for tests.
            fallback: Backend for calls the kernel does not cover. Defaults to
                the ``fused`` CPU backend.
        """
        self._limits_override = limits
        self._limits = VULKAN_FLOOR_LIMITS if limits is None else limits
        self._device_index = device_index
        self._open_context = open_context
        self._fallback: SdpaBackend = _fused_sdpa if fallback is None else fallback
        self._lock = threading.Lock()
        self._context: VulkanContext | None = None
        self._pipelines: dict[tuple[tuple[int, int], ...], VulkanPipeline] = {}
        self._buffers: list[VulkanBuffer | None] = [None] * BUFFER_COUNT
        self._device_usable = True
        self._device_calls = 0

    @property
    def limits(self) -> DeviceLimits:
        """The device limits tile plans are sized against.

        The Vulkan minimums until the device has been opened, then whatever it
        reports — unless the caller passed limits of their own, which always
        win.
        """
        return self._limits

    @property
    def device_calls(self) -> int:
        """How many calls this backend has served from the device."""
        return self._device_calls

    @property
    def device_usable(self) -> bool:
        """``False`` once a device failure has moved the backend to the CPU."""
        return self._device_usable

    @property
    def live_pipelines(self) -> int:
        """Distinct tile shapes compiled so far (the pipeline cache size)."""
        return len(self._pipelines)

    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        attn_mask: NDArray[np.floating] | NDArray[np.bool_] | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> Array:
        """Compute attention, on the device where the kernel covers the call.

        Args and returns follow
        :func:`portable_attention.scaled_dot_product_attention` exactly; the
        output is float32 with the leading axes of ``query``.
        """
        if (
            unsupported_reason(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                scale=scale,
                enable_gqa=enable_gqa,
            )
            is None
        ):
            out = self._run(query, key, value, is_causal=is_causal, scale=scale)
            if out is not None:
                return out
        result: Array = self._fallback(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        return result

    def close(self) -> None:
        """Release the device, its pipelines and its buffers. Idempotent."""
        with self._lock:
            self._close_held()

    def _close_held(self) -> None:
        """Release the device; the caller already holds the lock."""
        context, self._context = self._context, None
        if self._limits_override is None:
            self._limits = VULKAN_FLOOR_LIMITS
        self._pipelines.clear()
        self._buffers = [None] * BUFFER_COUNT
        if context is not None:
            context.close()

    def _run(
        self,
        query: Array,
        key: Array,
        value: Array,
        *,
        is_causal: bool,
        scale: float | None,
    ) -> Array | None:
        """Dispatch one call, or return ``None`` to mean "use the fallback"."""
        head_dim = int(query.shape[-1])
        seq_q, seq_k = int(query.shape[-2]), int(key.shape[-2])
        stack = int(np.prod(query.shape[:-2])) if query.ndim > 2 else 1
        flat_q = _as_stack(query, stack)
        flat_k = _as_stack(key, stack)
        flat_v = _as_stack(value, stack)
        try:
            with self._lock:
                if not self._device_usable:
                    # A device that failed once is not tried again, and the
                    # flag is only ever read here, under the lock.
                    return None
                # Whether a shape fits is a question about the device, so the
                # device is opened before the plan is made, not after.
                context = self._context_held()
                try:
                    launch = self._plan(
                        head_dim,
                        stack=stack,
                        seq_q=seq_q,
                        seq_k=seq_k,
                        scale=scale,
                        is_causal=is_causal,
                    )
                except (TileSizingError, ValueError):
                    # No tile shape fits this head dimension on this device, or
                    # the dispatch would be wider than Vulkan guarantees. Both
                    # are shape facts, not faults.
                    return None
                flat_out = self._dispatch(
                    context,
                    flat_q,
                    flat_k,
                    flat_v,
                    specialization=launch.specialization,
                    push_constants=launch.push_constants,
                    groups=launch.groups,
                )
                self._device_calls += 1
        except VulkanError as exc:
            self._retire_device(exc)
            return None
        return flat_out.reshape(*query.shape[:-2], seq_q, head_dim)

    def _plan(
        self,
        head_dim: int,
        *,
        stack: int,
        seq_q: int,
        seq_k: int,
        scale: float | None,
        is_causal: bool,
    ) -> AttentionLaunch:
        """Size the tiles for this shape against the current device limits."""
        plan = plan_tiles(
            head_dim,
            KERNEL_DTYPE_BYTES,
            self._limits,
            seq_len_q=seq_q,
            seq_len_k=seq_k,
        )
        return attention_launch(
            plan,
            stack=stack,
            seq_q=seq_q,
            seq_k=seq_k,
            scale=float(1.0 / np.sqrt(head_dim)) if scale is None else scale,
            is_causal=is_causal,
        )

    def _context_held(self) -> VulkanContext:
        """Return the open device, opening it on first use; lock held.

        Opening is also where the tile-sizing limits stop being the Vulkan
        minimums and become the ones this device reports, unless the caller
        supplied limits of their own.
        """
        context = self._context
        if context is None:
            context = self._open_context(device_index=self._device_index)
            self._context = context
            if self._limits_override is None:
                self._limits = context.tile_limits
        return context

    def _dispatch(
        self,
        context: VulkanContext,
        flat_q: NDArray[np.float32],
        flat_k: NDArray[np.float32],
        flat_v: NDArray[np.float32],
        *,
        specialization: Mapping[int, int],
        push_constants: bytes,
        groups: tuple[int, int, int],
    ) -> NDArray[np.float32]:
        """Copy in, run the pipeline for this tile shape, copy the output out."""
        for binding, array in (
            (_QUERY, flat_q),
            (_KEY, flat_k),
            (_VALUE, flat_v),
            (_OUTPUT, flat_q),
        ):
            self._buffer(context, binding, array.nbytes)
        for binding, array in ((_QUERY, flat_q), (_KEY, flat_k), (_VALUE, flat_v)):
            _live(self._buffers[binding]).write(array)

        pipeline = self._pipeline(context, specialization)
        pipeline.dispatch(
            [_live(buffer) for buffer in self._buffers],
            groups=groups,
            push_constants=push_constants,
        )
        # The buffer read is backed by immutable bytes, so it comes back
        # read-only; every other backend returns an array the caller may write
        # to, and one copy of the output is cheap next to the attention itself.
        out: NDArray[np.float32] = np.array(
            _live(self._buffers[_OUTPUT]).read(KERNEL_DTYPE, flat_q.shape),
            dtype=KERNEL_DTYPE,
        )
        return out

    def _buffer(self, context: VulkanContext, binding: int, nbytes: int) -> None:
        """Make sure binding ``binding`` has a buffer of at least ``nbytes``.

        Buffers only ever grow: a smaller call reuses the allocation and reads
        back the prefix it wrote, so a steady-state workload allocates once.
        """
        current = self._buffers[binding]
        if current is not None and current.nbytes >= nbytes:
            return
        self._buffers[binding] = context.allocate(nbytes)
        if current is not None:
            current.free()

    def _pipeline(
        self, context: VulkanContext, specialization: Mapping[int, int]
    ) -> VulkanPipeline:
        """Return the pipeline for this tile shape, compiling it once."""
        key = tuple(sorted(specialization.items()))
        pipeline = self._pipelines.get(key)
        if pipeline is None:
            pipeline = context.compute_pipeline(
                attention_spirv(),
                buffer_count=BUFFER_COUNT,
                push_constant_bytes=PUSH_CONSTANT_BYTES,
                specialization=specialization,
            )
            self._pipelines[key] = pipeline
        return pipeline

    def _retire_device(self, exc: VulkanError) -> None:
        """Stop using the device after a failure and say so once.

        Retirement, teardown and the warning are one atomic step, so a second
        thread that fails on the same device neither reopens it nor repeats the
        warning.
        """
        with self._lock:
            first = self._device_usable
            self._device_usable = False
            # Teardown of a device that just failed can fail in turn; the CPU
            # path is already the answer, so nothing here is worth raising.
            with contextlib.suppress(VulkanError):
                self._close_held()
        if first:
            warnings.warn(
                f"the Vulkan attention backend hit a device error ({exc}); "
                "the rest of this process is served by the CPU backend",
                RuntimeWarning,
                stacklevel=4,
            )


def _as_stack(array: Array, stack: int) -> NDArray[np.float32]:
    """View ``(*, seq, head_dim)`` as the contiguous ``(n, seq, head_dim)`` stack."""
    flat = np.ascontiguousarray(array).reshape(stack, array.shape[-2], array.shape[-1])
    return flat.astype(KERNEL_DTYPE, copy=False)


def _live(buffer: VulkanBuffer | None) -> VulkanBuffer:
    """Assert a binding has been allocated (it always has, by construction)."""
    if buffer is None:  # pragma: no cover - _dispatch allocates all four first
        raise VulkanError("attention buffer was not allocated")
    return buffer


def register_vulkan_backend(
    *,
    overwrite: bool = False,
    capability: VulkanCapability | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Register the ``"vulkan"`` backend if this host can run it.

    Args:
        overwrite: Replace an existing ``"vulkan"`` registration.
        capability: Detection result to trust; ``None`` probes the host.
        environ: Environment to read :data:`DISABLE_ENV_VAR` from; ``None``
            reads the process environment.

    Returns:
        ``True`` when the backend was registered, ``False`` when the host has
        no compute-capable Vulkan device or the opt-out is set.
    """
    env = os.environ if environ is None else environ
    if env.get(DISABLE_ENV_VAR):
        return False
    if capability is None:
        try:
            capability = detect_vulkan()
        except Exception:  # noqa: BLE001 - importing must never fail on a probe
            # Detection walks a foreign loader through ctypes. It handles the
            # failures it knows about, and an unknown one still has an obvious
            # answer here: this host does not get the backend.
            return False
    if not capability.available:
        return False
    register_backend("vulkan", VulkanAttention(), overwrite=overwrite)
    return True
