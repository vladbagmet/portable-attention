"""Tests for the ``vulkan`` backend: what it runs, what it hands to the CPU.

The host half — the support predicate, the reshape, the buffer and pipeline
caches, and what happens when a device fails mid-call — is exercised against a
fake context that computes the answer on the CPU, so it runs everywhere. The
real shader is then run through the same code path wherever a Vulkan compute
device exists, including the full conformance kit.
"""

from __future__ import annotations

import importlib
import struct
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from portable_attention import assert_conforms, available_backends, get_backend
from portable_attention.dispatch import _REGISTRY
from portable_attention.fused import scaled_dot_product_attention as fused_sdpa
from portable_attention.tiling import V3D_LIMITS
from portable_attention.vkattention import BUFFER_COUNT, PUSH_CONSTANT_BYTES
from portable_attention.vkbackend import (
    DISABLE_ENV_VAR,
    VULKAN_FLOOR_LIMITS,
    VulkanAttention,
    register_vulkan_backend,
    unsupported_reason,
)
from portable_attention.vkcompute import VulkanError
from portable_attention.vulkan import VulkanCapability, VulkanDevice, vulkan_available


@pytest.fixture(autouse=True)
def _restore_registry() -> Any:
    """Keep registrations made by a test out of the rest of the suite."""
    saved = dict(_REGISTRY)
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


def _inputs(
    stack: tuple[int, ...] = (2,),
    seq_q: int = 6,
    seq_k: int = 6,
    head_dim: int = 8,
    dtype: Any = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260814)
    shape = (*stack, seq_q, head_dim)
    q = rng.standard_normal(shape).astype(dtype)
    k = rng.standard_normal((*stack, seq_k, head_dim)).astype(dtype)
    v = rng.standard_normal((*stack, seq_k, head_dim)).astype(dtype)
    return q, k, v


# --------------------------------------------------------------------------
# what the kernel covers
# --------------------------------------------------------------------------


def _reason(*args: Any, **kwargs: Any) -> str | None:
    q, k, v = kwargs.pop("arrays", None) or _inputs()
    return unsupported_reason(
        q,
        k,
        v,
        kwargs.pop("attn_mask", None),
        kwargs.pop("dropout_p", 0.0),
        scale=kwargs.pop("scale", None),
        enable_gqa=kwargs.pop("enable_gqa", False),
    )


def test_plain_float32_attention_is_covered() -> None:
    """The shapes the shader was written for report no reason to fall back."""
    assert _reason() is None
    assert _reason(scale=0.25) is None


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"dropout_p": 0.1}, "dropout"),
        ({"attn_mask": np.ones((6, 6), dtype=bool)}, "attn_mask"),
        ({"enable_gqa": True}, "grouped-query"),
        ({"scale": float("inf")}, "not finite"),
    ],
)
def test_unsupported_options_name_themselves(kwargs: Any, expected: str) -> None:
    """Each unimplemented option is reported, not silently mishandled."""
    reason = _reason(**kwargs)
    assert reason is not None and expected in reason


@pytest.mark.parametrize("dtype", [np.float64, np.float16])
def test_other_dtypes_fall_back(dtype: Any) -> None:
    """The kernel's buffers are float32; anything else is a CPU call."""
    reason = _reason(arrays=_inputs(dtype=dtype))
    assert reason is not None and "float32" in reason


def test_rank_below_two_falls_back() -> None:
    """A vector has no (seq, head_dim) to dispatch over."""
    vec = np.zeros(4, dtype=np.float32)
    assert (
        reason := unsupported_reason(
            vec, vec, vec, None, 0.0, scale=None, enable_gqa=False
        )
    ) is not None
    assert "dims" in reason


def test_mismatched_leading_dims_fall_back() -> None:
    """The stack is flattened, not broadcast, so the leading axes must agree."""
    q, _, _ = _inputs(stack=(2, 3))
    _, k, v = _inputs(stack=(2, 1))
    reason = unsupported_reason(q, k, v, None, 0.0, scale=None, enable_gqa=False)
    assert reason is not None and "leading dimensions" in reason


def test_value_width_must_match_query() -> None:
    """The output buffer is query-shaped, so a narrower value is a CPU call."""
    q, k, _ = _inputs(head_dim=8)
    v = np.zeros((2, 6, 4), dtype=np.float32)
    reason = unsupported_reason(q, k, v, None, 0.0, scale=None, enable_gqa=False)
    assert reason is not None and "match query" in reason


def test_key_and_value_sequence_lengths_must_agree() -> None:
    """A key/value length mismatch is invalid input the CPU backend reports."""
    q, k, _ = _inputs(seq_k=6)
    v = np.zeros((2, 7, 8), dtype=np.float32)
    reason = unsupported_reason(q, k, v, None, 0.0, scale=None, enable_gqa=False)
    assert reason is not None and "same sequence length" in reason


def test_empty_shapes_fall_back() -> None:
    """A zero-length sequence or an empty stack has no dispatch geometry."""
    empty = np.zeros((0, 6, 8), dtype=np.float32)
    reason = unsupported_reason(
        empty, empty, empty, None, 0.0, scale=None, enable_gqa=False
    )
    assert reason is not None and "non-empty" in reason
    zero_seq = np.zeros((2, 0, 8), dtype=np.float32)
    assert (
        unsupported_reason(
            zero_seq, zero_seq, zero_seq, None, 0.0, scale=None, enable_gqa=False
        )
        is not None
    )


# --------------------------------------------------------------------------
# the host half, against a fake device
# --------------------------------------------------------------------------


class _FakeBuffer:
    """A mapped buffer stand-in: bytes with the same size rules as the real one."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes
        self.data = bytearray(nbytes)
        self.freed = False

    def write(self, array: np.ndarray) -> None:
        source = np.ascontiguousarray(array)
        assert not self.freed
        assert source.nbytes <= self.nbytes
        self.data[: source.nbytes] = source.tobytes()

    def read(self, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
        assert not self.freed
        wanted = int(np.prod(shape)) * np.dtype(dtype).itemsize
        assert wanted <= self.nbytes
        return np.frombuffer(bytes(self.data[:wanted]), dtype=dtype).reshape(shape)

    def free(self) -> None:
        self.freed = True


class _FakePipeline:
    """Computes what the shader would, from the bytes in the bound buffers."""

    def __init__(self, specialization: dict[int, int]) -> None:
        self.specialization = specialization
        self.dispatches: list[tuple[tuple[int, int, int], bytes]] = []

    def dispatch(
        self,
        buffers: list[_FakeBuffer],
        *,
        groups: tuple[int, int, int],
        push_constants: bytes,
    ) -> None:
        seq_q, seq_k, scale, causal = struct.unpack("=IIfI", push_constants)
        head_dim = self.specialization[4]
        q_buf, k_buf, v_buf, out_buf = buffers
        stack = groups[1]
        q = q_buf.read(np.float32, (stack, seq_q, head_dim))
        k = k_buf.read(np.float32, (stack, seq_k, head_dim))
        v = v_buf.read(np.float32, (stack, seq_k, head_dim))
        out = fused_sdpa(q, k, v, None, 0.0, bool(causal), scale=float(scale))
        out_buf.write(np.ascontiguousarray(out, dtype=np.float32))
        self.dispatches.append((groups, push_constants))


class _FakeContext:
    """Just enough of VulkanContext to drive the backend's device path."""

    def __init__(self) -> None:
        self.buffers: list[_FakeBuffer] = []
        self.pipelines: list[_FakePipeline] = []
        self.closed = False

    def allocate(self, nbytes: int) -> _FakeBuffer:
        buffer = _FakeBuffer(nbytes)
        self.buffers.append(buffer)
        return buffer

    def compute_pipeline(
        self,
        spirv: bytes,
        *,
        buffer_count: int,
        push_constant_bytes: int,
        specialization: dict[int, int],
    ) -> _FakePipeline:
        assert spirv[:4] == b"\x03\x02\x23\x07"
        assert buffer_count == BUFFER_COUNT
        assert push_constant_bytes == PUSH_CONSTANT_BYTES
        pipeline = _FakePipeline(dict(specialization))
        self.pipelines.append(pipeline)
        return pipeline

    def close(self) -> None:
        self.closed = True


def _fake_backend(context: _FakeContext | None = None) -> tuple[VulkanAttention, Any]:
    ctx = _FakeContext() if context is None else context
    backend = VulkanAttention(open_context=lambda *, device_index: ctx)  # type: ignore[arg-type]
    return backend, ctx


def test_device_path_returns_the_reference_answer() -> None:
    """The reshape, the buffers and the push constants add up to the right call."""
    backend, ctx = _fake_backend()
    q, k, v = _inputs(stack=(2, 3), seq_q=5, seq_k=7)

    got = backend(q, k, v, scale=0.3)

    np.testing.assert_allclose(
        got, get_backend("reference")(q, k, v, scale=0.3), rtol=1e-5, atol=1e-5
    )
    assert got.shape == q.shape and got.dtype == np.float32
    assert backend.device_calls == 1
    assert ctx.pipelines[0].dispatches[0][0][1] == 6  # one workgroup row per slice


def test_the_output_can_be_written_to() -> None:
    """A device read is backed by immutable bytes; the caller must not notice."""
    backend, _ = _fake_backend()
    got = backend(*_inputs())
    assert got.flags.writeable
    got += 1.0


def test_two_dimensional_inputs_dispatch_as_one_slice() -> None:
    """``(L, E)`` has no leading axes; it becomes a stack of one and comes back flat."""
    backend, _ = _fake_backend()
    q, k, v = _inputs(stack=(), seq_q=4, seq_k=4)

    got = backend(q, k, v, is_causal=True)

    assert got.shape == (4, 8)
    np.testing.assert_allclose(
        got, get_backend("reference")(q, k, v, is_causal=True), rtol=1e-5, atol=1e-5
    )


def test_pipelines_are_cached_per_tile_shape() -> None:
    """A repeated shape reuses its pipeline; a new head dim compiles another."""
    backend, ctx = _fake_backend()
    q, k, v = _inputs(head_dim=8)
    backend(q, k, v)
    backend(q, k, v, is_causal=True)
    assert backend.live_pipelines == 1
    assert len(ctx.pipelines) == 1

    wide = _inputs(head_dim=32)
    backend(*wide)
    assert backend.live_pipelines == 2


def test_buffers_grow_and_are_then_reused() -> None:
    """Bigger inputs reallocate and free the old buffer; smaller ones reuse it."""
    backend, ctx = _fake_backend()
    backend(*_inputs(seq_q=4, seq_k=4))
    assert len(ctx.buffers) == BUFFER_COUNT

    backend(*_inputs(seq_q=16, seq_k=16))
    assert len(ctx.buffers) == 2 * BUFFER_COUNT
    assert [b.freed for b in ctx.buffers[:BUFFER_COUNT]] == [True] * BUFFER_COUNT

    backend(*_inputs(seq_q=8, seq_k=8))
    assert len(ctx.buffers) == 2 * BUFFER_COUNT
    assert backend.device_calls == 3


def test_close_releases_the_context_and_reopens_on_demand() -> None:
    """Closing drops the caches; the next call opens a fresh context."""
    contexts: list[_FakeContext] = []

    def opener(*, device_index: int | None) -> Any:
        contexts.append(_FakeContext())
        return contexts[-1]

    backend = VulkanAttention(open_context=opener)  # type: ignore[arg-type]
    backend(*_inputs())
    backend.close()
    backend.close()
    assert contexts[0].closed and backend.live_pipelines == 0

    backend(*_inputs())
    assert len(contexts) == 2


# --------------------------------------------------------------------------
# falling back
# --------------------------------------------------------------------------


def test_masked_and_gqa_calls_go_to_the_cpu() -> None:
    """The device is never opened for a call the kernel does not implement."""
    backend, ctx = _fake_backend()
    q, k, v = _inputs(stack=(2, 4))
    mask = np.zeros((6, 6), dtype=bool)
    mask[:, :3] = True

    got = backend(q, k, v, mask)

    np.testing.assert_allclose(
        got, get_backend("reference")(q, k, v, mask), rtol=1e-5, atol=1e-5
    )
    assert backend.device_calls == 0
    assert ctx.buffers == [] and ctx.pipelines == []


def test_invalid_input_still_raises_the_documented_error() -> None:
    """Validation belongs to the CPU backend, so its errors survive the detour."""
    backend, _ = _fake_backend()
    q, k, v = _inputs()
    with pytest.raises(ValueError, match="not both"):
        backend(q, k, v, np.ones((6, 6), dtype=bool), is_causal=True)
    with pytest.raises(NotImplementedError):
        backend(q, k, v, None, 0.5)


def test_a_head_dim_no_plan_fits_falls_back() -> None:
    """When no tile shape fits the device budget, the CPU answers."""
    backend, ctx = _fake_backend()
    q, k, v = _inputs(seq_q=2, seq_k=2, head_dim=4096)

    got = backend(q, k, v)

    np.testing.assert_allclose(got, fused_sdpa(q, k, v), rtol=1e-5, atol=1e-5)
    assert backend.device_calls == 0 and ctx.pipelines == []


def test_a_device_failure_retires_the_device_once() -> None:
    """A broken device warns once, then the process is served from the CPU."""

    def opener(*, device_index: int | None) -> Any:
        raise VulkanError("device lost")

    backend = VulkanAttention(open_context=opener)  # type: ignore[arg-type]
    q, k, v = _inputs()

    with pytest.warns(RuntimeWarning, match="device error"):
        first = backend(q, k, v)
    np.testing.assert_allclose(first, fused_sdpa(q, k, v), rtol=1e-5, atol=1e-5)
    assert not backend.device_usable

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        second = backend(q, k, v)
    np.testing.assert_allclose(second, first, rtol=0, atol=0)
    assert backend.device_calls == 0


def test_concurrent_calls_share_one_context() -> None:
    """The lock covers opening the device, growing buffers and the counter."""
    contexts: list[_FakeContext] = []

    def opener(*, device_index: int | None) -> Any:
        contexts.append(_FakeContext())
        return contexts[-1]

    backend = VulkanAttention(open_context=opener)  # type: ignore[arg-type]
    # Varying lengths make the threads contend on buffer growth.
    cases = [_inputs(seq_q=n, seq_k=n) for n in (4, 8, 12, 16)] * 4
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda qkv: backend(*qkv), cases))

    assert len(contexts) == 1
    assert backend.device_calls == len(cases)
    for (q, k, v), got in zip(cases, results):
        np.testing.assert_allclose(got, fused_sdpa(q, k, v), rtol=1e-5, atol=1e-5)


def test_retiring_a_device_twice_warns_once() -> None:
    """Retirement is idempotent, so a racing second failure stays quiet."""
    backend, ctx = _fake_backend()
    backend(*_inputs())

    with pytest.warns(RuntimeWarning, match="device error"):
        backend._retire_device(VulkanError("device lost"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend._retire_device(VulkanError("device lost again"))
    assert ctx.closed


def test_a_concurrent_device_failure_warns_once() -> None:
    """Threads racing a dying device retire it together, not repeatedly."""

    def opener(*, device_index: int | None) -> Any:
        raise VulkanError("device lost")

    backend = VulkanAttention(open_context=opener)  # type: ignore[arg-type]
    cases = [_inputs() for _ in range(8)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda qkv: backend(*qkv), cases))

    assert len(caught) == 1
    assert not backend.device_usable
    for (q, k, v), got in zip(cases, results):
        np.testing.assert_allclose(got, fused_sdpa(q, k, v), rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


_AVAILABLE = VulkanCapability(
    available=True,
    loader="libvulkan.so.1",
    devices=(VulkanDevice(name="fake", api_version="1.2.0", compute=True),),
)


def test_registers_when_a_compute_device_is_reported() -> None:
    """With a device present the name resolves through the public registry."""
    assert register_vulkan_backend(capability=_AVAILABLE, overwrite=True, environ={})
    assert "vulkan" in available_backends()
    assert isinstance(get_backend("vulkan"), VulkanAttention)


def test_does_not_register_without_a_device() -> None:
    """A CPU-only host keeps the registry it had."""
    _REGISTRY.pop("vulkan", None)
    unavailable = VulkanCapability(available=False, loader=None, reason="no loader")
    assert not register_vulkan_backend(capability=unavailable, environ={})
    assert "vulkan" not in available_backends()


def test_the_opt_out_wins_over_a_present_device() -> None:
    """The environment variable is read before anything is detected or built."""
    _REGISTRY.pop("vulkan", None)
    assert not register_vulkan_backend(
        capability=_AVAILABLE, environ={DISABLE_ENV_VAR: "1"}
    )
    assert "vulkan" not in available_backends()


def test_registration_survives_a_reload_of_the_package() -> None:
    """The registry outlives the module, so the import-time call overwrites."""
    register_vulkan_backend(capability=_AVAILABLE, overwrite=True, environ={})
    importlib.reload(importlib.import_module("portable_attention"))
    assert "vulkan" in available_backends()


def test_a_detection_failure_leaves_the_backend_unregistered() -> None:
    """An import must not raise because probing a foreign loader went wrong."""
    _REGISTRY.pop("vulkan", None)

    def broken() -> VulkanCapability:
        raise RuntimeError("the loader segfaulted politely")

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr("portable_attention.vkbackend.detect_vulkan", broken)
        assert not register_vulkan_backend(environ={})
    finally:
        monkey.undo()
    assert "vulkan" not in available_backends()


def test_auto_does_not_route_to_the_device() -> None:
    """Selecting the device is the caller's decision until benchmarks say more."""
    register_vulkan_backend(capability=_AVAILABLE, overwrite=True, environ={})
    from portable_attention.dispatch import _auto_select

    q, _, _ = _inputs(stack=(2, 4))
    assert not isinstance(_auto_select(q), VulkanAttention)


def test_the_default_limits_are_the_vulkan_minimums() -> None:
    """Plans must fit any conformant device until limits can be queried."""
    backend = VulkanAttention()
    assert backend.limits is VULKAN_FLOOR_LIMITS
    assert VULKAN_FLOOR_LIMITS.shared_memory_bytes <= V3D_LIMITS.shared_memory_bytes
    assert VULKAN_FLOOR_LIMITS.max_threads_per_group <= V3D_LIMITS.max_threads_per_group


# --------------------------------------------------------------------------
# real hardware (skipped where no Vulkan device installed)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not vulkan_available(), reason="no Vulkan compute device")
def test_the_backend_is_registered_on_this_host() -> None:
    """Importing the package was enough to make the device reachable."""
    assert "vulkan" in available_backends()


@pytest.mark.skipif(not vulkan_available(), reason="no Vulkan compute device")
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    ("stack", "seq_q", "seq_k", "head_dim"),
    [((), 4, 4, 8), ((3,), 9, 5, 16), ((2, 2), 33, 33, 64), ((1,), 12, 12, 128)],
)
def test_device_output_matches_the_oracle(
    stack: tuple[int, ...], seq_q: int, seq_k: int, head_dim: int, is_causal: bool
) -> None:
    """The registered backend agrees with the reference on real hardware."""
    backend = VulkanAttention()
    q, k, v = _inputs(stack=stack, seq_q=seq_q, seq_k=seq_k, head_dim=head_dim)
    try:
        got = backend(q, k, v, is_causal=is_causal)
        assert backend.device_calls == 1
    finally:
        backend.close()
    oracle = get_backend("reference")(q, k, v, is_causal=is_causal)
    np.testing.assert_allclose(got, oracle, rtol=1e-5, atol=1e-5)
    assert got.flags.writeable


@pytest.mark.skipif(not vulkan_available(), reason="no Vulkan compute device")
def test_device_backend_passes_the_conformance_kit() -> None:
    """The full contract matrix, device path and CPU fallback together."""
    backend = VulkanAttention()
    try:
        assert_conforms(backend)
        assert backend.device_calls > 0
    finally:
        backend.close()
