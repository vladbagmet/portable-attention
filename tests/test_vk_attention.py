"""Tests for the Vulkan attention shader and the launch geometry it needs.

The pure half (specialization values, push constants, workgroup counts) runs
everywhere. The shader itself is dispatched only where a Vulkan compute device
exists, and is checked against ``blocked_attention`` — the NumPy model it was
transcribed from — before the reference oracle, so a mismatch says which of the
two disagrees.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np
import pytest

from portable_attention import get_backend
from portable_attention import vkcompute as vc
from portable_attention.blocked import blocked_attention
from portable_attention.tiling import V3D_LIMITS, TilePlan, plan_tiles
from portable_attention.vkattention import (
    BUFFER_COUNT,
    MAX_GROUPS_PER_AXIS,
    PUSH_CONSTANT_BYTES,
    attention_launch,
    attention_spirv,
)
from portable_attention.vulkan import detect_vulkan, vulkan_available

_SHADER_SOURCE = (Path(vc.__file__).parent / "shaders" / "attention.comp").read_text()

_SPIRV_MAGIC = b"\x03\x02\x23\x07"


def _plan(head_dim: int = 64, dtype_bytes: int = 4) -> TilePlan:
    return plan_tiles(head_dim, dtype_bytes, V3D_LIMITS)


# --------------------------------------------------------------------------
# the compiled module
# --------------------------------------------------------------------------


def test_spirv_is_a_module_and_is_read_once() -> None:
    """The committed .spv loads, looks like SPIR-V, and is cached."""
    spirv = attention_spirv()
    assert spirv[:4] == _SPIRV_MAGIC
    assert len(spirv) % 4 == 0
    assert attention_spirv() is spirv


def test_launch_supplies_every_constant_the_shader_declares() -> None:
    """No ``constant_id`` may fall back to its placeholder default.

    The defaults in the GLSL are 1s, which compile but compute nothing useful,
    so an id the host forgets to fill is a silent wrong answer rather than an
    error. This pins the two lists together.
    """
    ids = re.findall(r"constant_id = (\d+)", _SHADER_SOURCE)
    ids += re.findall(r"local_size_[xyz]_id = (\d+)", _SHADER_SOURCE)
    declared = {int(value) for value in ids}
    launch = attention_launch(_plan(), stack=1, seq_q=8, seq_k=8, scale=0.125)
    assert set(launch.specialization) == declared


def test_shared_array_lengths_match_the_tile_budget() -> None:
    """The lengths handed to the driver add up to the bytes the planner priced."""
    plan = _plan()
    spec = attention_launch(plan, stack=1, seq_q=8, seq_k=8, scale=0.5).specialization
    elements = spec[5] + 2 * spec[6] + spec[7] + 2 * plan.block_q
    assert elements * plan.dtype_bytes == plan.shared_memory_bytes


# --------------------------------------------------------------------------
# launch geometry (pure)
# --------------------------------------------------------------------------


def test_specialization_describes_the_plan() -> None:
    """Workgroup size, tile shape and array lengths all come from the plan."""
    plan = _plan(head_dim=32)
    launch = attention_launch(plan, stack=2, seq_q=10, seq_k=10, scale=0.25)
    spec = launch.specialization
    assert (spec[0], spec[1]) == (plan.block_k, plan.block_q)
    assert (spec[2], spec[3], spec[4]) == (plan.block_q, plan.block_k, plan.head_dim)
    assert spec[5] == plan.block_q * plan.head_dim
    assert spec[6] == plan.block_k * plan.head_dim
    assert spec[7] == plan.block_q * plan.block_k
    assert spec[8] == plan.accumulators_per_invocation
    assert launch.invocations_per_group == plan.threads_per_group
    assert launch.plan is plan


def test_groups_cover_the_query_rows_of_every_slice() -> None:
    """One workgroup per query tile per stack entry, and a partial tile counts."""
    plan = _plan(head_dim=16)
    launch = attention_launch(plan, stack=3, seq_q=plan.block_q + 1, seq_k=5, scale=1.0)
    assert launch.groups == (2, 3, 1)


@pytest.mark.parametrize("is_causal", [False, True])
def test_push_constants_round_trip(is_causal: bool) -> None:
    """The block is the shapes, the scale and the causal flag, in that order."""
    launch = attention_launch(
        _plan(), stack=1, seq_q=7, seq_k=13, scale=0.125, is_causal=is_causal
    )
    assert len(launch.push_constants) == PUSH_CONSTANT_BYTES
    assert struct.unpack("=IIfI", launch.push_constants) == (
        7,
        13,
        0.125,
        int(is_causal),
    )


def test_push_constants_are_packed_in_host_order() -> None:
    """The driver copies the block and the shader reads it as host layout.

    Packing it little-endian would be a wire format, which this is not; it
    would differ from what the device sees on a big-endian host.
    """
    native = struct.pack("=IIfI", 1, 1, 1.0, 0)
    assert (
        attention_launch(_plan(), stack=1, seq_q=1, seq_k=1, scale=1.0).push_constants
        == native
    )


def test_buffer_count_matches_the_shader_bindings() -> None:
    """Q, K, V and the output, one binding each."""
    bindings = re.findall(r"binding = (\d+)", _SHADER_SOURCE)
    assert sorted(int(value) for value in bindings) == list(range(BUFFER_COUNT))


# --------------------------------------------------------------------------
# rejected launches
# --------------------------------------------------------------------------


def test_rejects_a_plan_that_is_not_float32() -> None:
    with pytest.raises(ValueError, match="float32"):
        attention_launch(_plan(dtype_bytes=8), stack=1, seq_q=4, seq_k=4, scale=1.0)


@pytest.mark.parametrize("field", ["stack", "seq_q", "seq_k"])
@pytest.mark.parametrize("bad", [0, -1, 2.0, True, "8"])
def test_rejects_counts_that_are_not_positive_ints(field: str, bad: object) -> None:
    shapes = {"stack": 1, "seq_q": 4, "seq_k": 4}
    shapes[field] = bad  # type: ignore[assignment]
    with pytest.raises(ValueError, match=field):
        attention_launch(_plan(), scale=1.0, **shapes)  # type: ignore[arg-type]


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), float("-inf")])
def test_rejects_a_scale_that_is_not_finite(scale: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        attention_launch(_plan(), stack=1, seq_q=4, seq_k=4, scale=scale)


@pytest.mark.parametrize("axis", [0, 1])
def test_rejects_a_shape_needing_more_workgroups_than_vulkan_guarantees(
    axis: int,
) -> None:
    """An axis over 65535 workgroups is refused while the shape is still known.

    ``dispatch()`` would catch it too, but by then the message is about
    workgroup counts rather than about the sequence or batch that produced them.
    """
    plan = _plan()
    shapes = {"stack": 1, "seq_q": 8, "seq_k": 8}
    if axis == 0:
        shapes["seq_q"] = (MAX_GROUPS_PER_AXIS + 1) * plan.block_q
    else:
        shapes["stack"] = MAX_GROUPS_PER_AXIS + 1
    with pytest.raises(ValueError, match=f"axis {axis}"):
        attention_launch(plan, scale=1.0, **shapes)


def test_the_largest_supported_dispatch_is_accepted() -> None:
    """The limit itself is fine; only the row above it is refused."""
    plan = _plan()
    launch = attention_launch(
        plan,
        stack=MAX_GROUPS_PER_AXIS,
        seq_q=MAX_GROUPS_PER_AXIS * plan.block_q,
        seq_k=8,
        scale=1.0,
    )
    assert launch.groups == (MAX_GROUPS_PER_AXIS, MAX_GROUPS_PER_AXIS, 1)


def test_rejects_a_scale_the_float32_slot_cannot_hold() -> None:
    """A binary64 value too large for the slot fails here, not inside struct."""
    with pytest.raises(ValueError, match="out of range"):
        attention_launch(_plan(), stack=1, seq_q=4, seq_k=4, scale=3.5e38)


# --------------------------------------------------------------------------
# real hardware (skipped where no Vulkan device installed)
# --------------------------------------------------------------------------


def _compute_device_indices() -> list[int]:
    """Indices of every enumerated device that can run compute, or ``[]``."""
    capability = detect_vulkan()
    return [index for index, device in enumerate(capability.devices) if device.compute]


# (stack, seq_q, seq_k, head_dim). Between them these cover a single-element
# problem, partial query and key tiles, unequal sequence lengths, a head dim
# that needs several accumulators per invocation, and a plan whose workgroup
# is smaller than the device maximum. head_dim 6 and 13 are not multiples of
# four, which is what selects the shader's scalar dot product over its
# vectorized one.
_SHAPES = [
    (1, 1, 1, 8),
    (2, 6, 10, 8),
    (3, 7, 7, 16),
    (2, 17, 5, 32),
    (1, 33, 33, 64),
    (1, 12, 12, 128),
    (2, 9, 6, 6),
    (1, 5, 20, 13),
]


def _dispatch(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, *, plan, scale, is_causal, device_index
) -> np.ndarray:
    """Run the shader once on the named device and read the output back."""
    launch = attention_launch(
        plan,
        stack=q.shape[0],
        seq_q=q.shape[1],
        seq_k=k.shape[1],
        scale=scale,
        is_causal=is_causal,
    )
    with vc.VulkanContext.open(device_index=device_index) as ctx:
        buffers = [ctx.allocate(array.nbytes) for array in (q, k, v)]
        for buffer, array in zip(buffers, (q, k, v)):
            buffer.write(array)
        out = ctx.allocate(q.nbytes)
        pipeline = ctx.compute_pipeline(
            attention_spirv(),
            buffer_count=BUFFER_COUNT,
            push_constant_bytes=PUSH_CONSTANT_BYTES,
            specialization=launch.specialization,
        )
        pipeline.dispatch(
            [*buffers, out],
            groups=launch.groups,
            push_constants=launch.push_constants,
        )
        return out.read(np.float32, q.shape)


@pytest.mark.skipif(not vulkan_available(), reason="no Vulkan compute device")
@pytest.mark.parametrize("device_index", _compute_device_indices())
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(("stack", "seq_q", "seq_k", "head_dim"), _SHAPES)
def test_shader_matches_the_model_and_the_oracle(
    device_index: int,
    is_causal: bool,
    stack: int,
    seq_q: int,
    seq_k: int,
    head_dim: int,
) -> None:
    """The kernel agrees with blocked_attention, which agrees with the oracle.

    Both comparisons are made against the same inputs and the same tile plan,
    so a failure against the model is a transcription bug in the shader while a
    failure only against the oracle would be a bug in the model.
    """
    rng = np.random.default_rng(20260813)
    q = rng.standard_normal((stack, seq_q, head_dim), dtype=np.float32)
    k = rng.standard_normal((stack, seq_k, head_dim), dtype=np.float32)
    v = rng.standard_normal((stack, seq_k, head_dim), dtype=np.float32)
    plan = plan_tiles(head_dim, 4, V3D_LIMITS, seq_len_q=seq_q, seq_len_k=seq_k)
    scale = 1.0 / float(np.sqrt(head_dim))

    got = _dispatch(
        q,
        k,
        v,
        plan=plan,
        scale=scale,
        is_causal=is_causal,
        device_index=device_index,
    )
    model = blocked_attention(q, k, v, plan, scale=scale, is_causal=is_causal)
    oracle = get_backend("reference")(q, k, v, is_causal=is_causal)

    np.testing.assert_allclose(got, model, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(got, oracle, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not vulkan_available(), reason="no Vulkan compute device")
def test_causal_row_zero_attends_to_itself_only() -> None:
    """The first query row's output is exactly the first value row."""
    stack, seq, head_dim = 1, 4, 8
    rng = np.random.default_rng(7)
    q = rng.standard_normal((stack, seq, head_dim), dtype=np.float32)
    k = rng.standard_normal((stack, seq, head_dim), dtype=np.float32)
    v = rng.standard_normal((stack, seq, head_dim), dtype=np.float32)
    plan = plan_tiles(head_dim, 4, V3D_LIMITS, seq_len_q=seq, seq_len_k=seq)

    got = _dispatch(
        q,
        k,
        v,
        plan=plan,
        scale=1.0,
        is_causal=True,
        device_index=None,
    )
    np.testing.assert_allclose(got[0, 0], v[0, 0], rtol=1e-6, atol=1e-6)
