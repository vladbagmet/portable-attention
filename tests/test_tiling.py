"""Tests for the device-parameterized tile sizing policy.

The policy is the seam every blocked backend shares, so the invariants matter
more than any single tile shape: a plan must always be *legal* on the device it
was sized for (shared memory and workgroup size within limits), and it must be
derived from the limits alone — never from which device supplied them.
"""

from __future__ import annotations

import itertools

import pytest

from portable_attention.tiling import (
    M1_PRO_LIMITS,
    V3D_LIMITS,
    DeviceLimits,
    TilePlan,
    TileSizingError,
    plan_tiles,
    shared_memory_bytes_for,
)

# Head dimensions in common use (GPT-2 through Llama-class models) crossed with
# float16 / float32 / float64 element sizes.
HEAD_DIMS = [8, 16, 32, 64, 80, 96, 128, 256]
DTYPE_BYTES = [2, 4, 8]
DEVICES = [V3D_LIMITS, M1_PRO_LIMITS]


def test_known_device_limits_match_measurements():
    # These are the numbers recorded in ROADMAP.md from the two reference
    # devices; drifting from them silently would invalidate every plan.
    assert (
        V3D_LIMITS.shared_memory_bytes,
        V3D_LIMITS.max_threads_per_group,
        V3D_LIMITS.simd_width,
    ) == (16384, 256, 16)
    assert (
        M1_PRO_LIMITS.shared_memory_bytes,
        M1_PRO_LIMITS.max_threads_per_group,
        M1_PRO_LIMITS.simd_width,
    ) == (32768, 1024, 32)


@pytest.mark.parametrize(
    ("limits", "head_dim", "dtype_bytes"),
    [(d, h, b) for d, h, b in itertools.product(DEVICES, HEAD_DIMS, DTYPE_BYTES)],
)
def test_plan_is_legal_on_the_device(limits, head_dim, dtype_bytes):
    plan = plan_tiles(head_dim, dtype_bytes, limits)
    assert plan.shared_memory_bytes <= limits.shared_memory_bytes
    assert plan.threads_per_group <= limits.max_threads_per_group
    assert plan.threads_per_group == plan.block_q * plan.block_k
    # Powers of two keep the softmax reduction tree and the subgroup split exact.
    assert plan.block_q & (plan.block_q - 1) == 0
    assert plan.block_k & (plan.block_k - 1) == 0


@pytest.mark.parametrize(
    ("limits", "head_dim", "dtype_bytes"),
    [(d, h, b) for d, h, b in itertools.product(DEVICES, HEAD_DIMS, DTYPE_BYTES)],
)
def test_plan_reports_the_layout_it_priced(limits, head_dim, dtype_bytes):
    plan = plan_tiles(head_dim, dtype_bytes, limits)
    assert plan.shared_memory_bytes == shared_memory_bytes_for(
        plan.block_q, plan.block_k, head_dim, dtype_bytes
    )
    assert plan.head_dim == head_dim
    assert plan.dtype_bytes == dtype_bytes
    assert plan.limits is limits


@pytest.mark.parametrize(
    ("limits", "head_dim", "dtype_bytes"),
    [(d, h, b) for d, h, b in itertools.product(DEVICES, HEAD_DIMS, DTYPE_BYTES)],
)
def test_plan_is_maximal(limits, head_dim, dtype_bytes):
    # Nothing legal may beat the chosen plan on (threads, block_k, block_q).
    plan = plan_tiles(head_dim, dtype_bytes, limits)
    chosen = (plan.threads_per_group, plan.block_k, plan.block_q)
    block_sizes = [2**e for e in range(12)]
    for block_q, block_k in itertools.product(block_sizes, block_sizes):
        if block_q * block_k > limits.max_threads_per_group:
            continue
        used = shared_memory_bytes_for(block_q, block_k, head_dim, dtype_bytes)
        if used > limits.shared_memory_bytes:
            continue
        assert (block_q * block_k, block_k, block_q) <= chosen


def test_v3d_float32_head_dim_64_plan():
    # A worked example so a change in the preference order is visible in review:
    # 16x16 fills the 256-invocation workgroup inside 16 KiB.
    plan = plan_tiles(64, 4, V3D_LIMITS)
    assert (plan.block_q, plan.block_k) == (16, 16)
    assert plan.threads_per_group == 256
    assert plan.shared_memory_bytes == 13440
    assert plan.subgroups_per_group == 16.0
    assert 0.8 < plan.shared_memory_utilization < 0.9


def test_wider_device_is_never_worse():
    # Same workload, strictly more capable device: the plan may not shrink.
    for head_dim, dtype_bytes in itertools.product(HEAD_DIMS, DTYPE_BYTES):
        small = plan_tiles(head_dim, dtype_bytes, V3D_LIMITS)
        large = plan_tiles(head_dim, dtype_bytes, M1_PRO_LIMITS)
        assert large.threads_per_group >= small.threads_per_group


def test_policy_ignores_device_identity():
    # Two limit triples that differ only in the label must plan identically.
    anonymous = DeviceLimits(
        shared_memory_bytes=V3D_LIMITS.shared_memory_bytes,
        max_threads_per_group=V3D_LIMITS.max_threads_per_group,
        simd_width=V3D_LIMITS.simd_width,
    )
    named = plan_tiles(64, 4, V3D_LIMITS)
    unnamed = plan_tiles(64, 4, anonymous)
    assert (named.block_q, named.block_k) == (unnamed.block_q, unnamed.block_k)


def test_sequence_lengths_cap_the_blocks():
    # Caps round UP to the next power of two, so one tile can still cover the
    # whole sequence: 3 query rows permit block_q=4, 24 key rows permit 32.
    plan = plan_tiles(64, 4, M1_PRO_LIMITS, seq_len_q=3, seq_len_k=24)
    assert plan.block_q <= 4
    assert plan.block_k <= 32
    assert plan.q_tiles(3) == 1


def test_sequence_lengths_only_shrink_the_plan():
    unbounded = plan_tiles(32, 4, M1_PRO_LIMITS)
    bounded = plan_tiles(32, 4, M1_PRO_LIMITS, seq_len_q=8, seq_len_k=8)
    assert bounded.block_q <= min(8, unbounded.block_q)
    assert bounded.block_k <= min(8, unbounded.block_k)


def test_tile_counts_cover_the_sequence():
    plan = plan_tiles(64, 4, V3D_LIMITS)
    for seq_len in (1, plan.block_q, plan.block_q + 1, 1024):
        assert plan.q_tiles(seq_len) * plan.block_q >= seq_len
    for seq_len in (1, plan.block_k, plan.block_k + 1, 1024):
        assert plan.k_tiles(seq_len) * plan.block_k >= seq_len
    # A partial trailing tile is one extra step, not two.
    assert plan.q_tiles(plan.block_q + 1) == 2
    assert plan.k_tiles(plan.block_k * 3) == 3


@pytest.mark.parametrize("seq_len", [0, -4, 12.0, True])
def test_tile_counts_reject_non_positive_integer_lengths(seq_len):
    plan = plan_tiles(64, 4, V3D_LIMITS)
    with pytest.raises(ValueError, match="seq_len_q must be a positive integer"):
        plan.q_tiles(seq_len)
    with pytest.raises(ValueError, match="seq_len_k must be a positive integer"):
        plan.k_tiles(seq_len)


def test_tiny_workgroup_budget_still_plans():
    # A device that can only host one subgroup: the plan must stay legal rather
    # than round up to a workgroup the device cannot launch.
    limits = DeviceLimits(
        shared_memory_bytes=2048, max_threads_per_group=8, simd_width=8
    )
    plan = plan_tiles(16, 4, limits)
    assert plan.threads_per_group <= 8
    assert plan.shared_memory_bytes <= 2048


def test_shared_memory_starvation_forces_a_partial_subgroup():
    # Only a 1x1 tile fits, which is narrower than the SIMD width. That is a
    # legal (if wasteful) plan, and it must be reported as such.
    head_dim = 64
    dtype_bytes = 4
    limits = DeviceLimits(
        shared_memory_bytes=shared_memory_bytes_for(1, 1, head_dim, dtype_bytes),
        max_threads_per_group=256,
        simd_width=16,
    )
    plan = plan_tiles(head_dim, dtype_bytes, limits)
    assert (plan.block_q, plan.block_k) == (1, 1)
    assert plan.subgroups_per_group < 1.0
    assert plan.shared_memory_utilization == 1.0


def test_impossible_head_dim_raises():
    limits = DeviceLimits(
        shared_memory_bytes=1024, max_threads_per_group=256, simd_width=16, name="tiny"
    )
    with pytest.raises(TileSizingError, match="no tile fits tiny"):
        plan_tiles(4096, 4, limits)


def test_impossible_device_without_a_name_still_explains():
    limits = DeviceLimits(
        shared_memory_bytes=64, max_threads_per_group=64, simd_width=16
    )
    with pytest.raises(TileSizingError, match="no tile fits device"):
        plan_tiles(1024, 8, limits)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shared_memory_bytes": 0}, "shared_memory_bytes must be a positive integer"),
        ({"max_threads_per_group": -1}, "max_threads_per_group must be a positive"),
        ({"simd_width": 0}, "simd_width must be a positive integer"),
        # A float byte count would price a fractional layout, and bool is an
        # int subclass that would sail through a naive isinstance check.
        ({"shared_memory_bytes": 16384.0}, "shared_memory_bytes must be a positive"),
        ({"simd_width": 16.0}, "simd_width must be a positive integer"),
        ({"max_threads_per_group": True}, "max_threads_per_group must be a positive"),
        ({"simd_width": 24}, "simd_width must be a power of two"),
        ({"max_threads_per_group": 8}, "max_threads_per_group must be at least"),
    ],
)
def test_device_limits_validation(kwargs, message):
    base = {
        "shared_memory_bytes": 16384,
        "max_threads_per_group": 256,
        "simd_width": 16,
    }
    with pytest.raises(ValueError, match=message):
        DeviceLimits(**{**base, **kwargs})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"head_dim": 0}, "head_dim must be a positive integer"),
        ({"dtype_bytes": -2}, "dtype_bytes must be a positive integer"),
        ({"seq_len_q": 0}, "seq_len_q must be a positive integer"),
        ({"seq_len_k": -8}, "seq_len_k must be a positive integer"),
        ({"head_dim": 64.0}, "head_dim must be a positive integer"),
        ({"dtype_bytes": 4.5}, "dtype_bytes must be a positive integer"),
        ({"seq_len_k": True}, "seq_len_k must be a positive integer"),
    ],
)
def test_plan_tiles_validation(kwargs, message):
    base = {"head_dim": 64, "dtype_bytes": 4, "limits": V3D_LIMITS}
    with pytest.raises(ValueError, match=message):
        plan_tiles(**{**base, **kwargs})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"block_q": 0}, "block_q must be a positive integer"),
        ({"block_k": -1}, "block_k must be a positive integer"),
        ({"head_dim": 0}, "head_dim must be a positive integer"),
        ({"dtype_bytes": 0}, "dtype_bytes must be a positive integer"),
        ({"block_q": 8.0}, "block_q must be a positive integer"),
        ({"dtype_bytes": False}, "dtype_bytes must be a positive integer"),
    ],
)
def test_shared_memory_bytes_validation(kwargs, message):
    base = {"block_q": 8, "block_k": 16, "head_dim": 64, "dtype_bytes": 4}
    with pytest.raises(ValueError, match=message):
        shared_memory_bytes_for(**{**base, **kwargs})


def test_shared_memory_layout_is_the_documented_sum():
    # Q + K + V + scores + two running statistics per query row.
    block_q, block_k, head_dim, dtype_bytes = 8, 16, 64, 4
    expected = (
        block_q * head_dim + 2 * block_k * head_dim + block_q * block_k + 2 * block_q
    ) * dtype_bytes
    assert shared_memory_bytes_for(block_q, block_k, head_dim, dtype_bytes) == expected


def test_plans_are_immutable_and_comparable():
    plan = plan_tiles(64, 4, V3D_LIMITS)
    assert plan == plan_tiles(64, 4, V3D_LIMITS)
    with pytest.raises(AttributeError):
        plan.block_q = 1  # type: ignore[misc]
    assert isinstance(plan, TilePlan)
