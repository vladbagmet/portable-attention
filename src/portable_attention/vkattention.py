"""Launch geometry for the Vulkan blocked-attention shader.

``shaders/attention.comp`` is a transcription of
:func:`~portable_attention.blocked.blocked_attention`, and every tile dimension
it compiles for is a specialization constant. This module holds the host half
of that contract: given a :class:`~portable_attention.tiling.TilePlan` and the
shapes of a dispatch, it produces the specialization values, the push-constant
block and the workgroup counts the shader expects, plus the SPIR-V itself.

Keeping the arithmetic here — rather than inline at the call site — means the
numbers the driver bakes into the pipeline can be checked without a GPU, which
is most of what can go wrong before a kernel ever runs.

The buffer layout is fixed: bindings 0-3 are Q, K, V and the output, each a
flat ``(n, seq, head_dim)`` float32 stack in C order. Q and the output are
indexed by the query sequence length, K and V by the key one.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .tiling import TilePlan

__all__ = [
    "AttentionLaunch",
    "accumulator_slots",
    "attention_launch",
    "attention_spirv",
]

#: Output columns one accumulator slot covers when the head dimension is a
#: multiple of it. Matches the ``vec4`` the shader reads the V tile with, and
#: :data:`~portable_attention.tiling.VECTOR_LANES`, which prices the tiles.
VECTOR_COLUMNS = 4

#: Storage buffers the shader binds, in binding order: Q, K, V, output.
BUFFER_COUNT = 4

#: Size of the shader's push-constant block: seq_q, seq_k, scale, causal.
PUSH_CONSTANT_BYTES = 16

#: Workgroups a single dispatch axis may carry. Vulkan guarantees at least
#: this many everywhere (``maxComputeWorkGroupCount``), and a launch that wants
#: more is not portable, so a shape is refused here rather than at dispatch.
MAX_GROUPS_PER_AXIS = 65535

#: Element size the kernel is written for. It reads and writes float32 storage
#: buffers, so a plan sized for anything else describes a different shader.
KERNEL_DTYPE_BYTES = 4

_SPIRV_PATH = Path(__file__).parent / "shaders" / "attention.spv"

# The push-constant block is copied verbatim into memory the device then reads
# through the shader's declared layout, so it is the host's own representation
# of those values, not a byte stream with an independent order: pack it native
# ("="), never little-endian ("<").
_PUSH_CONSTANT_FORMAT = "=IIfI"


@lru_cache(maxsize=1)
def attention_spirv() -> bytes:
    """Return the compiled attention shader, read once and cached.

    Returns:
        The contents of ``shaders/attention.spv``, ready for
        :meth:`~portable_attention.vkcompute.VulkanContext.compute_pipeline`.
    """
    return _SPIRV_PATH.read_bytes()


@dataclass(frozen=True)
class AttentionLaunch:
    """Everything a dispatch of the attention shader is parameterized by.

    Attributes:
        specialization: Values for the shader's ``constant_id`` slots — the
            workgroup size, the tile shape, and the lengths of the shared
            arrays and the register accumulator.
        push_constants: The packed ``(seq_q, seq_k, scale, causal)`` block.
        groups: Workgroup counts ``(query tiles, stack, 1)``.
        plan: The tile plan these numbers were derived from.
    """

    specialization: Mapping[int, int]
    push_constants: bytes
    groups: tuple[int, int, int]
    plan: TilePlan

    @property
    def invocations_per_group(self) -> int:
        """Invocations in one workgroup (``block_q * block_k``)."""
        return self.plan.threads_per_group


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def attention_launch(
    plan: TilePlan,
    *,
    stack: int,
    seq_q: int,
    seq_k: int,
    scale: float,
    is_causal: bool = False,
) -> AttentionLaunch:
    """Derive the launch parameters for one attention dispatch.

    Args:
        plan: Tile shape the pipeline is (or will be) compiled for. Its
            ``dtype_bytes`` must be 4: the shader is float32.
        stack: Number of ``(seq, head_dim)`` slices, i.e. ``batch * heads``.
            One workgroup row per slice.
        seq_q: Query sequence length.
        seq_k: Key/value sequence length.
        scale: Softmax scale, applied to the raw dot products. Must be finite.
        is_causal: Whether query row ``i`` may attend only to key rows
            ``j <= i``, both counted from the start of the sequence.

    Returns:
        The :class:`AttentionLaunch` describing the dispatch.

    Raises:
        ValueError: If a count is not a positive integer, the plan is not a
            float32 plan, ``scale`` is not a finite float32 value, or the
            shape needs more workgroups on one axis than Vulkan guarantees.
    """
    if plan.dtype_bytes != KERNEL_DTYPE_BYTES:
        raise ValueError(
            f"the attention shader is float32; plan is sized for "
            f"dtype_bytes={plan.dtype_bytes}."
        )
    for name, value in (("stack", stack), ("seq_q", seq_q), ("seq_k", seq_k)):
        _require_positive_int(name, value)

    block_q, block_k = plan.block_q, plan.block_k
    head_dim = plan.head_dim
    specialization = {
        0: block_k,  # local_size_x: one lane per key of the tile
        1: block_q,  # local_size_y: one row per query of the tile
        2: block_q,
        3: block_k,
        4: head_dim,
        5: block_q * head_dim,  # Q tile
        6: block_k * head_dim,  # K and V tiles
        7: block_q * block_k,  # score tile
        8: accumulator_slots(plan),
    }
    groups = (plan.q_tiles(seq_q), stack, 1)
    for axis, (label, count) in enumerate(
        ((f"{seq_q} query rows in tiles of {block_q}", groups[0]), ("stack", stack))
    ):
        if count > MAX_GROUPS_PER_AXIS:
            raise ValueError(
                f"{label} needs {count} workgroups on axis {axis}, above the "
                f"{MAX_GROUPS_PER_AXIS} per axis Vulkan guarantees; split the "
                "dispatch."
            )
    return AttentionLaunch(
        specialization=specialization,
        push_constants=_pack_push_constants(
            seq_q=seq_q, seq_k=seq_k, scale=scale, is_causal=is_causal
        ),
        groups=groups,
        plan=plan,
    )


def accumulator_slots(plan: TilePlan) -> int:
    """Accumulator registers one invocation of the shader carries.

    An invocation owns a stripe of the output columns of its query row. When
    ``head_dim`` is a multiple of :data:`VECTOR_COLUMNS` the stripe is made of
    whole four-column groups, so the kernel accumulates into ``vec4`` registers
    and needs a quarter as many slots as it has columns to write; otherwise it
    owns single columns and the count is
    :attr:`~portable_attention.tiling.TilePlan.accumulators_per_invocation`.

    Args:
        plan: The tile plan the pipeline is compiled for.

    Returns:
        The value of the shader's ``ACC_LEN`` specialization constant.
    """
    if plan.head_dim % VECTOR_COLUMNS:
        return plan.accumulators_per_invocation
    groups = plan.head_dim // VECTOR_COLUMNS
    return -(-groups // plan.block_k)


def _pack_push_constants(
    *, seq_q: int, seq_k: int, scale: float, is_causal: bool
) -> bytes:
    """Pack the shader's push-constant block, rejecting an unusable scale.

    ``scale`` arrives as a Python float (binary64) and the slot is binary32, so
    a magnitude the narrower type cannot hold has to be refused here rather
    than escaping as an ``OverflowError`` from inside ``struct``.
    """
    if scale != scale or scale in (float("inf"), float("-inf")):
        raise ValueError(f"scale must be finite, got {scale!r}")
    try:
        return struct.pack(
            _PUSH_CONSTANT_FORMAT, seq_q, seq_k, scale, 1 if is_causal else 0
        )
    except OverflowError as exc:
        raise ValueError(
            f"scale {scale!r} is out of range for the shader's float32 slot"
        ) from exc
