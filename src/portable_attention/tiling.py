"""Device-parameterized tile sizing for blocked (flash-style) attention kernels.

A blocked attention kernel streams the key/value sequence in tiles so that the
full ``S x S`` score matrix never has to exist. How large those tiles may be is
not a property of the algorithm — it is a property of the device: how much
workgroup-shared memory a group can address, how many invocations a group may
contain, and how wide the SIMD/subgroup unit is.

This module holds that policy **once**. :func:`plan_tiles` takes the three
device numbers as data (:class:`DeviceLimits`) and returns the tile shape a
kernel should compile for; it never branches on device identity, so a second
backend brings a limits triple rather than a second hand-tuned kernel.

The shared-memory budget describes this per-workgroup layout, all in the
kernel's compute dtype::

    Q tile      block_q * head_dim
    K tile      block_k * head_dim
    V tile      block_k * head_dim
    S tile      block_q * block_k     (scores for the current KV tile)
    stats       2 * block_q           (running max and running sum per row)

The thread mapping is one invocation per score element, so a workgroup holds
``block_q * block_k`` invocations: each computes one ``q.k`` dot product, then
the group reduces along the key axis for the softmax. Both block sizes are
powers of two, which keeps the reduction tree and the subgroup split exact.

Among the tiles that fit, the policy prefers the largest workgroup (occupancy),
then the widest key tile (fewer passes over K/V), then the widest query tile.
Sequence lengths, when known, cap the blocks: each cap is the next power of two
at or above the length, so one tile can still cover the whole sequence
(``seq_len_q=3`` permits ``block_q=4``) while a 24-row key sequence never draws
a 512-row block. A partial trailing tile is the kernel's job to mask.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "M1_PRO_LIMITS",
    "V3D_LIMITS",
    "DeviceLimits",
    "TilePlan",
    "TileSizingError",
    "plan_tiles",
    "shared_memory_bytes_for",
]


class TileSizingError(ValueError):
    """Raised when no tile shape satisfies a device's limits."""


def _require_positive_int(name: str, value: object) -> int:
    """Return ``value`` as an ``int``, rejecting anything that is not one.

    Every quantity here counts bytes, rows, or invocations, so a float would
    silently produce fractional shared-memory budgets and a ``bool`` would slip
    through ``isinstance(..., int)``. Both are rejected before any arithmetic.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class DeviceLimits:
    """The three device numbers tile sizing depends on.

    Attributes:
        shared_memory_bytes: Bytes of workgroup-shared memory one workgroup may
            address (Vulkan ``maxComputeSharedMemorySize``, Metal
            ``maxThreadgroupMemoryLength``).
        max_threads_per_group: Invocations one workgroup may contain (Vulkan
            ``maxComputeWorkGroupInvocations``, Metal
            ``maxTotalThreadsPerThreadgroup`` of the compiled pipeline).
        simd_width: Width of the SIMD/subgroup execution unit (Vulkan
            ``subgroupSize``, Metal ``threadExecutionWidth``). Must be a power
            of two.
        name: Optional label for reports and error messages. It is never
            consulted by the sizing policy.
    """

    shared_memory_bytes: int
    max_threads_per_group: int
    simd_width: int
    name: str = ""

    def __post_init__(self) -> None:
        for field, value in (
            ("shared_memory_bytes", self.shared_memory_bytes),
            ("max_threads_per_group", self.max_threads_per_group),
            ("simd_width", self.simd_width),
        ):
            _require_positive_int(field, value)
        if self.simd_width & (self.simd_width - 1):
            raise ValueError(
                f"simd_width must be a power of two, got {self.simd_width}"
            )
        if self.max_threads_per_group < self.simd_width:
            raise ValueError(
                "max_threads_per_group must be at least simd_width "
                f"({self.max_threads_per_group} < {self.simd_width})"
            )


# Limits measured on the boards this project develops against. They are data
# for callers to pass in, not defaults the policy reaches for on its own.
V3D_LIMITS = DeviceLimits(
    shared_memory_bytes=16384,
    max_threads_per_group=256,
    simd_width=16,
    name="V3D (VideoCore VII, Mesa V3DV)",
)
M1_PRO_LIMITS = DeviceLimits(
    shared_memory_bytes=32768,
    max_threads_per_group=1024,
    simd_width=32,
    name="Apple M1 Pro (Metal)",
)


@dataclass(frozen=True)
class TilePlan:
    """A tile shape a blocked attention kernel can be compiled for.

    Attributes:
        block_q: Query rows held by one workgroup.
        block_k: Key/value rows loaded per streaming step.
        threads_per_group: Invocations per workgroup (``block_q * block_k``).
        shared_memory_bytes: Shared memory the layout consumes, in bytes.
        head_dim: Head dimension the plan was sized for.
        dtype_bytes: Size of one element of the kernel's compute dtype.
        limits: The device limits the plan satisfies.
    """

    block_q: int
    block_k: int
    threads_per_group: int
    shared_memory_bytes: int
    head_dim: int
    dtype_bytes: int
    limits: DeviceLimits

    @property
    def shared_memory_utilization(self) -> float:
        """Fraction of the device's shared memory the layout occupies."""
        return self.shared_memory_bytes / self.limits.shared_memory_bytes

    @property
    def subgroups_per_group(self) -> float:
        """Workgroup size in units of the device's SIMD width.

        Below 1.0 the workgroup does not fill a single subgroup, which wastes
        lanes; the policy avoids it whenever a wider tile fits.
        """
        return self.threads_per_group / self.limits.simd_width

    def q_tiles(self, seq_len_q: int) -> int:
        """Workgroups needed to cover ``seq_len_q`` query rows."""
        return _tiles(seq_len_q, self.block_q, "seq_len_q")

    def k_tiles(self, seq_len_k: int) -> int:
        """Streaming steps needed to cover ``seq_len_k`` key/value rows."""
        return _tiles(seq_len_k, self.block_k, "seq_len_k")


def _tiles(seq_len: int, block: int, label: str) -> int:
    _require_positive_int(label, seq_len)
    return -(-seq_len // block)


def shared_memory_bytes_for(
    block_q: int,
    block_k: int,
    head_dim: int,
    dtype_bytes: int,
) -> int:
    """Bytes of shared memory the documented tile layout needs.

    Kernels and the sizing policy must agree on the budget, so both compute it
    here rather than duplicating the arithmetic. See the module docstring for
    the layout this counts.

    Args:
        block_q: Query rows per workgroup.
        block_k: Key/value rows per streaming step.
        head_dim: Head dimension.
        dtype_bytes: Size of one element of the compute dtype.

    Returns:
        The total byte count.

    Raises:
        ValueError: If any argument is not a positive integer.
    """
    for field, value in (
        ("block_q", block_q),
        ("block_k", block_k),
        ("head_dim", head_dim),
        ("dtype_bytes", dtype_bytes),
    ):
        _require_positive_int(field, value)
    elements = (
        block_q * head_dim  # Q tile
        + 2 * block_k * head_dim  # K and V tiles
        + block_q * block_k  # scores for the current KV tile
        + 2 * block_q  # running max and running sum per query row
    )
    return elements * dtype_bytes


def _powers_of_two_upto(cap: int) -> list[int]:
    values: list[int] = []
    value = 1
    while value <= cap:
        values.append(value)
        value *= 2
    return values


def _next_power_of_two(value: int) -> int:
    power = 1
    while power < value:
        power *= 2
    return power


def plan_tiles(
    head_dim: int,
    dtype_bytes: int,
    limits: DeviceLimits,
    *,
    seq_len_q: int | None = None,
    seq_len_k: int | None = None,
) -> TilePlan:
    """Choose the tile shape for a blocked attention kernel on one device.

    Args:
        head_dim: Head dimension (``E``) the kernel will run.
        dtype_bytes: Size of one element of the kernel's compute dtype, e.g.
            4 for float32.
        limits: The target device's limits.
        seq_len_q: Query sequence length, when known. Caps ``block_q`` at the
            next power of two at or above it, so a short sequence does not
            reserve shared memory for rows that cannot exist.
        seq_len_k: Key/value sequence length, when known. Caps ``block_k`` the
            same way.

    Returns:
        The largest tile shape that fits, by the preference order documented in
        the module docstring.

    Raises:
        ValueError: If any argument is not a positive integer.
        TileSizingError: If not even a single query row against a single key
            row fits in the device's shared memory.
    """
    for field, value in (("head_dim", head_dim), ("dtype_bytes", dtype_bytes)):
        _require_positive_int(field, value)
    for field, optional in (("seq_len_q", seq_len_q), ("seq_len_k", seq_len_k)):
        if optional is not None:
            _require_positive_int(field, optional)

    max_threads = limits.max_threads_per_group
    q_cap = (
        max_threads
        if seq_len_q is None
        else min(max_threads, _next_power_of_two(seq_len_q))
    )
    k_cap = (
        max_threads
        if seq_len_k is None
        else min(max_threads, _next_power_of_two(seq_len_k))
    )

    best: TilePlan | None = None
    for block_q in _powers_of_two_upto(q_cap):
        for block_k in _powers_of_two_upto(k_cap):
            threads = block_q * block_k
            if threads > max_threads:
                break  # block_k only grows from here
            used = shared_memory_bytes_for(block_q, block_k, head_dim, dtype_bytes)
            if used > limits.shared_memory_bytes:
                break  # so does the shared-memory footprint
            candidate = TilePlan(
                block_q=block_q,
                block_k=block_k,
                threads_per_group=threads,
                shared_memory_bytes=used,
                head_dim=head_dim,
                dtype_bytes=dtype_bytes,
                limits=limits,
            )
            if best is None or _rank(candidate) > _rank(best):
                best = candidate

    if best is None:
        needed = shared_memory_bytes_for(1, 1, head_dim, dtype_bytes)
        raise TileSizingError(
            f"no tile fits {limits.name or 'device'}: a 1x1 tile at head_dim="
            f"{head_dim} and dtype_bytes={dtype_bytes} needs {needed} bytes of "
            f"shared memory but only {limits.shared_memory_bytes} are available"
        )
    return best


def _rank(plan: TilePlan) -> tuple[int, int, int]:
    """Sort key implementing the documented preference order."""
    return (plan.threads_per_group, plan.block_k, plan.block_q)
