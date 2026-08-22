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

The three data tiles are rounded up to a whole number of
:data:`VECTOR_LANES`-element vectors. GPUs load shared memory fastest in short
vectors, so a kernel is free to declare those tiles as ``vec4`` arrays, and the
budget prices the padding that rounding can add (at most three elements per
tile, and none at all when ``head_dim`` is a multiple of four).

The thread mapping is one invocation per score element, so a workgroup holds
``block_q * block_k`` invocations: each computes one ``q.k`` dot product, then
the group reduces along the key axis for the softmax. Both block sizes are
powers of two, which keeps the reduction tree and the subgroup split exact.

The output accumulator is deliberately absent from that budget. It is one
``block_q x head_dim`` block, about as large as the Q, K and V tiles together
at the shapes this policy plans, and paying shared memory for it would cost
more occupancy than it buys. It lives in registers instead: invocation
``(i, j)`` owns output columns ``j``, ``j + block_k``, ``j + 2 * block_k``, ...
of query row ``i`` and updates them from the score tile and the V tile that are
already in shared memory. So an invocation holds at most
``ceil(head_dim / block_k)`` accumulators, reported as
:attr:`TilePlan.accumulators_per_invocation`, a count the preference order
below keeps small by favouring wide key tiles.

Among the tiles that fit, the policy prefers the largest workgroup (occupancy),
then the widest key tile (fewer passes over K/V), then the widest query tile.
That order is a hypothesis about hardware, not a proof, so the whole feasible
set is available through :func:`candidate_plans` and a single shape can be
requested through :func:`tile_plan_for` — which is how a backend gets swept
across tile shapes without the policy being edited to run the experiment.
Sequence lengths, when known, cap the blocks: each cap is the next power of two
at or above the length, so one tile can still cover the whole sequence
(``seq_len_q=3`` permits ``block_q=4``) while a 24-row key sequence never draws
a 512-row block. A partial trailing tile is the kernel's job to mask.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "M1_PRO_LIMITS",
    "VECTOR_LANES",
    "V3D_LIMITS",
    "DeviceLimits",
    "TilePlan",
    "TileSizingError",
    "candidate_plans",
    "plan_tiles",
    "shared_memory_bytes_for",
    "tile_plan_for",
]


#: Elements per shared-memory vector. The Q/K/V tiles are allocated in whole
#: vectors of this many elements so a kernel can load them four-wide; see the
#: module docstring.
VECTOR_LANES = 4


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
    def accumulators_per_invocation(self) -> int:
        """Output accumulators one invocation keeps in registers.

        The output block is outside the shared-memory budget (see the module
        docstring): invocation ``(i, j)`` owns output columns ``j``,
        ``j + block_k``, ... of query row ``i``, so the busiest one carries
        ``ceil(head_dim / block_k)`` running values. Past a device's register
        budget a kernel spills to local memory, so this is the number to watch
        when a plan that fits turns out slow.
        """
        return -(-self.head_dim // self.block_k)

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
        _whole_vectors(block_q * head_dim)  # Q tile
        + 2 * _whole_vectors(block_k * head_dim)  # K and V tiles
        + block_q * block_k  # scores for the current KV tile
        + 2 * block_q  # running max and running sum per query row
    )
    return elements * dtype_bytes


def _whole_vectors(elements: int) -> int:
    """Round an element count up to a whole number of :data:`VECTOR_LANES`."""
    return -(-elements // VECTOR_LANES) * VECTOR_LANES


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


def tile_plan_for(
    block_q: int,
    block_k: int,
    head_dim: int,
    dtype_bytes: int,
    limits: DeviceLimits,
) -> TilePlan:
    """Build the plan for one explicitly chosen tile shape.

    This is the escape hatch from the preference order: a caller that wants to
    measure ``4 x 64`` against ``8 x 32`` names the shape and gets a plan that
    has been checked against the device, or an error saying which limit it
    misses. Legality is still the policy's to decide, so a shape that does not
    fit never reaches a kernel.

    Args:
        block_q: Query rows per workgroup. A power of two.
        block_k: Key/value rows per streaming step. A power of two.
        head_dim: Head dimension (``E``) the kernel will run.
        dtype_bytes: Size of one element of the kernel's compute dtype.
        limits: The target device's limits.

    Returns:
        The plan for exactly this shape.

    Raises:
        ValueError: If any argument is not a positive integer, or a block size
            is not a power of two.
        TileSizingError: If the shape exceeds the device's workgroup size or
            shared memory.
    """
    for field, value in (("block_q", block_q), ("block_k", block_k)):
        _require_positive_int(field, value)
        if value & (value - 1):
            raise ValueError(f"{field} must be a power of two, got {value}")
    threads = block_q * block_k
    if threads > limits.max_threads_per_group:
        raise TileSizingError(
            f"tile {block_q}x{block_k} needs {threads} invocations per "
            f"workgroup but {limits.name or 'the device'} allows "
            f"{limits.max_threads_per_group}"
        )
    used = shared_memory_bytes_for(block_q, block_k, head_dim, dtype_bytes)
    if used > limits.shared_memory_bytes:
        raise TileSizingError(
            f"tile {block_q}x{block_k} at head_dim={head_dim} and dtype_bytes="
            f"{dtype_bytes} needs {used} bytes of shared memory but "
            f"{limits.name or 'the device'} has {limits.shared_memory_bytes}"
        )
    return TilePlan(
        block_q=block_q,
        block_k=block_k,
        threads_per_group=threads,
        shared_memory_bytes=used,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
        limits=limits,
    )


def candidate_plans(
    head_dim: int,
    dtype_bytes: int,
    limits: DeviceLimits,
    *,
    seq_len_q: int | None = None,
    seq_len_k: int | None = None,
) -> list[TilePlan]:
    """Every tile shape that fits this device, most preferred first.

    :func:`plan_tiles` returns the head of this list. The tail is what a tuning
    sweep measures: same legality rules, no claim about which shape is fastest.

    Args:
        head_dim: Head dimension (``E``) the kernel will run.
        dtype_bytes: Size of one element of the kernel's compute dtype.
        limits: The target device's limits.
        seq_len_q: Query sequence length, when known. Caps ``block_q`` at the
            next power of two at or above it.
        seq_len_k: Key/value sequence length, when known. Caps ``block_k`` the
            same way.

    Returns:
        The feasible plans, sorted by the preference order documented in the
        module docstring. Empty when nothing fits.

    Raises:
        ValueError: If any argument is not a positive integer.
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

    plans: list[TilePlan] = []
    for block_q in _powers_of_two_upto(q_cap):
        for block_k in _powers_of_two_upto(k_cap):
            threads = block_q * block_k
            if threads > max_threads:
                break  # block_k only grows from here
            used = shared_memory_bytes_for(block_q, block_k, head_dim, dtype_bytes)
            if used > limits.shared_memory_bytes:
                break  # so does the shared-memory footprint
            plans.append(
                TilePlan(
                    block_q=block_q,
                    block_k=block_k,
                    threads_per_group=threads,
                    shared_memory_bytes=used,
                    head_dim=head_dim,
                    dtype_bytes=dtype_bytes,
                    limits=limits,
                )
            )
    plans.sort(key=_rank, reverse=True)
    return plans


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
    plans = candidate_plans(
        head_dim,
        dtype_bytes,
        limits,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
    )
    if not plans:
        needed = shared_memory_bytes_for(1, 1, head_dim, dtype_bytes)
        raise TileSizingError(
            f"no tile fits {limits.name or 'device'}: a 1x1 tile at head_dim="
            f"{head_dim} and dtype_bytes={dtype_bytes} needs {needed} bytes of "
            f"shared memory but only {limits.shared_memory_bytes} are available"
        )
    return plans[0]


def _rank(plan: TilePlan) -> tuple[int, int, int]:
    """Sort key implementing the documented preference order."""
    return (plan.threads_per_group, plan.block_k, plan.block_q)
