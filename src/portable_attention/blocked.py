"""Executable specification of the blocked (flash-style) attention algorithm.

Device kernels are hard to debug: a wrong answer on a GPU tells you nothing
about *which* of tiling, masking, the online-softmax rescale or the accumulator
layout went wrong. This module writes that algorithm down once in NumPy,
against the same :class:`~portable_attention.tiling.TilePlan` a kernel compiles
for, so a device kernel can be diffed against a host implementation that makes
exactly the same decisions it does — and so the algorithm itself can be checked
against the reference oracle before any shader exists.

It is a *specification*, not a performance backend: it walks tiles in Python
and is slower than either CPU backend. Nothing here should be used to serve
traffic; use ``backend="fused"`` for that.

What it models faithfully:

* **Tiles come from the plan.** Query rows are processed ``block_q`` at a time
  (one workgroup's worth), keys and values stream in ``block_k``-row steps.
* **Padded tiles.** A tile at the end of a sequence is loaded full-size with
  its out-of-range lanes masked out, the way a kernel that cannot resize its
  shared arrays has to do it. Padding therefore has to be *provably* inert,
  which is what the tests check.
* **Online softmax.** Each query row carries a running max ``m`` and running
  sum ``l``; a new tile rescales the accumulator by ``exp(m_old - m_new)``
  rather than revisiting earlier tiles. The full ``L x S`` score matrix never
  exists.
* **Register-resident output.** The accumulator is per query row and
  ``head_dim`` wide, and it lives in registers rather than shared memory (see
  the ``tiling`` module docstring). This model keeps it in a per-tile array of
  the same shape, so a plan whose accumulator would not fit in registers shows
  up here as an honest ``accumulators_per_invocation`` count, not as silently
  different arithmetic.
* **Causal tile skipping.** A key tile entirely above the diagonal contributes
  nothing, so it is never loaded — the reason blocked kernels are cheap on
  causal workloads.

What it does not model: dropout, explicit ``attn_mask``, grouped-query
attention, and the leading batch/head dimensions. Inputs are the flat
``(n, seq, head_dim)`` stack a kernel dispatch actually sees; callers reshape
``(*, H, S, E)`` down to it themselves.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .tiling import TilePlan

__all__ = ["blocked_attention"]

Array = NDArray[np.floating]

# Compute dtypes a tile plan's element size can stand for. Kernels pick one
# dtype for the whole tile layout, and the plan's byte count is what sized the
# shared-memory budget, so the two must agree.
_COMPUTE_DTYPES = {4: np.float32, 8: np.float64}


def _require_stack(name: str, array: Array) -> None:
    if array.ndim != 3:
        raise ValueError(
            f"{name} must have shape (n, seq, head_dim); got {array.ndim} dims "
            f"{array.shape}. Reshape leading batch/head axes into n first."
        )


def _compute_dtype(plan: TilePlan) -> np.dtype[np.floating]:
    dtype = _COMPUTE_DTYPES.get(plan.dtype_bytes)
    if dtype is None:
        raise ValueError(
            f"no compute dtype for dtype_bytes={plan.dtype_bytes}; this model "
            f"supports {sorted(_COMPUTE_DTYPES)} (float32, float64)."
        )
    return np.dtype(dtype)


def _load_tile(
    source: Array, start: int, block: int
) -> tuple[Array, NDArray[np.bool_]]:
    """Load ``block`` rows from ``start``, zero-padded, with a validity mask.

    A kernel's shared arrays are sized at compile time, so the last tile of a
    sequence is read as a full tile whose trailing rows are out of range. The
    mask says which rows are real; the padded rows are zeroed so that a stray
    read produces a finite number rather than whatever was in memory.
    """
    rows = source.shape[0]
    stop = min(start + block, rows)
    count = stop - start
    tile = np.zeros((block, source.shape[1]), dtype=source.dtype)
    tile[:count] = source[start:stop]
    valid = np.zeros(block, dtype=bool)
    valid[:count] = True
    return tile, valid


def blocked_attention(
    query: Array,
    key: Array,
    value: Array,
    plan: TilePlan,
    *,
    scale: float | None = None,
    is_causal: bool = False,
) -> Array:
    """Run tiled attention over a flat stack, following ``plan``'s tile shape.

    Args:
        query: Queries, shape ``(n, L, E)``.
        key: Keys, shape ``(n, S, E)``.
        value: Values, shape ``(n, S, E)``.
        plan: The tile shape to follow. Its ``head_dim`` must match ``E`` and
            its ``dtype_bytes`` selects the compute dtype (4 for float32, 8 for
            float64).
        scale: Softmax scale. Defaults to ``1 / sqrt(E)``.
        is_causal: If ``True``, query row ``i`` may only attend to key rows
            ``j <= i``, counting both from the start of the sequence (the
            reference backend's convention).

    Returns:
        The attention output, shape ``(n, L, E)``, in ``query``'s dtype.

    Raises:
        ValueError: If an input is not a 3D stack, the stacks disagree on ``n``
            or ``E``, key and value disagree on ``S``, ``E`` differs from the
            plan's ``head_dim``, or the plan's element size names no compute
            dtype.
    """
    _require_stack("query", query)
    _require_stack("key", key)
    _require_stack("value", value)
    if not (query.shape[0] == key.shape[0] == value.shape[0]):
        raise ValueError(
            "query, key, and value must agree on the stack dimension; got "
            f"{query.shape[0]}, {key.shape[0]}, {value.shape[0]}."
        )
    if key.shape[1] != value.shape[1]:
        raise ValueError(
            f"key/value sequence dims differ: {key.shape[1]} vs {value.shape[1]}."
        )
    head_dim = query.shape[2]
    if not (head_dim == key.shape[2] == value.shape[2]):
        raise ValueError(
            "the tile budget sizes the Q, K, and V tiles alike, so this model "
            "needs one head dim across all three; got "
            f"{head_dim}, {key.shape[2]}, {value.shape[2]}."
        )
    if head_dim != plan.head_dim:
        raise ValueError(
            f"plan was sized for head_dim={plan.head_dim} but the inputs carry "
            f"{head_dim}."
        )

    compute = _compute_dtype(plan)
    if scale is None:
        scale = 1.0 / float(np.sqrt(head_dim))

    q = query.astype(compute)
    k = key.astype(compute)
    v = value.astype(compute)
    stack, seq_q = q.shape[0], q.shape[1]
    seq_k = k.shape[1]
    out = np.zeros((stack, seq_q, head_dim), dtype=compute)

    for n in range(stack):
        for q_start in range(0, seq_q, plan.block_q):
            tile = _attend_query_tile(
                q[n],
                k[n],
                v[n],
                plan=plan,
                q_start=q_start,
                seq_k=seq_k,
                scale=scale,
                is_causal=is_causal,
            )
            rows = min(plan.block_q, seq_q - q_start)
            out[n, q_start : q_start + rows] = tile[:rows]

    return out.astype(query.dtype)


def _attend_query_tile(
    q: Array,
    k: Array,
    v: Array,
    *,
    plan: TilePlan,
    q_start: int,
    seq_k: int,
    scale: float,
    is_causal: bool,
) -> Array:
    """Compute one workgroup's ``block_q`` output rows by streaming K/V tiles."""
    compute = q.dtype
    q_tile, q_valid = _load_tile(q, q_start, plan.block_q)

    # Per-row softmax state and the output accumulator, all workgroup-local.
    running_max = np.full(plan.block_q, -np.inf, dtype=compute)
    running_sum = np.zeros(plan.block_q, dtype=compute)
    acc = np.zeros((plan.block_q, plan.head_dim), dtype=compute)

    q_index = q_start + np.arange(plan.block_q)
    for k_start in range(0, seq_k, plan.block_k):
        if is_causal and k_start > int(q_index[-1]):
            # Every key in this tile sits strictly after every query row of the
            # tile, so the whole tile is masked out: skip the loads entirely.
            break
        k_tile, k_valid = _load_tile(k, k_start, plan.block_k)
        v_tile, _ = _load_tile(v, k_start, plan.block_k)

        scores = (q_tile @ k_tile.T) * compute.type(scale)
        allowed = q_valid[:, None] & k_valid[None, :]
        if is_causal:
            allowed &= (k_start + np.arange(plan.block_k))[None, :] <= q_index[:, None]
        scores = np.where(allowed, scores, -np.inf)

        tile_max = np.max(scores, axis=1)
        new_max = np.maximum(running_max, tile_max)
        # A row with nothing allowed yet is still -inf; shifting it by 0 keeps
        # the exponentials at 0 instead of producing -inf - -inf = nan.
        new_max = np.where(np.isfinite(new_max), new_max, compute.type(0.0))

        rescale = np.exp(running_max - new_max)
        weights = np.exp(scores - new_max[:, None])
        running_sum = running_sum * rescale + np.sum(weights, axis=1)
        acc = acc * rescale[:, None] + weights @ v_tile
        running_max = new_max

    # Rows that never saw an allowed key (a padded lane, or a causal row with
    # no history) divide 0 by 0; the contract makes their output exactly zero.
    denom = running_sum[:, None]
    result: Array = np.divide(acc, denom, out=np.zeros_like(acc), where=denom > 0)
    return result
