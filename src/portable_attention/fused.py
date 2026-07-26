"""Thread-pinned, dtype-preserving CPU backend for scaled dot-product attention.

This is a performance-oriented sibling of the CPU ``reference`` backend. It
computes exactly the same forward attention but with two structural changes
that remove the multi-head latency cliff the reference hits under default
OpenBLAS threading (see issue #8):

* **Native precision.** The reference upcasts every input to ``float64`` for a
  rock-solid oracle. This backend instead computes in the input's own precision
  (promoting only ``float16`` to ``float32`` for a numerically safe softmax),
  which halves the memory traffic and GEMM cost for the common ``float32`` path.
* **Single-threaded BLAS for the GEMM region.** Multi-head attention issues
  many small per-slice GEMMs. The default OpenBLAS policy (one thread per core)
  spends more time synchronising threads than computing them, so throughput
  *drops* as cores are added. Attention GEMMs are also "thin" (the contracted
  head dimension is small), which multithreaded BLAS handles poorly even for a
  single long-sequence head. This backend pins BLAS to one thread for the whole
  compute region via ``threadpoolctl`` when it is installed, which reliably
  removes the cliff at default threads; without ``threadpoolctl`` it still runs
  correctly (just without the pin — install it, or cap ``OPENBLAS_NUM_THREADS``
  yourself, to get the speedup).

Cross-slice parallelism (spreading heads/batches across cores instead of BLAS
threads) is a future refinement; single-threaded BLAS is the reliable floor.

The output dtype always matches ``query`` — identical to the reference contract.
This backend is validated against the reference oracle by the conformance suite.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Generator
from contextlib import contextmanager
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["scaled_dot_product_attention"]

Array = NDArray[np.floating]

# BLAS thread count used for the compute region. Attention GEMMs are small and
# thin, so a single thread avoids the OpenBLAS thread-sync overhead that makes
# the default (one-thread-per-core) policy slower.
_BLAS_THREADS = 1


def _threadpoolctl() -> ModuleType | None:
    """Import the optional ``threadpoolctl`` module, or ``None`` if absent."""
    if importlib.util.find_spec("threadpoolctl") is None:
        return None
    import threadpoolctl

    return threadpoolctl


@contextmanager
def _single_thread_blas(*, pin: bool) -> Generator[None, None, None]:
    """Pin BLAS to one thread for the duration of the context.

    A no-op when ``pin`` is ``False`` (a single GEMM slice — see below) or when
    ``threadpoolctl`` is not installed; the computation is correct either way.

    The many-small-GEMM thread-sync cliff only appears when there is more than
    one batched slice, so pinning is skipped for a single slice: for a lone
    sub-millisecond GEMM the fixed cost of querying and resetting the BLAS
    thread pool would otherwise dominate the call.
    """
    mod = _threadpoolctl()
    if not pin or mod is None:
        yield
        return
    with mod.threadpool_limits(limits=_BLAS_THREADS):
        yield


def scaled_dot_product_attention(
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
    """Compute scaled dot-product attention on CPU (thread-pinned, native dtype).

    The signature, parameter contract, and errors mirror the CPU ``reference``
    backend exactly (see :func:`portable_attention.reference` for the full
    documentation); the only differences are internal: compute precision and
    BLAS thread pinning. The result matches the reference to floating tolerance
    and preserves ``query``'s dtype.

    Args:
        query: Query tensor of shape ``(*, L, E)``.
        key: Key tensor of shape ``(*, S, E)``.
        value: Value tensor of shape ``(*, S, Ev)``.
        attn_mask: Optional mask broadcastable to ``(*, L, S)``. Boolean masks
            keep ``True`` positions; floating masks are added to the scores.
        dropout_p: Only ``0.0`` is supported.
        is_causal: Apply a causal mask. Mutually exclusive with ``attn_mask``.
        scale: Softmax scale (keyword-only). Defaults to ``1 / sqrt(E)``.
        enable_gqa: Grouped-query attention (not yet supported).

    Returns:
        The attention output of shape ``(*, L, Ev)`` with ``query``'s dtype.

    Raises:
        ValueError: On incompatible shapes or ``is_causal`` combined with
            ``attn_mask``.
        NotImplementedError: If ``dropout_p != 0.0`` or ``enable_gqa`` is set.
    """
    if dropout_p != 0.0:
        raise NotImplementedError(
            "dropout_p is not supported by the fused CPU backend; pass dropout_p=0.0."
        )
    if enable_gqa:
        raise NotImplementedError(
            "enable_gqa (grouped-query attention) is not yet supported; "
            "pass enable_gqa=False."
        )
    if is_causal and attn_mask is not None:
        raise ValueError("Pass either is_causal=True or attn_mask, not both.")
    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError("query, key, and value must each have at least 2 dims.")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            f"query/key embedding dims differ: {query.shape[-1]} vs {key.shape[-1]}."
        )
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(
            f"key/value sequence dims differ: {key.shape[-2]} vs {value.shape[-2]}."
        )

    # Compute in the widest floating precision among all operands (query, key,
    # value, and an additive mask), promoting float16 up to float32 so the
    # softmax stays numerically safe. This preserves precision for mixed-dtype
    # inputs instead of silently downcasting to query's dtype.
    operand_dtypes: list[np.dtype[Any]] = [query.dtype, key.dtype, value.dtype]
    if attn_mask is not None and attn_mask.dtype != np.bool_:
        operand_dtypes.append(attn_mask.dtype)
    compute_dtype = np.result_type(*operand_dtypes, np.float32)
    q = query.astype(compute_dtype, copy=False)
    k = key.astype(compute_dtype, copy=False)
    v = value.astype(compute_dtype, copy=False)

    e = q.shape[-1]
    # Keep the scale a plain Python float so it does not (under NEP 50) upcast a
    # float32 score array back to float64.
    scale_val: float = float(1.0 / np.sqrt(e)) if scale is None else scale

    # Number of batched GEMM slices = product of the leading (non-matrix) dims.
    slices = int(np.prod(q.shape[:-2])) if q.ndim > 2 else 1
    with _single_thread_blas(pin=slices > 1):
        scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale_val

        if is_causal:
            length, source = scores.shape[-2], scores.shape[-1]
            keep = np.tril(np.ones((length, source), dtype=bool))
            scores = np.where(keep, scores, -np.inf)
        elif attn_mask is not None:
            if attn_mask.dtype == np.bool_:
                scores = np.where(attn_mask, scores, -np.inf)
            else:
                scores = scores + attn_mask.astype(compute_dtype)

        # Shift by the per-row max for stability; a fully-masked row is all
        # -inf (max -inf), so shift those rows by 0 to avoid -inf - -inf = nan.
        row_max = np.max(scores, axis=-1, keepdims=True)
        row_max = np.where(np.isfinite(row_max), row_max, 0.0)
        scores = scores - row_max
        weights = np.exp(scores)
        denom = np.sum(weights, axis=-1, keepdims=True)
        # Fully-masked rows have denom 0; define their output as exactly 0.
        weights = np.divide(weights, denom, out=np.zeros_like(weights), where=denom > 0)
        attended = np.matmul(weights, v)

    out: Array = attended.astype(query.dtype, copy=False)
    return out
