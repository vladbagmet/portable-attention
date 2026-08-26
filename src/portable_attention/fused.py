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

from .backward import expected_output_shape, fold_gqa_heads, sum_to_shape

__all__ = [
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_backward",
]

Array = NDArray[np.floating]

# BLAS thread count used for the compute region. Attention GEMMs are small and
# thin, so a single thread avoids the OpenBLAS thread-sync overhead that makes
# the default (one-thread-per-core) policy slower.
_BLAS_THREADS = 1


def _expand_kv_for_gqa(query: Array, key: Array, value: Array) -> tuple[Array, Array]:
    """Repeat key/value heads to match query heads for grouped-query attention.

    Under grouped-query attention the query carries ``H_q`` heads while key and
    value share ``H_kv`` heads, with ``H_q`` a positive multiple of ``H_kv``;
    each group of ``H_q // H_kv`` query heads attends to one key/value head. The
    head axis is ``-3`` (inputs shaped ``(*, H, S, E)``), so each key/value head
    is repeated in place along that axis, matching ``torch``'s
    ``repeat_interleave`` grouping. The repeat preserves dtype.

    Raises:
        ValueError: If any input lacks a head axis (fewer than 3 dims), the key
            and value head counts differ, or ``H_q`` is not a positive multiple
            of ``H_kv``.
    """
    if query.ndim < 3 or key.ndim < 3 or value.ndim < 3:
        raise ValueError(
            "enable_gqa=True requires a head dimension: query, key, and value "
            "must each have at least 3 dims (*, H, S, E)."
        )
    q_heads = query.shape[-3]
    k_heads = key.shape[-3]
    if k_heads != value.shape[-3]:
        raise ValueError(f"key/value head dims differ: {k_heads} vs {value.shape[-3]}.")
    if q_heads == 0 or k_heads == 0 or q_heads % k_heads != 0:
        raise ValueError(
            f"query head count {q_heads} must be a positive multiple of the "
            f"key/value head count {k_heads} for grouped-query attention."
        )
    repeats = q_heads // k_heads
    if repeats == 1:
        return key, value
    return np.repeat(key, repeats, axis=-3), np.repeat(value, repeats, axis=-3)


def _threadpoolctl() -> ModuleType | None:
    """Import the optional ``threadpoolctl`` module, or ``None`` if absent."""
    if importlib.util.find_spec("threadpoolctl") is None:
        return None
    import threadpoolctl

    return threadpoolctl


def _validate_and_expand(
    query: Array,
    key: Array,
    value: Array,
    attn_mask: NDArray[np.floating] | NDArray[np.bool_] | None,
    is_causal: bool,
    enable_gqa: bool,
) -> tuple[Array, Array]:
    """Check the shape/argument contract and expand key/value for GQA.

    Shared by this backend's forward and backward passes so both accept exactly
    the same inputs and reject the same ones with the same messages. Returns the
    ``(key, value)`` pair to attend with: the originals, or their head-expanded
    views when ``enable_gqa`` is set.
    """
    if is_causal and attn_mask is not None:
        raise ValueError("Pass either is_causal=True or attn_mask, not both.")
    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError("query, key, and value must each have at least 2 dims.")
    if enable_gqa:
        key, value = _expand_kv_for_gqa(query, key, value)
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            f"query/key embedding dims differ: {query.shape[-1]} vs {key.shape[-1]}."
        )
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(
            f"key/value sequence dims differ: {key.shape[-2]} vs {value.shape[-2]}."
        )
    return key, value


def _compute_dtype(
    operands: tuple[Array, ...],
    attn_mask: NDArray[np.floating] | NDArray[np.bool_] | None,
) -> np.dtype[Any]:
    """Pick the arithmetic dtype: the widest operand, at least ``float32``.

    Computing in the widest floating precision among all operands preserves
    precision for mixed-dtype inputs instead of silently downcasting to
    ``query``'s dtype; the ``float32`` floor keeps a ``float16`` softmax
    numerically safe. A boolean mask carries no precision and is ignored.
    """
    dtypes: list[np.dtype[Any]] = [operand.dtype for operand in operands]
    if attn_mask is not None and attn_mask.dtype != np.bool_:
        dtypes.append(attn_mask.dtype)
    return np.result_type(*dtypes, np.float32)


def _softmax_weights(
    query: Array,
    key: Array,
    attn_mask: NDArray[np.floating] | NDArray[np.bool_] | None,
    is_causal: bool,
    scale: float,
    compute_dtype: np.dtype[Any],
) -> Array:
    """Return the attention probabilities in ``compute_dtype``.

    ``query`` and ``key`` must already be validated, head-expanded, and cast to
    ``compute_dtype``. Factored out so the backward pass differentiates the same
    probabilities the forward pass computes rather than a second, subtly
    different formulation.
    """
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) * scale

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
    normalized: Array = np.divide(
        weights, denom, out=np.zeros_like(weights), where=denom > 0
    )
    return normalized


def _slice_count(array: Array) -> int:
    """Number of batched GEMM slices: the product of the leading dims."""
    return int(np.prod(array.shape[:-2])) if array.ndim > 2 else 1


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
        enable_gqa: If ``True``, enable grouped-query attention: ``key`` and
            ``value`` may carry fewer heads than ``query`` (on axis ``-3``),
            their heads repeated to match before attention.

    Returns:
        The attention output of shape ``(*, L, Ev)`` with ``query``'s dtype.

    Raises:
        ValueError: On incompatible shapes, ``is_causal`` combined with
            ``attn_mask``, or an invalid grouped-query head configuration.
        NotImplementedError: If ``dropout_p != 0.0``.
    """
    if dropout_p != 0.0:
        raise NotImplementedError(
            "dropout_p is not supported by the fused CPU backend; pass dropout_p=0.0."
        )
    key, value = _validate_and_expand(
        query, key, value, attn_mask, is_causal, enable_gqa
    )
    compute_dtype = _compute_dtype((query, key, value), attn_mask)
    q = query.astype(compute_dtype, copy=False)
    k = key.astype(compute_dtype, copy=False)
    v = value.astype(compute_dtype, copy=False)

    # Keep the scale a plain Python float so it does not (under NEP 50) upcast a
    # float32 score array back to float64.
    scale_val: float = float(1.0 / np.sqrt(q.shape[-1])) if scale is None else scale

    with _single_thread_blas(pin=_slice_count(q) > 1):
        weights = _softmax_weights(q, k, attn_mask, is_causal, scale_val, compute_dtype)
        attended = np.matmul(weights, v)

    out: Array = attended.astype(query.dtype, copy=False)
    return out


def scaled_dot_product_attention_backward(
    grad_output: Array,
    query: Array,
    key: Array,
    value: Array,
    attn_mask: NDArray[np.floating] | NDArray[np.bool_] | None = None,
    is_causal: bool = False,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> tuple[Array, Array, Array]:
    """Compute the gradients of this backend's forward pass.

    Same vector-Jacobian product as
    :func:`portable_attention.backward.scaled_dot_product_attention_backward`
    (see it for the derivation and the argument contract), evaluated in the
    native compute precision with BLAS pinned, so a training loop keeps the
    forward pass's precision and threading behaviour on the way back. The
    probabilities are recomputed from the inputs, sharing the forward pass's
    softmax.

    Args:
        grad_output: Upstream gradient, shaped like the forward output.
        query: Query tensor of shape ``(*, L, E)``.
        key: Key tensor of shape ``(*, S, E)``.
        value: Value tensor of shape ``(*, S, Ev)``.
        attn_mask: The mask the forward pass used, if any.
        is_causal: Whether the forward pass applied a causal mask.
        scale: Softmax scale (keyword-only). Defaults to ``1 / sqrt(E)``.
        enable_gqa: Whether the forward pass ran grouped-query attention.

    Returns:
        ``(dq, dk, dv)``, each shaped like — and carrying the dtype of — the
        corresponding input.

    Raises:
        ValueError: On the same input violations the forward pass rejects, or if
            ``grad_output`` does not match the forward output's shape.
    """
    expanded_key, expanded_value = _validate_and_expand(
        query, key, value, attn_mask, is_causal, enable_gqa
    )
    compute_dtype = _compute_dtype(
        (grad_output, query, expanded_key, expanded_value), attn_mask
    )
    g = grad_output.astype(compute_dtype, copy=False)
    q = query.astype(compute_dtype, copy=False)
    k = expanded_key.astype(compute_dtype, copy=False)
    v = expanded_value.astype(compute_dtype, copy=False)
    scale_val: float = float(1.0 / np.sqrt(q.shape[-1])) if scale is None else scale

    with _single_thread_blas(pin=_slice_count(q) > 1):
        weights = _softmax_weights(q, k, attn_mask, is_causal, scale_val, compute_dtype)
        expected = expected_output_shape(weights, v)
        if g.shape != expected:
            raise ValueError(
                f"grad_output has shape {grad_output.shape}, but the forward "
                f"output for these inputs has shape {expected}."
            )

        # dV = Pᵀ @ g, dP = g @ Vᵀ, and dS = P ⊙ (dP − Σ(dP ⊙ P)). Masked
        # entries and fully-masked rows have P = 0 and drop out on their own.
        grad_value = np.matmul(np.swapaxes(weights, -1, -2), g)
        grad_weights = np.matmul(g, np.swapaxes(v, -1, -2))
        row_dot = np.sum(grad_weights * weights, axis=-1, keepdims=True)
        grad_scores = weights * (grad_weights - row_dot)
        grad_query = np.matmul(grad_scores, k) * scale_val
        grad_key = np.matmul(np.swapaxes(grad_scores, -1, -2), q) * scale_val

    if enable_gqa:
        grad_key = fold_gqa_heads(grad_key, key.shape[-3])
        grad_value = fold_gqa_heads(grad_value, value.shape[-3])

    dq: Array = sum_to_shape(grad_query, query.shape).astype(query.dtype)
    dk: Array = sum_to_shape(grad_key, key.shape).astype(key.dtype)
    dv: Array = sum_to_shape(grad_value, value.shape).astype(value.dtype)
    return dq, dk, dv
