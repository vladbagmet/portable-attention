"""CPU reference backend for scaled dot-product attention.

This is the correctness oracle for the whole project: a straightforward,
numerically stable NumPy implementation with no backend-specific tricks. It is
deliberately simple — every future backend is validated against the output of
this function. It runs anywhere NumPy runs, which is the project's portability
floor.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["scaled_dot_product_attention"]

Array = NDArray[np.floating]


def _expand_kv_for_gqa(query: Array, key: Array, value: Array) -> tuple[Array, Array]:
    """Repeat key/value heads to match query heads for grouped-query attention.

    Under grouped-query attention the query carries ``H_q`` heads while key and
    value share ``H_kv`` heads, with ``H_q`` a positive multiple of ``H_kv``;
    each group of ``H_q // H_kv`` query heads attends to one key/value head. The
    head axis is ``-3`` (inputs shaped ``(*, H, S, E)``), so each key/value head
    is repeated in place along that axis. This matches ``torch``'s
    ``repeat_interleave`` grouping, so the result equals plain multi-head
    attention on the expanded key/value.

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
    if k_heads == 0 or q_heads % k_heads != 0:
        raise ValueError(
            f"query head count {q_heads} must be a positive multiple of the "
            f"key/value head count {k_heads} for grouped-query attention."
        )
    repeats = q_heads // k_heads
    if repeats == 1:
        return key, value
    return np.repeat(key, repeats, axis=-3), np.repeat(value, repeats, axis=-3)


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
    """Compute scaled dot-product attention on CPU.

    The signature and parameter order mirror
    ``torch.nn.functional.scaled_dot_product_attention`` so this can act as a
    drop-in for the inference path. The CPU reference implements the forward,
    non-dropout computation. ``dropout_p`` must be left at ``0.0`` (a non-zero
    value raises ``NotImplementedError``); ``enable_gqa=True`` is supported and
    activates grouped-query attention (see below).

    Inputs are batched with an arbitrary number of leading dimensions:

    - ``query``: shape ``(*, L, E)``
    - ``key``:   shape ``(*, S, E)``
    - ``value``: shape ``(*, S, Ev)``

    Returns an array of shape ``(*, L, Ev)``.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        attn_mask: Optional mask broadcastable to ``(*, L, S)``. A boolean mask
            keeps positions that are ``True`` (``False`` positions are masked
            out); a floating mask is added to the attention scores.
        dropout_p: Attention dropout probability. Only ``0.0`` is supported by
            this deterministic CPU reference; any other value raises
            ``NotImplementedError``.
        is_causal: If ``True``, apply a causal (lower-triangular) mask. Must not
            be combined with an explicit ``attn_mask``.
        scale: Softmax scale (keyword-only). Defaults to ``1 / sqrt(E)``.
        enable_gqa: If ``True``, enable grouped-query attention: ``key`` and
            ``value`` may carry fewer heads than ``query`` (on axis ``-3``),
            with ``query``'s head count a positive multiple of theirs. Their
            heads are repeated to match ``query`` before attention.

    Returns:
        The attention output.

    Raises:
        ValueError: On incompatible shapes, ``is_causal`` combined with
            ``attn_mask``, or an invalid grouped-query head configuration.
        NotImplementedError: If ``dropout_p != 0.0``.
    """
    if dropout_p != 0.0:
        raise NotImplementedError(
            "dropout_p is not supported by the CPU reference backend; "
            "pass dropout_p=0.0."
        )
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

    e = query.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(e)

    # Compute in float64 for a stable oracle regardless of input dtype.
    q = query.astype(np.float64)
    k = key.astype(np.float64)
    v = value.astype(np.float64)

    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale

    if is_causal:
        length, source = scores.shape[-2], scores.shape[-1]
        causal = np.tril(np.ones((length, source), dtype=bool))
        scores = np.where(causal, scores, -np.inf)
    elif attn_mask is not None:
        if attn_mask.dtype == np.bool_:
            scores = np.where(attn_mask, scores, -np.inf)
        else:
            scores = scores + attn_mask.astype(np.float64)

    # Subtract the per-row max for numerical stability. A fully-masked row is
    # all -inf, whose max is -inf; avoid the -inf - -inf = nan trap by shifting
    # such rows by 0 (they still exp() to 0 and get zeroed by the divide guard).
    row_max = np.max(scores, axis=-1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    scores = scores - row_max
    weights = np.exp(scores)
    denom = np.sum(weights, axis=-1, keepdims=True)
    # Rows fully masked to -inf yield 0/0; define their output as 0.
    weights = np.divide(weights, denom, out=np.zeros_like(weights), where=denom > 0)

    out: Array = np.matmul(weights, v).astype(query.dtype)
    return out
