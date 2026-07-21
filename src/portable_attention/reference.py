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


def scaled_dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    attn_mask: NDArray[np.floating] | NDArray[np.bool_] | None = None,
    is_causal: bool = False,
    scale: float | None = None,
) -> Array:
    """Compute scaled dot-product attention on CPU.

    The surface mirrors ``torch.nn.functional.scaled_dot_product_attention``.
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
        is_causal: If ``True``, apply a causal (lower-triangular) mask. Must not
            be combined with an explicit ``attn_mask``.
        scale: Softmax scale. Defaults to ``1 / sqrt(E)``.

    Returns:
        The attention output.
    """
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

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    denom = np.sum(weights, axis=-1, keepdims=True)
    # Rows fully masked to -inf yield 0/0; define their output as 0.
    weights = np.divide(weights, denom, out=np.zeros_like(weights), where=denom > 0)

    out: Array = np.matmul(weights, v).astype(query.dtype)
    return out
