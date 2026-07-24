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
    non-dropout computation; ``dropout_p`` and ``enable_gqa`` are accepted to
    keep the surface identical but must be left at their defaults (a non-default
    value raises ``NotImplementedError`` rather than silently ignoring it).

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
        enable_gqa: Grouped-query attention. Not yet supported; only ``False``
            is accepted (any other value raises ``NotImplementedError``).

    Returns:
        The attention output.

    Raises:
        ValueError: On incompatible shapes or ``is_causal`` combined with
            ``attn_mask``.
        NotImplementedError: If ``dropout_p != 0.0`` or ``enable_gqa`` is set.
    """
    if dropout_p != 0.0:
        raise NotImplementedError(
            "dropout_p is not supported by the CPU reference backend; "
            "pass dropout_p=0.0."
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
