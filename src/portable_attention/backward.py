"""CPU reference backward pass (VJP) for scaled dot-product attention.

The forward layer is only half of a training-usable attention op: a framework
that wants to fit `portable-attention` under its autograd also needs the
gradients. This module is the oracle for that half, in the same spirit as
:mod:`portable_attention.reference` — a plain, float64 NumPy transcription of
the analytic derivative, with no backend-specific tricks, that faster backends
(and eventually a device kernel) are validated against.

Given the upstream gradient ``g = dL/dout`` and the same inputs the forward
pass saw, :func:`scaled_dot_product_attention_backward` returns
``(dq, dk, dv)``. Writing ``P`` for the softmax weights and ``s`` for the
softmax scale, the derivative is::

    dV = Pᵀ @ g
    dP = g @ Vᵀ
    dS = P ⊙ (dP − rowsum(dP ⊙ P))          # softmax Jacobian, per row
    dQ = s · (dS @ K)
    dK = s · (dSᵀ @ Q)

The masking rules need no separate term: a masked position has ``P = 0``, so
``dS`` is zero there too, and a fully-masked row (whose forward output is
defined as exactly zero) contributes no gradient at all.

The weights ``P`` are recomputed here rather than carried over from a forward
call — the same recompute a fused/flash-style kernel does to avoid keeping the
``L × S`` score matrix around, and the reason this function takes the *inputs*
rather than a saved activation. It shares
:func:`portable_attention.reference.attention_weights` with the forward pass, so
the probabilities differentiated are the ones the oracle produced.

Not covered: gradients with respect to ``attn_mask`` (an additive mask receives
``dS``; callers who need it can compute it from the pieces above) and dropout,
which the forward pass does not implement either.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .reference import (
    Array,
    attention_weights,
    default_scale,
    validate_inputs,
)

__all__ = ["scaled_dot_product_attention_backward"]


def _sum_to_shape(grad: NDArray[np.float64], shape: tuple[int, ...]) -> Array:
    """Reduce ``grad`` back onto an input that NumPy broadcast in the forward.

    ``matmul`` broadcasts leading dimensions, so an input may be smaller than
    the gradient computed for it (a key of shape ``(1, H, S, E)`` attended by a
    query of shape ``(B, H, L, E)``). The chain rule sums over every axis that
    was broadcast: leading axes the input does not have at all, plus its
    length-1 axes.
    """
    reduced = grad
    while reduced.ndim > len(shape):
        reduced = reduced.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and reduced.shape[axis] != 1:
            reduced = reduced.sum(axis=axis, keepdims=True)
    return reduced


def _fold_gqa_heads(grad: NDArray[np.float64], heads: int) -> NDArray[np.float64]:
    """Sum a query-head-shaped gradient back onto ``heads`` key/value heads.

    Grouped-query attention repeats each key/value head ``H_q // H_kv`` times
    along axis ``-3``, so each original head receives the sum of its group.
    """
    repeats = grad.shape[-3] // heads
    if repeats == 1:
        return grad
    grouped = grad.reshape(*grad.shape[:-3], heads, repeats, *grad.shape[-2:])
    folded: NDArray[np.float64] = grouped.sum(axis=-3)
    return folded


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
    """Compute the gradients of scaled dot-product attention on CPU.

    The parameters after ``grad_output`` mirror the forward function's and must
    describe the same call: same inputs, same mask, same ``scale``, same
    ``enable_gqa``. ``dropout_p`` has no counterpart here because the forward
    pass supports only ``0.0``.

    Args:
        grad_output: Upstream gradient, shaped like the forward output
            ``(*, L, Ev)``.
        query: Query tensor of shape ``(*, L, E)``.
        key: Key tensor of shape ``(*, S, E)``.
        value: Value tensor of shape ``(*, S, Ev)``.
        attn_mask: The mask the forward pass used, if any.
        is_causal: Whether the forward pass applied a causal mask. Mutually
            exclusive with ``attn_mask``.
        scale: Softmax scale (keyword-only). Defaults to ``1 / sqrt(E)``.
        enable_gqa: Whether the forward pass ran grouped-query attention. The
            returned ``dk``/``dv`` are folded back onto the key/value head
            count.

    Returns:
        ``(dq, dk, dv)``, each shaped like — and carrying the dtype of — the
        corresponding input, including when the forward pass broadcast it.

    Raises:
        ValueError: On the same input violations the forward pass rejects, or
            if ``grad_output`` does not match the forward output's shape.
    """
    expanded_key, expanded_value = validate_inputs(
        query, key, value, attn_mask, is_causal, enable_gqa
    )
    weights = attention_weights(query, expanded_key, attn_mask, is_causal, scale)
    expected = (*weights.shape[:-1], expanded_value.shape[-1])
    if grad_output.shape != expected:
        raise ValueError(
            f"grad_output has shape {grad_output.shape}, but the forward output "
            f"for these inputs has shape {expected}."
        )

    scale_val = default_scale(query) if scale is None else scale
    g = grad_output.astype(np.float64)
    q = query.astype(np.float64)
    k = expanded_key.astype(np.float64)
    v = expanded_value.astype(np.float64)

    # dV = Pᵀ @ g, and dP = g @ Vᵀ.
    grad_value = np.matmul(np.swapaxes(weights, -1, -2), g)
    grad_weights = np.matmul(g, np.swapaxes(v, -1, -2))

    # Softmax Jacobian, applied row-wise: dS = P ⊙ (dP − Σ(dP ⊙ P)). Masked
    # entries and fully-masked rows have P = 0 and so drop out on their own.
    row_dot = np.sum(grad_weights * weights, axis=-1, keepdims=True)
    grad_scores = weights * (grad_weights - row_dot)

    grad_query = np.matmul(grad_scores, k) * scale_val
    grad_key = np.matmul(np.swapaxes(grad_scores, -1, -2), q) * scale_val

    if enable_gqa:
        grad_key = _fold_gqa_heads(grad_key, key.shape[-3])
        grad_value = _fold_gqa_heads(grad_value, value.shape[-3])

    dq: Array = _sum_to_shape(grad_query, query.shape).astype(query.dtype)
    dk: Array = _sum_to_shape(grad_key, key.shape).astype(key.dtype)
    dv: Array = _sum_to_shape(grad_value, value.shape).astype(value.dtype)
    return dq, dk, dv
