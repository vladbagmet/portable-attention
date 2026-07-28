"""Pluggable backend contract, registry, and public dispatch.

A *backend* is any callable that computes scaled dot-product attention with the
same signature as :func:`scaled_dot_product_attention`. Backends register under
a short name and are looked up with :func:`get_backend`. The CPU ``reference``
backend is the project's correctness oracle: it is always registered, and every
future backend is validated against its output.

The public :func:`scaled_dot_product_attention` keeps a torch-compatible
signature (no ``backend`` keyword) so it stays a drop-in. To run a specific
backend explicitly, resolve it and call it directly::

    from portable_attention import get_backend

    out = get_backend("reference")(query, key, value)

The special name ``"auto"`` resolves to a shape-aware policy rather than a
fixed backend: it inspects each call's ``query`` and routes *batched*
(multi-slice) inputs to the fast CPU ``fused`` backend while keeping
single-slice inputs on the ``reference`` oracle path (where the two perform
equivalently). This is where the selection policy lives as vendor backends
land.

The CPU ``fused`` backend computes the same forward attention as ``reference``
but in the input's native precision, with BLAS pinned to a single thread for
batched (multi-slice) inputs, which removes the multi-head latency cliff the
reference hits under default OpenBLAS threading (issue #8). ``"auto"`` selects
it automatically for batched work; select it unconditionally with
``get_backend("fused")``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .fused import scaled_dot_product_attention as _fused_sdpa
from .reference import scaled_dot_product_attention as _reference_sdpa

__all__ = [
    "SdpaBackend",
    "available_backends",
    "get_backend",
    "register_backend",
    "scaled_dot_product_attention",
]

Array = NDArray[np.floating]


@runtime_checkable
class SdpaBackend(Protocol):
    """Structural contract every attention backend must satisfy.

    A backend is a callable whose signature matches the public
    :func:`scaled_dot_product_attention`. Implementing this protocol (and
    passing the shared conformance suite) is the complete definition of "a
    portable-attention backend"; nothing else is required.
    """

    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        attn_mask: NDArray[np.floating] | NDArray[np.bool_] | None = ...,
        dropout_p: float = ...,
        is_causal: bool = ...,
        *,
        scale: float | None = ...,
        enable_gqa: bool = ...,
    ) -> Array:
        """Compute scaled dot-product attention. See the public function."""
        ...


_RESERVED = frozenset({"auto"})

_REGISTRY: dict[str, SdpaBackend] = {}


def _reject_bad_registration(name: object, backend: object) -> None:
    """Type-guard registration inputs for untyped callers."""
    if not isinstance(name, str):
        raise TypeError(f"backend name must be a string, got {type(name).__name__}.")
    if not callable(backend):
        raise TypeError("backend must be callable (an SdpaBackend).")


def register_backend(
    name: str, backend: SdpaBackend, *, overwrite: bool = False
) -> None:
    """Register ``backend`` under ``name``.

    Args:
        name: Non-empty lookup name. Must not be a reserved name (``"auto"``).
        backend: A callable satisfying :class:`SdpaBackend`.
        overwrite: If ``False`` (default), registering a name that already
            exists raises; pass ``True`` to replace it.

    Raises:
        TypeError: If ``name`` is not a string or ``backend`` is not callable.
        ValueError: If ``name`` is empty, reserved, or already registered while
            ``overwrite`` is ``False``.
    """
    # Guards accept ``object`` so they also protect untyped callers (the type
    # annotations above only bind static callers).
    _reject_bad_registration(name, backend)
    if not name:
        raise ValueError("backend name must be a non-empty string.")
    if name in _RESERVED:
        raise ValueError(f"backend name {name!r} is reserved.")
    if name in _REGISTRY and not overwrite:
        raise ValueError(
            f"backend {name!r} is already registered; pass overwrite=True "
            "to replace it."
        )
    _REGISTRY[name] = backend


def available_backends() -> list[str]:
    """Return the sorted names of all registered backends (excludes ``auto``)."""
    return sorted(_REGISTRY)


def _is_batched(query: object) -> bool:
    """Return ``True`` when ``query`` has more than one batched GEMM slice.

    Mirrors the fused backend's slice count: the product of all leading
    (non-matrix) dimensions. Anything without a usable numeric shape (an
    untyped caller, or a 0/1/2-D input) counts as *not* batched, so the policy
    stays conservative and the chosen backend performs its own input
    validation.
    """
    shape = getattr(query, "shape", None)
    if shape is None or len(shape) < 3:
        return False
    return int(np.prod(shape[:-2])) > 1


def _auto_select(query: object) -> SdpaBackend:
    """Return the backend the ``"auto"`` policy picks for ``query``.

    The ``fused`` backend removes the multi-head OpenBLAS latency cliff
    (issue #8) on batched (multi-slice) inputs, so ``"auto"`` routes those to it
    when it is registered. Single-slice inputs — where the two backends perform
    equivalently — stay on the ``reference`` oracle path. If ``fused`` is not
    registered (e.g. it was unregistered), ``"auto"`` always falls back to
    ``reference``.
    """
    fused = _REGISTRY.get("fused")
    if fused is not None and _is_batched(query):
        return fused
    return _REGISTRY["reference"]


def _auto_dispatch(
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
    """The ``"auto"`` backend: pick a backend per call and delegate to it.

    Selection is driven by :func:`_auto_select` (batched inputs go to the fast
    ``fused`` backend, single-slice inputs to the ``reference`` oracle). The
    arguments are forwarded unchanged, so the observable contract is identical
    to calling the selected backend directly.
    """
    return _auto_select(query)(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def get_backend(name: str = "auto") -> SdpaBackend:
    """Resolve a backend by name.

    Args:
        name: A registered backend name, or ``"auto"`` (default) for the
            shape-aware policy that selects the best backend per call.

    Returns:
        The backend callable. For ``"auto"`` this is a stable dispatcher that
        chooses a concrete backend on each call from the input shape.

    Raises:
        ValueError: If ``name`` is not a known backend.
    """
    if name == "auto":
        return _auto_dispatch
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown backend {name!r}; available: {available_backends()}"
        ) from None


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
    """Compute scaled dot-product attention using the default backend.

    This is the package's public entry point. Its signature mirrors
    ``torch.nn.functional.scaled_dot_product_attention`` so it can act as a
    drop-in on hardware where the fast vendor path is missing. The call is
    dispatched to the ``"auto"`` backend, whose shape-aware policy routes
    batched inputs to the fast CPU ``fused`` backend and single-slice inputs to
    the CPU ``reference`` implementation; to force a specific backend, use
    ``get_backend(name)(...)`` directly.

    See the reference backend for the full parameter and error contract; this
    wrapper forwards its arguments unchanged.

    Args:
        query: Query tensor of shape ``(*, L, E)``.
        key: Key tensor of shape ``(*, S, E)``.
        value: Value tensor of shape ``(*, S, Ev)``.
        attn_mask: Optional mask broadcastable to ``(*, L, S)``.
        dropout_p: Attention dropout probability (only ``0.0`` is supported).
        is_causal: Apply a causal mask. Mutually exclusive with ``attn_mask``.
        scale: Softmax scale (keyword-only). Defaults to ``1 / sqrt(E)``.
        enable_gqa: Grouped-query attention (not yet supported).

    Returns:
        The attention output of shape ``(*, L, Ev)``.
    """
    return get_backend("auto")(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


# The reference backend is the correctness oracle and is always available; the
# fused backend is the fast, dtype-preserving CPU path validated against it.
register_backend("reference", _reference_sdpa)
register_backend("fused", _fused_sdpa)
