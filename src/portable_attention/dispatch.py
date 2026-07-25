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

The special name ``"auto"`` resolves to the best backend currently available
(today that is always ``"reference"``); as vendor backends land, ``"auto"`` is
where the selection policy will live.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

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


# Name of the backend that ``"auto"`` resolves to. Kept as a plain constant for
# now; when vendor backends land this becomes a capability-based policy.
_AUTO_TARGET = "reference"
_RESERVED = frozenset({"auto"})

_REGISTRY: dict[str, SdpaBackend] = {}


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
        ValueError: If ``name`` is empty, reserved, or already registered while
            ``overwrite`` is ``False``.
    """
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


def get_backend(name: str = "auto") -> SdpaBackend:
    """Resolve a backend by name.

    Args:
        name: A registered backend name, or ``"auto"`` (default) to select the
            best currently available backend.

    Returns:
        The backend callable.

    Raises:
        ValueError: If ``name`` is not a known backend.
    """
    resolved = _AUTO_TARGET if name == "auto" else name
    try:
        return _REGISTRY[resolved]
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
    dispatched to the ``"auto"`` backend (currently the CPU ``reference``
    implementation); to force a specific backend, use
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


# The reference backend is the correctness oracle and is always available.
register_backend("reference", _reference_sdpa)
