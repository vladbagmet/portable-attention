"""Freeze the public API surface and the version single-source-of-truth.

These tests are a contract, not behavior checks: they fail loudly if the
exported names or the ``scaled_dot_product_attention`` signature drift, or if
the packaged metadata version stops matching ``__version__``. The signature is
frozen to mirror ``torch.nn.functional.scaled_dot_product_attention``.
"""

from __future__ import annotations

import inspect
from importlib.metadata import version

import portable_attention
from portable_attention import scaled_dot_product_attention


def test_public_exports_are_frozen():
    assert set(portable_attention.__all__) == {
        "__version__",
        "scaled_dot_product_attention",
    }


def test_version_is_nonempty_string():
    assert isinstance(portable_attention.__version__, str)
    assert portable_attention.__version__


def test_metadata_version_matches_dunder():
    # Single source of truth: the installed distribution metadata must equal the
    # __version__ the build backend read from the package.
    assert version("portable-attention") == portable_attention.__version__


def test_signature_mirrors_torch_sdpa():
    params = list(inspect.signature(scaled_dot_product_attention).parameters)
    assert params == ["query", "key", "value", "attn_mask", "is_causal", "scale"]


def test_defaults_match_torch_sdpa():
    defaults = {
        name: p.default
        for name, p in inspect.signature(
            scaled_dot_product_attention
        ).parameters.items()
    }
    assert defaults["attn_mask"] is None
    assert defaults["is_causal"] is False
    assert defaults["scale"] is None
