"""Freeze the public API surface and the version single-source-of-truth.

These tests are a contract, not behavior checks: they fail loudly if the
exported names or the ``scaled_dot_product_attention`` signature drift, or if
the packaged metadata version stops matching ``__version__``. The signature is
frozen to mirror ``torch.nn.functional.scaled_dot_product_attention``.
"""

from __future__ import annotations

import inspect
from importlib.metadata import version

import numpy as np
import pytest

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
    # Order and names match torch.nn.functional.scaled_dot_product_attention:
    # (query, key, value, attn_mask, dropout_p, is_causal, *, scale, enable_gqa)
    params = list(inspect.signature(scaled_dot_product_attention).parameters)
    assert params == [
        "query",
        "key",
        "value",
        "attn_mask",
        "dropout_p",
        "is_causal",
        "scale",
        "enable_gqa",
    ]


def test_scale_and_enable_gqa_are_keyword_only():
    params = inspect.signature(scaled_dot_product_attention).parameters
    assert params["scale"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["enable_gqa"].kind is inspect.Parameter.KEYWORD_ONLY


def test_defaults_match_torch_sdpa():
    defaults = {
        name: p.default
        for name, p in inspect.signature(
            scaled_dot_product_attention
        ).parameters.items()
    }
    assert defaults["attn_mask"] is None
    assert defaults["dropout_p"] == 0.0
    assert defaults["is_causal"] is False
    assert defaults["scale"] is None
    assert defaults["enable_gqa"] is False


def test_unsupported_dropout_raises():
    q = np.zeros((2, 4))
    k = np.zeros((3, 4))
    v = np.zeros((3, 5))
    with pytest.raises(NotImplementedError, match="dropout_p"):
        scaled_dot_product_attention(q, k, v, dropout_p=0.1)


def test_unsupported_gqa_raises():
    q = np.zeros((2, 4))
    k = np.zeros((3, 4))
    v = np.zeros((3, 5))
    with pytest.raises(NotImplementedError, match="enable_gqa"):
        scaled_dot_product_attention(q, k, v, enable_gqa=True)
