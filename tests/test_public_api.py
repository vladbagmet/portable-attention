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
    # The SDPA entry point and version SoT are the load-bearing surface; the
    # backend-registry and conformance-kit names are the M1 additive extension,
    # and the Vulkan detection plus tile-sizing helpers are the M2 groundwork
    # additive extension (the drop-in SDPA signature itself stays frozen to
    # torch, verified below).
    assert set(portable_attention.__all__) == {
        "ConformanceCase",
        "ConformanceResult",
        "DeviceLimits",
        "SdpaBackend",
        "TilePlan",
        "TileSizingError",
        "VulkanBuffer",
        "VulkanCapability",
        "VulkanContext",
        "VulkanDevice",
        "VulkanError",
        "VulkanPipeline",
        "__version__",
        "assert_conforms",
        "available_backends",
        "blocked_attention",
        "check_backend",
        "conformance_cases",
        "detect_vulkan",
        "get_backend",
        "plan_tiles",
        "register_backend",
        "scaled_dot_product_attention",
        "shared_memory_bytes_for",
        "vulkan_available",
    }


def test_every_exported_name_is_bound():
    # A name can be listed in ``__all__`` yet be missing from the module (a typo
    # in the export list), which breaks ``from portable_attention import *`` and
    # top-level attribute access. Verify each declared export actually resolves.
    for name in portable_attention.__all__:
        assert hasattr(portable_attention, name), f"{name!r} is exported but unbound"


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


def test_gqa_without_head_dim_raises():
    # enable_gqa needs a head axis; 2-D inputs have nothing to group over.
    q = np.zeros((2, 4))
    k = np.zeros((3, 4))
    v = np.zeros((3, 5))
    with pytest.raises(ValueError, match="head dimension"):
        scaled_dot_product_attention(q, k, v, enable_gqa=True)


def test_gqa_runs_end_to_end():
    # Grouped-query attention through the public entry point: 8 query heads
    # share 2 key/value heads, output keeps the query head count.
    rng = np.random.default_rng(12)
    q = rng.standard_normal((2, 8, 5, 16))
    k = rng.standard_normal((2, 2, 7, 16))
    v = rng.standard_normal((2, 2, 7, 4))
    out = scaled_dot_product_attention(q, k, v, enable_gqa=True)
    assert out.shape == (2, 8, 5, 4)
