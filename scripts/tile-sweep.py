#!/usr/bin/env python3
"""Time the Vulkan attention kernel at several tile shapes.

The sizing policy prefers the largest workgroup, then the widest key tile
(see ``portable_attention.tiling``). That order is a hypothesis; this script
measures it. Each tile shape is registered as its own backend and run through
the normal benchmark harness, so the output is the same Markdown table the
cross-backend comparison produces and can be pasted into ``BENCHMARKS.md``.

    python scripts/tile-sweep.py --tiles 16x16,8x16,32x8 --repeats 10

A shape that does not fit the device at a given head dimension takes the CPU
fallback, which would silently turn the sweep into a CPU measurement, so the
device call count is checked after every run and reported per tile.
"""

from __future__ import annotations

import argparse
import sys

from portable_attention import vulkan_available
from portable_attention.benchmark import (
    DEFAULT_SHAPES,
    format_comparison,
    run_comparison,
)
from portable_attention.dispatch import register_backend
from portable_attention.vkbackend import VulkanAttention

DEFAULT_TILES = "16x16,32x8,8x16,16x8,4x16,8x8"


def parse_pair(value: str) -> tuple[int, int]:
    """Parse a ``block_q x block_k`` tile shape."""
    try:
        block_q, block_k = (int(part) for part in value.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a tile shape like 16x16, got {value!r}"
        ) from None
    return block_q, block_k


def parse_shape(value: str) -> tuple[int, int, int, int]:
    """Parse a ``B x H x S x D`` benchmark shape."""
    dims = tuple(int(part) for part in value.lower().split("x"))
    if len(dims) != 4 or any(dim < 1 for dim in dims):
        raise argparse.ArgumentTypeError(
            f"expected four positive dims like 1x8x128x64, got {value!r}"
        )
    return dims[0], dims[1], dims[2], dims[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiles",
        default=DEFAULT_TILES,
        help=f"comma-separated block_q x block_k shapes (default: {DEFAULT_TILES})",
    )
    parser.add_argument(
        "--shapes",
        default=None,
        help="comma-separated BxHxSxD shapes (default: the benchmark shapes)",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--baseline",
        default="fused",
        help="CPU backend the speedup columns compare against (default: fused)",
    )
    args = parser.parse_args(argv)

    if not vulkan_available():
        print("no Vulkan compute device on this host", file=sys.stderr)
        return 1

    tiles = [parse_pair(part) for part in args.tiles.split(",") if part.strip()]
    shapes = (
        list(DEFAULT_SHAPES)
        if args.shapes is None
        else [parse_shape(part) for part in args.shapes.split(",") if part.strip()]
    )

    backends = {
        f"vk {block_q}x{block_k}": VulkanAttention(tile_shape=(block_q, block_k))
        for block_q, block_k in tiles
    }
    for name, backend in backends.items():
        register_backend(name, backend, overwrite=True)

    try:
        results = run_comparison(
            backends=[args.baseline, *backends],
            repeats=args.repeats,
            shapes=shapes,
        )
        print(format_comparison(results, baseline=args.baseline))
        print()
        for name, backend in backends.items():
            served = backend.device_calls
            note = "" if served else "  <- never reached the device"
            print(f"{name}: {served} device calls, limits {backend.limits.name}{note}")
    finally:
        for backend in backends.values():
            backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
