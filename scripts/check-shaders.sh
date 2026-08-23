#!/usr/bin/env bash
# Recompile every committed GLSL compute shader and compare it byte for byte
# with the .spv checked in beside it.
#
# The .spv files are build artifacts that ship in the wheel, and nothing in the
# test suite reads the .comp sources: editing a shader without recompiling
# leaves the whole gate green while the kernel that actually runs is the old
# one. Run this after touching any .comp.
#
#   ./scripts/check-shaders.sh
#
# Byte equality is a property of one compiler version. The artifacts in the
# tree were produced by glslangValidator 12.0.0; a different version can emit
# a different-but-valid module, so a mismatch means "recompile and commit the
# result", not necessarily "the source and the artifact disagree". The version
# in use is printed to make that distinction possible.
set -euo pipefail

GLSLANG="${GLSLANG:-glslangValidator}"

if ! command -v "$GLSLANG" >/dev/null 2>&1; then
  echo "$GLSLANG not found; install glslang-tools (Debian) or set GLSLANG" >&2
  exit 1
fi

cd "$(dirname "$0")/.."
"$GLSLANG" --version | head -1

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

status=0
while IFS= read -r comp; do
  spv="${comp%.comp}.spv"
  if [ ! -f "$spv" ]; then
    echo "MISSING  $spv (no artifact committed for $comp)" >&2
    status=1
    continue
  fi
  out="$tmp/$(echo "$comp" | tr / _).spv"
  if ! "$GLSLANG" -V --target-env vulkan1.0 -S comp -o "$out" "$comp" >"$tmp/log" 2>&1; then
    echo "FAILED   $comp does not compile" >&2
    cat "$tmp/log" >&2
    status=1
    continue
  fi
  if cmp -s "$out" "$spv"; then
    echo "ok       $spv"
  else
    echo "STALE    $spv differs from a fresh compile of $comp" >&2
    status=1
  fi
done < <(find src tests -name '*.comp' | sort)

exit "$status"
