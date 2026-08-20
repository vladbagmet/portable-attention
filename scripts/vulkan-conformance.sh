#!/usr/bin/env bash
# Run the Vulkan tests against whatever Vulkan implementation the loader finds.
#
# The suite skips its device tests when no Vulkan device is present, which is
# right for a laptop but useless as a gate: a job that installs a driver and
# then skips everything passes silently. So this script probes first and exits
# non-zero when nothing compute-capable turned up, and only then runs pytest.
#
#   ./scripts/vulkan-conformance.sh                 # device tests
#   ./scripts/vulkan-conformance.sh tests -q        # or any pytest arguments
#
# Point the loader at one driver with VK_DRIVER_FILES (VK_ICD_FILENAMES on
# loaders older than 1.3.207) to choose between several installed ICDs, naming
# a file from /usr/share/vulkan/icd.d/ — the exact name is the distribution's,
# e.g. lvp_icd.json or lvp_icd.x86_64.json for Mesa's software rasteriser:
#
#   VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
#     ./scripts/vulkan-conformance.sh
set -euo pipefail

RUN="${RUN:-uv run}"
VK_TESTS=(
  tests/test_vulkan_detection.py
  tests/test_vkcompute.py
  tests/test_vk_attention.py
  tests/test_vk_backend.py
)

$RUN python - <<'PY'
import sys

from portable_attention import available_backends, detect_vulkan
from portable_attention.vkcompute import VulkanContext, VulkanError

cap = detect_vulkan()
print(f"loader: {cap.loader or 'not found'}")
print()
print("| device | API | compute |")
print("| --- | --- | --- |")
for device in cap.devices:
    print(f"| {device.name} | {device.api_version} | {device.compute} |")
print()

if not cap.available:
    print(f"no Vulkan compute device: {cap.reason}", file=sys.stderr)
    sys.exit(1)

# The tile plan comes from the limits the opened device reports, so print them:
# a job running on a different implementation than expected shows up here.
try:
    with VulkanContext.open() as ctx:
        limits = ctx.tile_limits
        print(f"opened: {ctx.device_name} (Vulkan {ctx.api_version})")
        print(
            f"tile limits: {limits.shared_memory_bytes} B shared, "
            f"{limits.max_threads_per_group} invocations, "
            f"SIMD {limits.simd_width}"
        )
except VulkanError as exc:
    print(f"device enumerated but would not open: {exc}", file=sys.stderr)
    sys.exit(1)

print(f"backends: {', '.join(available_backends())}")
if "vulkan" not in available_backends():
    print("the vulkan backend did not register on this host", file=sys.stderr)
    sys.exit(1)
PY

if [ "$#" -gt 0 ]; then
  exec $RUN pytest "$@"
fi
exec $RUN pytest "${VK_TESTS[@]}"
