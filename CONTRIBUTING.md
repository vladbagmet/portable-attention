# Contributing

Thanks for your interest — contributions are welcome, including at this early
(pre-MVP) stage.

## Ground rules

- **Correctness first.** Attention kernels live or die on numerical accuracy:
  every behavioral change needs tests, and shape/dtype edge cases matter.
- **CI must be green** before merge; lint and typecheck gates are not advisory.
- **Portability floor.** Changes must not break the project's hard floor:
  every release runs end-to-end on an $80 computer. If your change raises the
  hardware bar, it needs an explicit discussion first.
- **Small PRs win.** One cohesive change per PR, with an honest description of
  what was tested and how.

## Development setup

The project is CPU-first and needs no GPU. With [uv](https://docs.astral.sh/uv/):

```sh
uv venv
uv pip install -e ".[dev]"
```

Run the same gate CI runs, before you push — a single command:

```sh
./scripts/check.sh
```

It runs, in order: `ruff check` (lint), `ruff format --check` (formatting),
`pyright` (strict typecheck), `bandit` (security), `pip-audit` (dependency
vulnerabilities), and `coverage run -m pytest` + `coverage report` (tests with
a coverage floor). Every step must pass; do not weaken a check to get green.

`ruff` is pinned to an exact version in the `dev` extra. Its formatter picks up
new constructs across minor releases, and a floating range means the version in
your environment and the one CI resolves can disagree about the same file. If
you update the pin, run `ruff format .` and commit the result together with the
bump. Re-run `uv pip install -e ".[dev]"` after pulling a change to the pin so
your venv follows.

## Running the Vulkan tests

The Vulkan device tests skip themselves where the loader finds no
compute-capable device, so a machine without a GPU still runs the rest of the
suite. To run them:

```sh
./scripts/vulkan-conformance.sh
```

It reports the devices the loader enumerated and the tile limits of the one it
opened, exits non-zero when there is none — a skipped device test is not a
pass — and then runs the Vulkan test modules. Any arguments you pass go to
pytest instead of the default module list.

CI runs that script against Mesa's `lavapipe` (`mesa-vulkan-drivers`), a
software Vulkan 1.3 implementation, so the shader, the dispatch path and the
conformance kit are checked on every pull request without a GPU runner. It is a
correctness target only; benchmark numbers come from real devices. Where
several drivers are installed, choose one with `VK_DRIVER_FILES`
(`VK_ICD_FILENAMES` on loaders older than 1.3.207), naming a file from
`/usr/share/vulkan/icd.d/` — distributions differ on whether the name carries
an architecture suffix:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
  ./scripts/vulkan-conformance.sh
```

## Sweeping tile shapes

When a kernel or the sizing policy changes, `scripts/tile-sweep.py` times the
device at several `block_q x block_k` shapes and prints a Markdown table:

```sh
python scripts/tile-sweep.py --tiles 16x16,32x8,8x16 --repeats 8
```

It registers one backend per tile shape and reports the device call count for
each, so a shape that silently fell back to the CPU is visible instead of
turning up as a suspiciously good latency.

## Verifying the Metal backend

No CI runner the project controls has an Apple GPU, so Metal changes cannot be
verified by the same machinery as everything else. A Metal pull request is
signed off by a human running, on a Mac:

```sh
uv pip install -e ".[dev,metal]"
./scripts/verify-metal.sh
```

The script prints the device name, its threadgroup/SIMD limits, whether the MSL
kernel compiled through the Metal runtime compiler, conformance results against
the reference oracle, and benchmark numbers. Paste that report into the PR
as-is. Until it appears, describe the change as compiled but not run on
hardware; never claim a result that no report backs.

## Reviews

Every non-draft PR gets an automatic advisory review from CodeRabbit plus a
maintainer review. Address correctness findings; style is enforced by the
linters in CI, not by review comments.

## License

Apache-2.0. By submitting a contribution you agree it is licensed under the
project's Apache-2.0 license.
