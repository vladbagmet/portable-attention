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

## Reviews

Every non-draft PR gets an automatic advisory review from CodeRabbit plus a
maintainer review. Address correctness findings; style is enforced by the
linters in CI, not by review comments.

## License

Apache-2.0. By submitting a contribution you agree it is licensed under the
project's Apache-2.0 license.
