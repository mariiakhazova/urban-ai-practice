# AGENTS.md

Boilerplate for IDU research Python packages, not a finished library. Placeholder identifiers (`my_package`, author "Vasilii Starikov", README links, Sphinx project "MyPackage") are meant to be renamed when adopted; do it consistently across `pyproject.toml`, `my_package/__init__.py`, `docs/source/conf.py`, and `README.rst`.

## Commands

- Editable install with dev tools: `pip install -e '.[dev]'` (`make install-dev`)
- Test: `pytest tests` (`make tests`); coverage: `pytest tests --cov` (`make tests-cov`)
- Lint: `pylint my_package` (`make lint`); format: `isort my_package` then `black my_package` (`make format`)
- Docs: `pip install -e '.[docs]'`, then `sphinx-build docs/source docs/build` (requires pandoc)
- On Windows prefer the direct commands above: the Makefile assumes POSIX (`python3`, `rm -rf`)

## Gotchas

- Makefile/CI mismatches: `make install-test` installs `.[test]` but the real extra is `tests` (use `pip install -e '.[tests]'`); `tests.yml` calls `make test` but the Makefile target is `tests` — run `pytest tests` locally.
- Version is derived by setuptools_scm from git history/tags (post-release scheme) — never hardcode it; builds need full history (`fetch-depth: 0` in CI). `my_package/__init__.py` calls `importlib.metadata.version("my_package")`, so the package must be pip-installed before `import my_package` works.
- `pyproject.toml` declares license file `LICENCE`, but the actual file is `LICENSE`.
- Style: 120-char lines (black/isort/pylint); isort leaves 2 blank lines after imports; pre-commit fixes whitespace/EOF and runs black.
- Release flow: pushing a `v*` tag triggers `pypi.yml` (release tags from `main`, pre-release `.dev`/`a`/`b`/`rc` from `develop`). `documentation.yml` deploys docs to the `documentation` branch only on push to `main`.

## Layout

- `my_package/` — library code (`my_module.py` is placeholder demo code)
- `tests/` — pytest suite
- `examples/` — demo notebook + `data/`
- `docs/source/` — Sphinx; `index.rst` includes README slices between the markers `.. logo-start/end` and `.. readme-start/end` — keep those markers when editing README
- Generated/ignored: `.venv/`, `my_package.egg-info/`, `dist/`, `docs/build/`
