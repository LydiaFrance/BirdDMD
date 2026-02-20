See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://scientific-python-cookie.readthedocs.io/guide/intro

# Setting up a development environment manually

## Git LFS

This repository uses [Git LFS](https://git-lfs.com/) to store large files
(animated GIFs in `notebooks/figures/`). Install Git LFS before cloning:

```bash
# macOS
brew install git-lfs

# Ubuntu / Debian
sudo apt install git-lfs

# Then, one-time setup
git lfs install
```


Without Git LFS installed, the GIF files will appear as small text pointer
files instead of the actual images.

## Python environment

Using [uv](https://docs.astral.sh/uv/) (preferred):

```bash
uv sync --extra dev
```

Or with pip:

```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -v -e ".[dev]"
```

# Post setup

You should prepare pre-commit, which will help you by checking that commits pass
required checks:

```bash
# Install pre-commit
uv tool install pre-commit  # or: pip install pre-commit / brew install pre-commit

# Install the git hook
pre-commit install
```

You can also/alternatively run `pre-commit run` (changes only) or
`pre-commit run --all-files` to check even without installing the hook.

# Testing

Use pytest to run the unit checks:

```bash
uv run pytest
```

Or without uv:

```bash
pytest
```

# Coverage

Use pytest-cov to generate coverage reports:

```bash
uv run pytest --cov=birddmd
```

Or without uv:

```bash
pytest --cov=birddmd
```

# Pre-commit

This project uses pre-commit for all style checking. Run:

```bash
uvx pre-commit run -a
```

Or without uv:

```bash
pre-commit run -a
```

to check all files.
