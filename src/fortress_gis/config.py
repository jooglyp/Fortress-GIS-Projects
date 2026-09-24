"""Repository paths and shared constants.

Paths resolve relative to the repository root so that notebooks under ``demos/``,
tests under ``tests/`` and CLI invocations all agree on where data and exports live.
Override the root with the ``FORTRESS_GIS_ROOT`` environment variable when the package
is installed outside the repository.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Literal

Domain = Literal["hydrology", "oceanography", "airspace", "maritime"]
DOMAINS: Final[tuple[Domain, ...]] = ("hydrology", "oceanography", "airspace", "maritime")


def _resolve_repo_root() -> Path:
    env_root = os.environ.get("FORTRESS_GIS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    here = Path(__file__).resolve()
    # src/fortress_gis/config.py -> repo root is three levels up
    candidate = here.parents[2]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return Path.cwd()


REPO_ROOT: Final[Path] = _resolve_repo_root()
DATA_DIR: Final[Path] = REPO_ROOT / "data"
EXPORTS_DIR: Final[Path] = REPO_ROOT / "exports"
DEMOS_DIR: Final[Path] = REPO_ROOT / "demos"


def domain_data_dir(domain: Domain, *, create: bool = False) -> Path:
    """Return ``data/<domain>`` for one of the supported domains."""
    if domain not in DOMAINS:
        msg = f"Unknown domain {domain!r}; expected one of {DOMAINS}"
        raise ValueError(msg)
    path = DATA_DIR / domain
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def domain_export_dir(domain: Domain, *, create: bool = True) -> Path:
    """Return ``exports/<domain>`` where Kepler.gl HTML and QGIS artifacts are written."""
    if domain not in DOMAINS:
        msg = f"Unknown domain {domain!r}; expected one of {DOMAINS}"
        raise ValueError(msg)
    path = EXPORTS_DIR / domain
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path | str) -> Path:
    """``mkdir -p`` and return the path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
