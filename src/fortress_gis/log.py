"""Logging setup shared by the CLI, notebooks and tests."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def init(level: int = logging.INFO, *, quiet_libraries: bool = True) -> logging.Logger:
    """Configure the ``fortress_gis`` logger once and return it.

    Idempotent: calling it repeatedly (e.g. from several notebook cells) does not stack handlers.
    """
    logger = logging.getLogger("fortress_gis")
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    logger.propagate = False
    if quiet_libraries:
        for noisy in ("fiona", "rasterio", "pyogrio", "distributed", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Child logger under the ``fortress_gis`` namespace."""
    return logging.getLogger(name if name.startswith("fortress_gis") else f"fortress_gis.{name}")
