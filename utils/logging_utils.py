"""Logging utilities for MindCare.

Provides a unified, clean logger format that works smoothly across console
and API server components without polluting stdout with noisy traces.
"""

import logging
import sys

_LOGGER_NAME = "mindcare"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger
