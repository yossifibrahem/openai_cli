"""Utility helpers — including the application-wide shared Rich Console.

All modules must import ``console`` from here instead of creating their own
``Console()`` instances.  Multiple Console instances writing to the same
stdout bypass Rich's Live cursor-management, causing text to appear twice
(once from the Live renderer, once from the foreign console write).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console

# ── Single shared console for the whole application ──────────────────────────
# Import this everywhere:  from .utils import console
console: Console = Console()


def setup_logging(level: str = "WARNING", log_file: Path | None = None) -> None:
    """Configure root logger for the application."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

    # Silence noisy third-party loggers unless DEBUG
    if level != "DEBUG":
        for noisy in ("httpx", "httpcore", "openai", "anyio"):
            logging.getLogger(noisy).setLevel(logging.WARNING)