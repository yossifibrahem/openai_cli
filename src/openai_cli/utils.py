"""Shared Rich console — import from here, never create Console() elsewhere."""

from __future__ import annotations

from rich.console import Console

console: Console = Console()
