"""Model management — list available models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .utils import console

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, client: "AsyncOpenAI") -> None:
        self._client = client
        self._cache: list[str] = []

    async def list_models(self, *, force_refresh: bool = False) -> list[str]:
        if self._cache and not force_refresh:
            return self._cache
        try:
            page = await self._client.models.list()
            self._cache = sorted(m.id for m in page.data)
        except Exception as exc:
            logger.warning("Could not fetch models: %s", exc)
        return self._cache

    async def list_and_display(self) -> None:
        with console.status("[cyan]Fetching models…[/cyan]"):
            models = await self.list_models(force_refresh=True)
        if not models:
            console.print("[yellow]No models returned.[/yellow]")
            return
        for m in models:
            console.print(f"  [dim]○[/dim] {m}")
