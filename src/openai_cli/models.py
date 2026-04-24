"""Model management — list and validate models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .utils import console

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from .config import Settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Fetch, cache, and display available models."""

    def __init__(self, settings: "Settings", client: "AsyncOpenAI | None" = None) -> None:
        self._settings = settings
        self._client = client
        self._cached_models: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    async def list_models(self, *, force_refresh: bool = False) -> list[str]:
        """Return a sorted list of available model IDs, using a cache by default."""
        if self._cached_models and not force_refresh:
            return self._cached_models

        if self._client is None:
            return []

        try:
            page = await self._client.models.list()
            self._cached_models = sorted(m.id for m in page.data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch model list: %s", exc)

        return self._cached_models

    async def list_and_display(self) -> None:
        """Print available models, marking the currently active one."""
        with console.status("[cyan]Fetching models…[/cyan]"):
            models = await self.list_models(force_refresh=True)

        if not models:
            console.print("[yellow]No models returned by the API.[/yellow]")
            return

        current = self._settings.model
        for model_id in models:
            if model_id == current:
                console.print(f"  [bold cyan]● {model_id}[/bold cyan]  [dim]← current[/dim]")
            else:
                console.print(f"  [dim]○[/dim] {model_id}")

    async def validate_model(self, name: str) -> str:
        """Verify the model exists; returns the name unchanged.

        Issues a soft warning rather than raising — non-OpenAI deployments
        may serve custom model IDs that won't appear in the listed models.
        """
        models = await self.list_models()
        if models and name not in models:
            logger.warning("Model %r not in listed models (may still work)", name)
        return name