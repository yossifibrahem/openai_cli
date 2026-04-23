"""Model management — list and validate OpenAI models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rich.table import Table
from .utils import console

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from .config import Settings

logger = logging.getLogger(__name__)

# Models known to support tool/function calling
TOOL_CAPABLE_PREFIXES = ("gpt-4", "gpt-3.5-turbo", "o1", "o3", "o4")

# Shorthand aliases users can type
MODEL_ALIASES: dict[str, str] = {
    "4o": "gpt-4o",
    "4o-mini": "gpt-4o-mini",
    "4": "gpt-4",
    "4-turbo": "gpt-4-turbo",
    "3.5": "gpt-3.5-turbo",
    "o1": "o1",
    "o1-mini": "o1-mini",
    "o3": "o3",
    "o3-mini": "o3-mini",
    "o4-mini": "o4-mini",
}


class ModelManager:
    """Fetch, cache, and display available models."""

    def __init__(self, settings: "Settings", client: "AsyncOpenAI | None" = None) -> None:
        self._settings = settings
        self._client = client
        self._cached_models: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    async def list_models(self, *, force_refresh: bool = False) -> list[str]:
        """Return sorted list of available model IDs."""
        if self._cached_models and not force_refresh:
            return self._cached_models

        if self._client is None:
            return self._default_models()

        try:
            page = await self._client.models.list()
            ids = sorted(
                (m.id for m in page.data if "gpt" in m.id or m.id.startswith(("o1", "o3", "o4"))),
                key=lambda x: (0 if x.startswith("gpt-4o") else 1 if x.startswith("gpt-4") else 2, x),
            )
            self._cached_models = ids or self._default_models()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch model list: %s", exc)
            self._cached_models = self._default_models()

        return self._cached_models

    async def list_and_display(self) -> None:
        """Print a rich table of available models."""
        with console.status("[cyan]Fetching models…[/cyan]"):
            models = await self.list_models(force_refresh=True)

        table = Table(title="Available Models", show_header=True, header_style="bold cyan")
        table.add_column("Model ID", style="cyan")
        table.add_column("Tools", justify="center")
        table.add_column("Alias", style="dim")

        # Build reverse alias map
        rev_aliases: dict[str, str] = {v: k for k, v in MODEL_ALIASES.items()}

        for model_id in models:
            tools = "✓" if self._supports_tools(model_id) else "—"
            alias = rev_aliases.get(model_id, "")
            marker = " ← current" if model_id == self._settings.model else ""
            table.add_row(model_id + marker, tools, alias)

        console.print(table)
        console.print(f"\n[dim]Current model: [bold]{self._settings.model}[/bold][/dim]")

    def resolve_alias(self, name: str) -> str:
        """Expand an alias like '4o' → 'gpt-4o'."""
        return MODEL_ALIASES.get(name.lower(), name)

    async def validate_model(self, name: str) -> str:
        """Resolve alias and verify the model exists. Returns the final name."""
        resolved = self.resolve_alias(name)
        models = await self.list_models()
        if resolved not in models:
            # Soft warning — non-OpenAI deployments may have custom models
            logger.warning("Model %r not in listed models (may still work)", resolved)
        return resolved

    @staticmethod
    def _supports_tools(model_id: str) -> bool:
        return any(model_id.startswith(p) for p in TOOL_CAPABLE_PREFIXES)

    @staticmethod
    def _default_models() -> list[str]:
        """Fallback list when the API is unreachable or returns nothing."""
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
            "o3-mini",
            "o4-mini",
        ]