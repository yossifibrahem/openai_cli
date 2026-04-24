"""Configuration — load from file, env vars, and CLI args.

Priority (highest → lowest):
  1. CLI arguments
  2. Environment variables (OPENAI_API_KEY / AI_*)
  3. Config file  (~/.config/openai-cli/config.json)
  4. Hard-coded defaults
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from .utils import console

logger = logging.getLogger(__name__)

CONFIG_FILE = Path.home() / ".config" / "openai-cli" / "config.json"


@dataclass
class Settings:
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    system_prompt: str = "You are a helpful assistant."
    mcp_file: Path = field(default_factory=lambda: Path.cwd() / "mcp.json")
    no_mcp: bool = False

    def __post_init__(self) -> None:
        # JSON and CLI args deliver these as plain strings; ensure Path types.
        if not isinstance(self.mcp_file, Path):
            self.mcp_file = Path(self.mcp_file)

    def save(self) -> None:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "system_prompt": self.system_prompt,
        }
        CONFIG_FILE.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, overrides: dict[str, Any] | None = None) -> "Settings":
        """Build Settings by merging config file + env vars + overrides."""
        data: dict[str, Any] = {}

        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read config file %s: %s", CONFIG_FILE, exc)
                console.print(f"[yellow]Warning:[/yellow] Could not read config file: {exc}")

        # Environment variables win over config file
        if key := os.environ.get("OPENAI_API_KEY"):
            data["api_key"] = key
        for env_key, field_name in [
            ("AI_MODEL", "model"),
            ("AI_BASE_URL", "base_url"),
            ("AI_SYSTEM_PROMPT", "system_prompt"),
        ]:
            if val := os.environ.get(env_key):
                data[field_name] = val

        # CLI overrides win over everything
        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None})

        settings = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

        if not settings.api_key:
            console.print(
                "[red]Error:[/red] No API key found.\n"
                "Set [bold]OPENAI_API_KEY[/bold] or run [bold]ai --setup[/bold]."
            )
            raise SystemExit(1)

        return settings


def config_exists() -> bool:
    return CONFIG_FILE.exists()


def run_setup() -> Settings:
    """Interactive first-run setup wizard."""
    console.print("[bold cyan]OpenAI CLI — Setup[/bold cyan]\n")

    # API key
    env_key = os.environ.get("OPENAI_API_KEY", "")
    if env_key:
        console.print(f"[dim]Found OPENAI_API_KEY in environment.[/dim]")
        inp = console.input("Press Enter to keep it, or type a new one: ").strip()
        api_key = inp or env_key
    else:
        console.print("Get your key at: [dim]https://platform.openai.com/api-keys[/dim]")
        while True:
            api_key = console.input("API key: ").strip()
            if api_key:
                break
            console.print("[red]Cannot be empty.[/red]")

    # Base URL
    console.print("\n[bold]Base URL[/bold]  (Enter = OpenAI default)")
    console.print("  [dim]https://api.openai.com/v1[/dim]  ← default")
    console.print("  [dim]http://localhost:11434/v1[/dim]   ← Ollama")
    url = console.input("Base URL [https://api.openai.com/v1]: ").strip()
    base_url = url or "https://api.openai.com/v1"

    # Model — fetch then pick
    models: list[str] = []
    try:
        with console.status("[cyan]Fetching models…[/cyan]"):
            _client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            page = asyncio.run(_client.models.list())
            models = sorted(m.id for m in page.data)
    except Exception as exc:
        console.print(f"[yellow]Could not fetch models: {exc}[/yellow]")

    model = "gpt-4o"
    if models:
        console.print("\n[bold]Available models:[/bold]")
        for i, m in enumerate(models[:15], 1):
            console.print(f"  [{i}] {m}")
        if len(models) > 15:
            console.print(f"  [dim]… and {len(models) - 15} more[/dim]")
        choice = console.input("\nPick a model [gpt-4o]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            model = models[int(choice) - 1]
        elif choice:
            model = choice
    else:
        model = console.input("\nDefault model [gpt-4o]: ").strip() or "gpt-4o"

    settings = Settings(api_key=api_key, base_url=base_url, model=model)
    settings.save()
    console.print(f"\n[green]✓[/green] Saved to {CONFIG_FILE}\n")
    return settings