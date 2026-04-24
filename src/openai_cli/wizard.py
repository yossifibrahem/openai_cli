"""Interactive first-run configuration wizard."""

from __future__ import annotations

import os

from openai import AsyncOpenAI

# BUG FIX: previously redefined AVAILABLE_THEMES locally, risking divergence
# from the canonical list in config.py. Import it instead.
from .config import DEFAULT_CONFIG_FILE, AVAILABLE_THEMES, Settings, save_config
from .utils import console


def _prompt_api_key() -> str:
    """Prompt for API key, checking env var first."""
    env_key = os.environ.get("OPENAI_API_KEY", "")
    if env_key:
        console.print("[dim]Using API key from OPENAI_API_KEY environment variable.[/dim]")
        console.print("[dim]Press Enter to keep it, or type a new one:[/dim] ")
        user_input = console.input().strip()
        return user_input if user_input else env_key

    console.print("[bold cyan]Welcome to OpenAI CLI![/bold cyan]")
    console.print()
    console.print("It looks like this is your first run. Let's set things up.")
    console.print()
    console.print("[bold]Step 1: API Key[/bold]")
    console.print("Get your API key from: [dim]https://platform.openai.com/api-keys[/dim]")
    console.print()
    while True:
        api_key = console.input("Enter your API key: ").strip()
        if api_key:
            return api_key
        console.print("[red]API key cannot be empty. Please try again.[/red]")


def _prompt_base_url() -> str:
    """Prompt for API base URL."""
    console.print()
    console.print("[bold]Step 2: Base URL[/bold]")
    console.print("Leave as default for OpenAI API, or enter your own endpoint.")
    console.print("[dim]Examples:[/dim]")
    console.print("  [dim]https://api.openai.com/v1[/dim]      ← OpenAI (default)")
    console.print("  [dim]http://localhost:11434/v1[/dim]        ← Ollama")
    console.print("  [dim]https://api.mistral.ai/v1[/dim]       ← Mistral")

    while True:
        base_url = console.input("Base URL [https://api.openai.com/v1]: ").strip()
        if not base_url:
            return "https://api.openai.com/v1"
        if base_url.startswith(("http://", "https://")):
            return base_url
        console.print("[red]URL must start with http:// or https://[/red]")


async def _fetch_models(api_key: str, base_url: str) -> list[str]:
    """Fetch available models from the API, returning an empty list on failure."""
    try:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        page = await client.models.list()
        return sorted(m.id for m in page.data)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Could not fetch models: {exc}[/yellow]")
        return []


async def _prompt_model(api_key: str, base_url: str) -> str:
    """Prompt for model selection with live model list."""
    console.print()
    console.print("[bold]Step 3: Choose a Model[/bold]")

    with console.status("[cyan]Fetching available models…[/cyan]"):
        models = await _fetch_models(api_key, base_url)

    if not models:
        console.print("[yellow]Could not retrieve model list. Using default.[/yellow]")
        console.print("[dim]You can change this later with /model command.[/dim]")
        return "gpt-4o"

    # Sort preferred models to the top of the displayed list.
    preferred = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    seen: set[str] = set()
    shown: list[str] = []

    for pref in preferred:
        for m in models:
            if m not in seen and (m == pref or m.startswith(pref + "-")):
                shown.append(m)
                seen.add(m)

    for m in models:
        if m not in seen:
            shown.append(m)
            seen.add(m)

    console.print("[bold]Available models:[/bold]")
    console.print()
    for i, model_id in enumerate(shown[:15], 1):
        console.print(f"  [{i}] {model_id}")
    if len(shown) > 15:
        console.print(f"  [dim]... and {len(shown) - 15} more[/dim]")
    console.print()
    console.print("[dim]Enter the number or name (default: 1 = gpt-4o)[/dim]")

    while True:
        choice = console.input("Pick a model: ").strip()
        if not choice:
            return "gpt-4o"
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(shown):
                return shown[idx]
            console.print(f"[red]Please enter a number between 1 and {len(shown)}[/red]")
        elif choice in models:
            return choice
        else:
            # Partial prefix match against the displayed list
            for m in shown:
                if m.startswith(choice.lower()):
                    return m
            console.print("[red]Model not found. Try the number or pick from the list.[/red]")


def _prompt_system_prompt() -> str:
    """Prompt for system prompt."""
    console.print()
    console.print("[bold]Step 4: System Prompt[/bold]")
    console.print("This defines how the AI assistant behaves.")
    console.print("[dim]Examples:[/dim]")
    console.print("  [dim]You are a helpful assistant.[/dim]")
    console.print("  [dim]You are a pirate who speaks in Old English.[/dim]")
    console.print("  [dim]You are a code reviewer. Focus on security issues.[/dim]")
    console.print()

    default = "You are a helpful assistant."
    system_prompt = console.input(f"System prompt [{default}]: ").strip()
    return system_prompt if system_prompt else default


def _prompt_theme() -> str:
    """Prompt for theme selection."""
    console.print()
    console.print("[bold]Step 5: Theme[/bold]")
    console.print("Choose a color theme for the terminal UI.")
    console.print()

    for i, theme in enumerate(AVAILABLE_THEMES, 1):
        console.print(f"  [{i}] {theme}")
    console.print()
    console.print("[dim]Enter the number (default: 1 = monokai)[/dim]")

    while True:
        choice = console.input("Pick a theme: ").strip()
        if not choice:
            return "monokai"
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(AVAILABLE_THEMES):
                return AVAILABLE_THEMES[idx]
            console.print(f"[red]Please enter a number between 1 and {len(AVAILABLE_THEMES)}[/red]")
        elif choice in AVAILABLE_THEMES:
            return choice
        console.print("[red]Invalid theme. Try a number or name from the list.[/red]")


async def run_wizard() -> Settings:
    """Run the interactive first-run wizard and return the new Settings."""
    api_key = _prompt_api_key()
    base_url = _prompt_base_url()
    model = await _prompt_model(api_key, base_url)
    system_prompt = _prompt_system_prompt()
    theme = _prompt_theme()

    console.print()
    console.print("[bold]Step 6: Saving configuration…[/bold]")

    settings_obj = Settings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        system_prompt=system_prompt,
        theme=theme,
    )

    DEFAULT_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_config(settings_obj)

    console.print(f"[green]Configuration saved to {DEFAULT_CONFIG_FILE}[/green]")
    console.print()
    console.print("[bold cyan]Setup complete! You can now start chatting.[/bold cyan]")
    console.print("[dim]Type /help in the chat for available commands.[/dim]")
    console.print()

    return settings_obj