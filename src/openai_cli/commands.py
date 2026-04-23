"""Slash command registry.

Each command is a dataclass + async handler function.
The registry is used both to execute commands and to feed the autocompleter.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from rich.table import Table
from .utils import console

if TYPE_CHECKING:
    from .chat import ChatSession

logger = logging.getLogger(__name__)

AsyncHandler = Callable[["CommandContext"], Coroutine[Any, Any, None]]


@dataclass
class SlashCommand:
    name: str                      # e.g. "model"
    description: str
    usage: str                     # e.g. "/model <name>"
    handler: AsyncHandler
    aliases: list[str] = field(default_factory=list)
    completer_choices: list[str] = field(default_factory=list)  # dynamic — filled at runtime


@dataclass
class CommandContext:
    session: "ChatSession"
    args: str                      # everything after the command name


# ── Registry ──────────────────────────────────────────────────────────────────


class CommandRegistry:
    """Holds all registered slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(self, cmd: SlashCommand) -> None:
        self._commands[cmd.name] = cmd
        for alias in cmd.aliases:
            self._commands[alias] = cmd

    def get(self, name: str) -> SlashCommand | None:
        return self._commands.get(name.lstrip("/").split()[0].lower())

    def all_names(self) -> list[str]:
        """Unique command names (excluding aliases that point to the same cmd)."""
        seen: set[str] = set()
        result: list[str] = []
        for cmd in self._commands.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                result.append(cmd.name)
        return sorted(result)

    def all_commands(self) -> list[SlashCommand]:
        seen: set[str] = set()
        result: list[SlashCommand] = []
        for cmd in self._commands.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                result.append(cmd)
        return sorted(result, key=lambda c: c.name)

    async def execute(self, raw: str, session: "ChatSession") -> bool:
        """Parse and execute a slash command. Returns True if handled."""
        stripped = raw.lstrip("/").strip()
        parts = stripped.split(None, 1)
        name = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        cmd = self.get(name)
        if cmd is None:
            console.print(f"[red]Unknown command:[/red] /{name}  (try [bold]/help[/bold])")
            return True

        ctx = CommandContext(session=session, args=args)
        try:
            await cmd.handler(ctx)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Command /%s failed", name)
            console.print(f"[red]Command error:[/red] {exc}")
        return True


# ── Handlers ──────────────────────────────────────────────────────────────────


async def _cmd_help(ctx: CommandContext) -> None:
    table = Table(title="Slash Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="cyan", width=28)
    table.add_column("Description", style="white")

    for cmd in ctx.session.registry.all_commands():
        aliases = f"  [dim]({', '.join('/' + a for a in cmd.aliases)})[/dim]" if cmd.aliases else ""
        table.add_row(cmd.usage + aliases, cmd.description)

    console.print(table)


async def _cmd_model(ctx: CommandContext) -> None:
    session = ctx.session
    if not ctx.args:
        console.print(f"Current model: [bold cyan]{session.model}[/bold cyan]")
        console.print("Usage: [bold]/model <name>[/bold] or [bold]/models[/bold] to list all")
        return

    new_model = await session.model_manager.validate_model(ctx.args.strip())
    session.model = new_model
    console.print(f"[green]✓[/green] Switched to [bold cyan]{new_model}[/bold cyan]")


async def _cmd_models(ctx: CommandContext) -> None:
    await ctx.session.model_manager.list_and_display()


async def _cmd_clear(ctx: CommandContext) -> None:
    ctx.session.clear_history()
    console.clear()
    console.print(ctx.session._welcome_banner())


async def _cmd_system(ctx: CommandContext) -> None:
    if not ctx.args:
        console.print(f"System prompt:\n[dim]{ctx.session.system_prompt}[/dim]")
        return
    ctx.session.system_prompt = ctx.args.strip()
    ctx.session.clear_history()
    console.print("[green]✓[/green] System prompt updated (history cleared).")


async def _cmd_save(ctx: CommandContext) -> None:
    session = ctx.session
    if not session.history:
        console.print("[yellow]Nothing to save.[/yellow]")
        return

    filename = ctx.args.strip() or f"chat_{datetime.now():%Y%m%d_%H%M%S}.json"
    if not filename.endswith(".json"):
        filename += ".json"

    save_dir = session.settings.history_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename

    payload = {
        "model": session.model,
        "system_prompt": session.system_prompt,
        "created_at": datetime.now().isoformat(),
        "messages": session.history,
    }
    save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    console.print(f"[green]✓[/green] Saved to [bold]{save_path}[/bold]")


async def _cmd_load(ctx: CommandContext) -> None:
    session = ctx.session
    path_str = ctx.args.strip()

    if not path_str:
        # List saved conversations
        history_dir = session.settings.history_dir
        if not history_dir.exists():
            console.print("[dim]No saved conversations.[/dim]")
            return
        files = sorted(history_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            console.print("[dim]No saved conversations.[/dim]")
            return
        console.print("[cyan]Saved conversations:[/cyan]")
        for i, f in enumerate(files[:10], 1):
            console.print(f"  {i}. [bold]{f.name}[/bold]  [dim]{f.stat().st_size // 1024}KB[/dim]")
        return

    # Try as absolute path, then relative to history_dir
    path = Path(path_str)
    if not path.is_absolute():
        path = session.settings.history_dir / path_str

    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        return

    data = json.loads(path.read_text())
    session.model = data.get("model", session.model)
    session.system_prompt = data.get("system_prompt", session.system_prompt)
    session.history = data.get("messages", [])
    console.print(
        f"[green]✓[/green] Loaded [bold]{path.name}[/bold] "
        f"({len(session.history)} messages, model: {session.model})"
    )


async def _cmd_history(ctx: CommandContext) -> None:
    session = ctx.session
    if not session.history:
        console.print("[dim]No messages in history.[/dim]")
        return

    n = int(ctx.args.strip()) if ctx.args.strip().isdigit() else len(session.history)
    recent = session.history[-n:]

    for msg in recent:
        role = msg["role"]
        content = str(msg.get("content") or "")
        if role == "assistant":
            label = "[cyan]Assistant[/cyan]"
        elif role == "user":
            label = "[green]You[/green]"
        else:
            label = f"[yellow]{role.title()}[/yellow]"

        snippet = content[:120].replace("\n", " ")
        if len(content) > 120:
            snippet += "…"
        console.print(f"  {label}: {snippet}")

    console.print(f"\n[dim]{len(session.history)} message(s) in context window[/dim]")


async def _cmd_retry(ctx: CommandContext) -> None:
    """Re-send the last user message."""
    session = ctx.session
    # Find last user message
    last_user = None
    for msg in reversed(session.history):
        if msg["role"] == "user":
            last_user = msg["content"]
            break

    if last_user is None:
        console.print("[yellow]No previous user message to retry.[/yellow]")
        return

    # Remove last assistant response if present
    if session.history and session.history[-1]["role"] == "assistant":
        session.history.pop()
    if session.history and session.history[-1]["role"] == "user":
        session.history.pop()

    console.print(f"[dim]Retrying: {str(last_user)[:60]}…[/dim]")
    await session.send_message(str(last_user), from_retry=True)


async def _cmd_copy(ctx: CommandContext) -> None:
    """Copy last assistant response to clipboard."""
    session = ctx.session
    for msg in reversed(session.history):
        if msg["role"] == "assistant":
            content = msg.get("content") or ""
            try:
                import pyperclip
                pyperclip.copy(str(content))
                console.print("[green]✓[/green] Copied to clipboard.")
            except ImportError:
                console.print("[yellow]pyperclip not installed.[/yellow]")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Could not copy:[/red] {exc}")
            return
    console.print("[yellow]No assistant message found.[/yellow]")


async def _cmd_tokens(ctx: CommandContext) -> None:
    session = ctx.session
    console.print(
        f"Total tokens used this session: [bold cyan]{session.total_tokens:,}[/bold cyan]\n"
        f"Messages in context: [bold]{len(session.history)}[/bold] / "
        f"[dim]{session.settings.context_window}[/dim]"
    )


async def _cmd_temp(ctx: CommandContext) -> None:
    if not ctx.args.strip():
        console.print(f"Temperature: [bold]{ctx.session.temperature}[/bold]")
        return
    try:
        val = float(ctx.args.strip())
        if not 0.0 <= val <= 2.0:
            raise ValueError
        ctx.session.temperature = val
        console.print(f"[green]✓[/green] Temperature set to [bold]{val}[/bold]")
    except ValueError:
        console.print("[red]Temperature must be a float between 0.0 and 2.0[/red]")


async def _cmd_mcp(ctx: CommandContext) -> None:
    ctx.session.mcp_manager.display_servers()


async def _cmd_export(ctx: CommandContext) -> None:
    """Export conversation as Markdown."""
    session = ctx.session
    if not session.history:
        console.print("[yellow]Nothing to export.[/yellow]")
        return

    filename = ctx.args.strip() or f"chat_{datetime.now():%Y%m%d_%H%M%S}.md"
    if not filename.endswith(".md"):
        filename += ".md"

    lines = [f"# Chat Export — {datetime.now():%Y-%m-%d %H:%M}\n", f"**Model:** {session.model}\n\n"]
    for msg in session.history:
        role = msg["role"].title()
        content = msg.get("content") or ""
        lines.append(f"## {role}\n\n{content}\n\n---\n\n")

    path = Path(filename)
    path.write_text("".join(lines), encoding="utf-8")
    console.print(f"[green]✓[/green] Exported to [bold]{path}[/bold]")


async def _cmd_exit(ctx: CommandContext) -> None:
    console.print("[dim]Goodbye![/dim]")
    raise SystemExit(0)


async def _cmd_multiline(ctx: CommandContext) -> None:
    """Enter multi-line input mode (end with a line containing only '.')."""
    console.print("[dim]Multi-line mode: enter text, end with a single '.' on its own line.[/dim]")
    lines: list[str] = []
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, input, "... ")
        except (EOFError, KeyboardInterrupt):
            break
        if line == ".":
            break
        lines.append(line)

    combined = "\n".join(lines).strip()
    if combined:
        await ctx.session.send_message(combined)
    else:
        console.print("[dim]Empty input cancelled.[/dim]")


# ── Build default registry ────────────────────────────────────────────────────


def build_registry() -> CommandRegistry:
    registry = CommandRegistry()

    commands: list[SlashCommand] = [
        SlashCommand("help",      "Show all available commands",              "/help",                _cmd_help,      aliases=["?"]),
        SlashCommand("model",     "Switch to a different model",              "/model [name]",        _cmd_model),
        SlashCommand("models",    "List all available models",                "/models",              _cmd_models),
        SlashCommand("clear",     "Clear conversation history",               "/clear",               _cmd_clear,     aliases=["reset"]),
        SlashCommand("system",    "View or set the system prompt",            "/system [prompt]",     _cmd_system),
        SlashCommand("save",      "Save conversation to JSON",                "/save [filename]",     _cmd_save),
        SlashCommand("load",      "Load a saved conversation",                "/load [filename]",     _cmd_load),
        SlashCommand("history",   "Show recent conversation history",         "/history [n]",         _cmd_history,   aliases=["h"]),
        SlashCommand("retry",     "Retry the last message",                   "/retry",               _cmd_retry),
        SlashCommand("copy",      "Copy last response to clipboard",          "/copy",                _cmd_copy),
        SlashCommand("tokens",    "Show token usage for this session",        "/tokens",              _cmd_tokens),
        SlashCommand("temp",      "View or set temperature",                  "/temp [value]",        _cmd_temp),
        SlashCommand("mcp",       "Show MCP servers and tools",               "/mcp",                 _cmd_mcp),
        SlashCommand("export",    "Export conversation as Markdown",          "/export [filename]",   _cmd_export),
        SlashCommand("multi",     "Enter multi-line input mode",              "/multi",               _cmd_multiline, aliases=["ml"]),
        SlashCommand("exit",      "Exit the application",                     "/exit",                _cmd_exit,      aliases=["quit", "q"]),
    ]

    for cmd in commands:
        registry.register(cmd)

    return registry