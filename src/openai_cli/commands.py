"""Slash command registry — /clear, /exit, /help, /mcp, /model, /models, /multi."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from rich.table import Table

from .utils import console

if TYPE_CHECKING:
    from .chat import ChatSession

AsyncHandler = Callable[["CommandContext"], Coroutine[Any, Any, None]]


@dataclass
class SlashCommand:
    name: str
    description: str
    usage: str
    handler: AsyncHandler
    aliases: list[str] = field(default_factory=list)
    completer_choices: list[str] = field(default_factory=list)


@dataclass
class CommandContext:
    session: "ChatSession"
    args: str


# ── Registry ──────────────────────────────────────────────────────────────────


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(self, cmd: SlashCommand) -> None:
        self._commands[cmd.name] = cmd
        for alias in cmd.aliases:
            self._commands[alias] = cmd

    def get(self, name: str) -> SlashCommand | None:
        return self._commands.get(name.lstrip("/").split()[0].lower())

    def all_names(self) -> list[str]:
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

    async def execute(self, raw: str, session: "ChatSession") -> None:
        stripped = raw.lstrip("/").strip()
        parts = stripped.split(None, 1)
        name = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        cmd = self.get(name)
        if cmd is None:
            console.print(f"[red]Unknown command:[/red] /{name}  (try [bold]/help[/bold])")
            return

        ctx = CommandContext(session=session, args=args)
        try:
            await cmd.handler(ctx)
        except SystemExit:
            raise
        except Exception as exc:
            console.print(f"[red]Command error:[/red] {exc}")


# ── Handlers ──────────────────────────────────────────────────────────────────


async def _cmd_help(ctx: CommandContext) -> None:
    table = Table(title="Slash Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="cyan", width=20)
    table.add_column("Description", style="white")
    for cmd in ctx.session.registry.all_commands():
        aliases = f"  [dim]({', '.join('/' + a for a in cmd.aliases)})[/dim]" if cmd.aliases else ""
        table.add_row(cmd.usage + aliases, cmd.description)
    console.print(table)


async def _cmd_clear(ctx: CommandContext) -> None:
    ctx.session.history.clear()
    console.clear()
    console.print(ctx.session._welcome_banner())


async def _cmd_exit(ctx: CommandContext) -> None:
    console.print("[dim]Goodbye![/dim]")
    raise SystemExit(0)


async def _cmd_mcp(ctx: CommandContext) -> None:
    ctx.session.mcp_manager.display_servers()


async def _cmd_model(ctx: CommandContext) -> None:
    session = ctx.session
    if not ctx.args:
        console.print(f"Current model: [bold cyan]{session.model}[/bold cyan]")
        console.print("Usage: [bold]/model <name>[/bold] or [bold]/models[/bold] to list all")
        return
    new_model = ctx.args.strip()
    session.model = new_model
    console.print(f"[green]✓[/green] Switched to [bold cyan]{new_model}[/bold cyan]")


async def _cmd_models(ctx: CommandContext) -> None:
    await ctx.session.model_manager.list_and_display()


async def _cmd_multi(ctx: CommandContext) -> None:
    console.print("[dim]Multi-line mode — end with a single '.' on its own line.[/dim]")
    lines: list[str] = []
    loop = asyncio.get_running_loop()
    while True:
        try:
            line = await loop.run_in_executor(None, input, "... ")
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


# ── Build registry ────────────────────────────────────────────────────────────


def build_registry() -> CommandRegistry:
    registry = CommandRegistry()
    for cmd in [
        SlashCommand("clear",  "Clear conversation history",      "/clear",        _cmd_clear,  aliases=["reset"]),
        SlashCommand("exit",   "Exit the application",            "/exit",         _cmd_exit,   aliases=["quit", "q"]),
        SlashCommand("help",   "Show all available commands",     "/help",         _cmd_help,   aliases=["?"]),
        SlashCommand("mcp",    "Show MCP servers and tools",      "/mcp",          _cmd_mcp),
        SlashCommand("model",  "View or switch model",            "/model [name]", _cmd_model),
        SlashCommand("models", "List all available models",       "/models",       _cmd_models),
        SlashCommand("multi",  "Enter multi-line input mode",     "/multi",        _cmd_multi,  aliases=["ml"]),
    ]:
        registry.register(cmd)
    return registry
