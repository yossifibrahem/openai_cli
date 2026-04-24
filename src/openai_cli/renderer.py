"""Streaming markdown renderer using Rich."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from enum import Enum
from typing import AsyncIterator

from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from .utils import console


class ToolChoice(str, Enum):
    ALLOW  = "allow"
    DENY   = "deny"
    CANCEL = "cancel"


class StreamingRenderer:
    def __init__(self, theme: str = "monokai") -> None:
        self._theme = theme
        self._buffer = ""
        self._live: Live | None = None

    @property
    def text(self) -> str:
        return self._buffer

    def push(self, chunk: str) -> None:
        self._buffer += chunk
        if self._live and self._buffer.strip():
            self._live.update(Markdown(self._buffer, code_theme=self._theme))

    def mark_interrupted(self) -> None:
        if self._live:
            self._live.update(Text(""))

    @asynccontextmanager
    async def live_display(
        self, model: str, *, show_thinking: bool = True
    ) -> AsyncIterator["StreamingRenderer"]:
        self._buffer = ""
        initial = (
            Spinner("dots", text=Text(f" {model} is thinking…", style="dim italic"))
            if show_thinking else Text("")
        )
        with Live(initial, console=console, refresh_per_second=15,
                  vertical_overflow="visible", transient=True) as live:
            self._live = live
            try:
                yield self
            finally:
                self._live = None

        if self._buffer.strip():
            console.print(Markdown(self._buffer, code_theme=self._theme))


async def confirm_tool_call(name: str, args: str) -> ToolChoice:
    """Render a tool-call panel and wait for a keypress: A / D / C."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings

    preview = args[:400] + ("..." if len(args) > 400 else "")
    console.print(Panel(
        f"[bold]{name}[/bold]\n[dim]{preview}[/dim]",
        title="[yellow]Tool Call - Confirmation Required[/yellow]",
        border_style="yellow", expand=False,
    ))
    console.print(
        "  [bold green][A]llow[/bold green]"
        "  [bold red][D]eny[/bold red]"
        "  [bold dim][C]ancel all[/bold dim]"
    )

    kb = KeyBindings()

    @kb.add("a")
    @kb.add("A")
    @kb.add("enter")
    def _allow(event): event.app.exit(result=ToolChoice.ALLOW)

    @kb.add("d")
    @kb.add("D")
    def _deny(event): event.app.exit(result=ToolChoice.DENY)

    @kb.add("c")
    @kb.add("C")
    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event): event.app.exit(result=ToolChoice.CANCEL)

    session: PromptSession = PromptSession(key_bindings=kb)
    try:
        choice = await session.prompt_async("  > ")
    except (EOFError, KeyboardInterrupt):
        choice = ToolChoice.CANCEL

    if not isinstance(choice, ToolChoice):
        choice = ToolChoice.ALLOW

    labels = {
        ToolChoice.ALLOW: "[green]Allowed[/green]",
        ToolChoice.DENY: "[red]Denied[/red]",
        ToolChoice.CANCEL: "[dim]Cancelled[/dim]",
    }
    console.print(f"  {labels[choice]}\n")
    return choice


def render_tool_call(name: str, args: str) -> None:
    console.print(Panel(
        f"[bold]{name}[/bold]\n[dim]{args}[/dim]",
        title="[yellow]⚙ Tool Call[/yellow]", border_style="yellow", expand=False,
    ))

def render_tool_result(name: str, result: str) -> None:
    content = result[:800] + ("…" if len(result) > 800 else "")
    console.print(Panel(
        content, title=f"[green]✓ {name}[/green]",
        border_style="green dim", expand=False,
    ))

def render_error(message: str) -> None:
    console.print(f"\n[bold red]Error:[/bold red] {message}")

def render_info(message: str) -> None:
    console.print(f"[dim]{message}[/dim]")

def render_separator() -> None:
    console.print("[dim]" + "─" * 60 + "[/dim]")