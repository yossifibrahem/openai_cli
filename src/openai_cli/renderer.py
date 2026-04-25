"""Streaming markdown renderer using Rich.

Four rules prevent text duplication:

1. One shared ``Console`` (from ``.utils``) — multiple instances bypass
   Rich's cursor management and cause double-printing.

2. The thinking spinner is the *initial renderable* of the ``Live`` session,
   never a separate ``console.status`` — two sequential Live sessions leave
   the cursor in an unpredictable position.

3. Incremental block-commit rendering — complete Markdown blocks are printed
   permanently above the Live area via ``console.print()``; the Live area
   shows only the current in-progress block.  Rich can always compute its
   exact line-count so cursor-up always lands correctly.  ``transient=True``
   clears the live area on exit; the remainder is printed once as Markdown.

4. Echo suppression — between ``prompt_async()`` calls the terminal is in
   cooked mode (ECHO on), so keystrokes are echoed directly to stdout by the
   OS, outside Rich's awareness.  The stray byte shifts the cursor, making
   cursor-up land a row too low and leaving the previous line intact.
   ``_no_echo()`` clears the ECHO flag for the Live session's lifetime.
   Buffered keystrokes are preserved and handled correctly by prompt_toolkit
   on the next prompt — no stdin drain needed or wanted.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager, contextmanager
from enum import Enum
from typing import AsyncIterator, Generator

from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from .utils import console   # ← shared singleton; never create Console() here


# ── Tool confirmation ─────────────────────────────────────────────────────────


class ToolChoice(str, Enum):
    """Result of the per-tool confirmation prompt."""
    ALLOW  = "allow"   # execute this tool
    DENY   = "deny"    # skip this tool, return a refusal result
    CANCEL = "cancel"  # skip this tool *and* all remaining tools


# ── Terminal helpers ──────────────────────────────────────────────────────────


@contextmanager
def _no_echo() -> Generator[None, None, None]:
    """Suppress terminal echo for the lifetime of the Rich Live session.

    In cooked mode (between prompt_toolkit prompts) the OS echoes keystrokes
    directly to stdout, outside Rich's Console.  The stray byte shifts the
    cursor so Rich's cursor-up lands a row too low, leaving the previous
    live-area line intact — the duplication bug.

    Buffered keystrokes are *not* drained on exit: the OS never retroactively
    echoes already-buffered input, and prompt_toolkit reads them correctly
    through its own raw-mode pipeline on the next prompt.

    ``TCSADRAIN`` is used for both set and restore so Rich's in-flight escape
    sequences land before the attribute change takes effect.

    No-op on Windows (no ``termios``) and non-tty stdin.
    """
    try:
        import termios
    except ImportError:
        yield
        return

    if not sys.stdin.isatty():
        yield
        return

    fd = sys.stdin.fileno()
    old: list[int] = termios.tcgetattr(fd)
    try:
        new = termios.tcgetattr(fd)
        new[3] &= ~termios.ECHO  # lflags — clear the ECHO bit only
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── Streaming renderer ────────────────────────────────────────────────────────


class StreamingRenderer:
    """Collects streamed chunks and renders markdown with incremental commits.

    Usage::

        renderer = StreamingRenderer(theme="monokai")
        async with renderer.live_display(model) as renderer:
            async for chunk in openai_stream:
                renderer.push(chunk)
        full_text = renderer.text

    The Live area is kept small (one in-progress block at a time) so Rich
    can always track cursor position accurately, regardless of response length.
    Complete blocks are committed permanently above the Live area via
    ``console.print()``, which Rich handles without cursor gymnastics.
    """

    def __init__(self, theme: str = "monokai") -> None:
        self._theme = theme
        self._buffer = ""       # full accumulated response text
        self._commit_pos = 0    # bytes of _buffer already printed permanently
        self._live: Live | None = None
        self._interrupted: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def text(self) -> str:
        return self._buffer

    def push(self, chunk: str) -> None:
        """Append a text chunk, commit any newly completed blocks, refresh live."""
        self._buffer += chunk
        if self._live is None:
            return

        # Commit every complete block that has arrived since last push.
        self._flush_complete_blocks()

        # Update the live area with only the current in-progress block.
        pending = self._buffer[self._commit_pos:]
        if pending.strip():
            self._live.update(Markdown(pending, code_theme=self._theme))

    def mark_interrupted(self) -> None:
        """Flush all pending content and flag the session as interrupted.

        The caller should print a visual notice after the Live context exits.
        """
        self._interrupted = True
        if self._live is not None:
            # Treat a partial last line as complete before flushing.
            if self._buffer and not self._buffer.endswith("\n"):
                self._buffer += "\n"
            self._flush_complete_blocks()
            self._live.update(Text(""))  # clear the in-progress live area

    @asynccontextmanager
    async def live_display(
        self,
        model: str,
        *,
        show_thinking: bool = True,
    ) -> AsyncIterator["StreamingRenderer"]:
        """Async context manager owning the Rich Live session.

        Shows a thinking spinner until the first chunk arrives, then streams
        content with incremental block commits.  On exit the live area is
        erased (``transient=True``) and the uncommitted remainder is printed
        once as Markdown.
        """
        self._buffer = ""
        self._commit_pos = 0
        self._interrupted = False

        initial_renderable = (
            Spinner("dots", text=Text(f" {model} is thinking…", style="dim italic"))
            if show_thinking
            else Text("")
        )

        # _no_echo() suppresses OS keystroke echoes that would corrupt Rich's
        # cursor tracking; see _no_echo() for the full rationale.
        with _no_echo(), Live(
            initial_renderable,
            console=console,
            refresh_per_second=15,
            vertical_overflow="visible",
            transient=True,
        ) as live:
            self._live = live
            try:
                yield self
            finally:
                self._live = None

        # Print the last in-progress block now that it's complete.
        remainder = self._buffer[self._commit_pos:]
        if remainder.strip():
            console.print(Markdown(remainder, code_theme=self._theme))

    # ── Private ───────────────────────────────────────────────────────────────

    def _flush_complete_blocks(self) -> None:
        """Commit every newly completed block to the console.

        Loops until no further commit boundary can be found in the pending
        portion of the buffer.  Each committed block is printed permanently
        above the Rich Live area via ``console.print()``.
        """
        while True:
            pending = self._buffer[self._commit_pos:]
            boundary = self._find_commit_boundary(pending)
            if boundary <= 0:
                break

            block = pending[:boundary]
            if block.strip():
                console.print(Markdown(block, code_theme=self._theme))
            self._commit_pos += boundary

    @staticmethod
    def _find_commit_boundary(text: str) -> int:
        """Index up to which *text* can be safely committed.

        Safe positions: a blank line outside any open code fence, or the line
        that closes a code fence.  The last (possibly incomplete) line is
        never included.  Returns 0 if no boundary exists yet.
        """
        if not text or "\n" not in text:
            return 0

        lines = text.split("\n")
        in_fence = False
        pos = 0
        safe_boundary = 0

        for i, line in enumerate(lines):
            is_last_line = i == len(lines) - 1
            line_end = pos + len(line) + (0 if is_last_line else 1)  # +1 for the \n

            if is_last_line:
                # Never commit a partial (possibly incomplete) line.
                break

            stripped = line.strip()
            is_fence_marker = stripped.startswith("```") or stripped.startswith("~~~")

            if is_fence_marker:
                in_fence = not in_fence
                if not in_fence:
                    # Closing fence — safe to commit up to end of this line.
                    safe_boundary = line_end
            elif not in_fence and stripped == "":
                # Blank line outside a code fence — paragraph boundary.
                safe_boundary = line_end

            pos = line_end

        return safe_boundary


# ── Non-streaming helpers — all use the shared console ───────────────────────


async def confirm_tool_call(name: str, args: str) -> ToolChoice:
    """Render a tool-call panel and present an arrow-key menu for confirmation.

    Navigate with ↑ / ↓ (or k / j), confirm with Enter, or press Esc / Ctrl-C
    to cancel immediately.

    Options
    -------
    Allow      — execute this tool normally.
    Deny       — skip this tool; the model receives a refusal result.
    Cancel all — skip this tool *and* every remaining tool in the round.
    """
    from prompt_toolkit import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    # ── Tool preview panel (via Rich) ─────────────────────────────────────────
    preview = args[:400] + ("..." if len(args) > 400 else "")
    console.print(
        Panel(
            f"[bold]{name}[/bold]\n[dim]{preview}[/dim]",
            title="[yellow]⚙ Tool Call — Confirmation Required[/yellow]",
            border_style="yellow",
            expand=False,
        )
    )

    # ── Menu definition ───────────────────────────────────────────────────────
    _OPTIONS: list[tuple[ToolChoice, str, str, str]] = [
        # (value,              label,        style-selected,   style-normal)
        (ToolChoice.ALLOW,  "  Allow",       "fg:ansigreen bold",  "fg:ansigreen"),
        (ToolChoice.DENY,   "  Deny",        "fg:ansired bold",    "fg:ansired"),
        (ToolChoice.CANCEL, "  Cancel all",  "fg:ansiwhite bold",  "fg:ansibrightblack"),
    ]
    _CURSOR = "❯"
    _SPACER = " "

    state: dict[str, int] = {"idx": 0}
    result: list[ToolChoice] = [ToolChoice.CANCEL]

    # ── Live renderer (called on every keypress redraw) ───────────────────────
    def _render() -> FormattedText:
        fragments: list[tuple[str, str]] = []
        for i, (_, label, style_sel, style_norm) in enumerate(_OPTIONS):
            active = i == state["idx"]
            cursor = _CURSOR if active else _SPACER
            style  = style_sel if active else style_norm
            fragments.append((style, f" {cursor}{label}\n"))
        fragments.append(("italic ansibrightblack", "\n  ↑↓ to move  ·  Enter to select  ·  Esc to cancel\n"))
        return FormattedText(fragments)

    # ── Key bindings ──────────────────────────────────────────────────────────
    kb: KeyBindings = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(event) -> None:
        state["idx"] = (state["idx"] - 1) % len(_OPTIONS)

    @kb.add("down")
    @kb.add("j")
    def _down(event) -> None:
        state["idx"] = (state["idx"] + 1) % len(_OPTIONS)

    @kb.add("enter")
    def _select(event) -> None:
        result[0] = _OPTIONS[state["idx"]][0]
        event.app.exit()

    @kb.add("escape")
    @kb.add("c-c")
    def _abort(event) -> None:
        result[0] = ToolChoice.CANCEL
        event.app.exit()

    # ── Run the interactive menu ──────────────────────────────────────────────
    app: Application[None] = Application(
        layout=Layout(
            Window(
                content=FormattedTextControl(_render, focusable=True),
                dont_extend_height=True,
            )
        ),
        key_bindings=kb,
        full_screen=False,
        mouse_support=False,
        erase_when_done=True,   # wipe the menu lines after selection
    )
    try:
        await app.run_async()
    except (EOFError, KeyboardInterrupt):
        result[0] = ToolChoice.CANCEL

    # ── Print the resolved choice for the scrollback record ──────────────────
    _LABELS = {
        ToolChoice.ALLOW:  "[green]✔ Allowed[/green]",
        ToolChoice.DENY:   "[red]✖ Denied[/red]",
        ToolChoice.CANCEL: "[dim]⊘ Cancelled[/dim]",
    }
    console.print(f"  {_LABELS[result[0]]}\n")
    return result[0]

def render_tool_call(name: str, args: str) -> None:
    console.print(
        Panel(
            f"[bold]{name}[/bold]\n[dim]{args}[/dim]",
            title="[yellow]⚙ Tool Call[/yellow]",
            border_style="yellow",
            expand=False,
        )
    )


def render_tool_result(name: str, result: str) -> None:
    content = result[:800] + ("…" if len(result) > 800 else "")
    console.print(
        Panel(
            content,
            title=f"[green]✓ {name}[/green]",
            border_style="green dim",
            expand=False,
        )
    )


def render_error(message: str) -> None:
    console.print(f"\n[bold red]Error:[/bold red] {message}")


def render_info(message: str) -> None:
    console.print(f"[dim]{message}[/dim]")


def render_separator() -> None:
    console.print("[dim]" + "─" * 60 + "[/dim]")