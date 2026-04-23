"""Streaming markdown renderer using Rich.

Design rules that prevent text duplication:

1. A single shared ``Console`` instance (imported from ``.utils``) is used
   everywhere.  Multiple ``Console()`` instances writing to the same stdout
   bypass Rich's Live cursor-management and cause double-printing.

2. The "thinking" spinner is the *initial renderable* of the ``Live`` session
   itself — not a separate ``console.status`` call.  Starting two sequential
   ``Live`` sessions (status uses Live internally) on the same console leaves
   the cursor in an unpredictable position.

3. **Incremental block-commit rendering** eliminates the duplication bug:

   The root cause of the duplication bug is that re-rendering the entire
   response buffer as a single ``Markdown`` object inside a Rich ``Live``
   session causes the live area to grow unboundedly.  Rich tracks cursor
   position by computing the rendered line-count of the previous update and
   issuing cursor-up escape codes to overwrite it.  When the rendered
   content exceeds the terminal height, cursor-up can only move to the top
   of the visible screen — leaving old content partially intact — and the
   new render is written below it, producing apparent duplication.

   The fix:

   a. As chunks arrive, scan the uncommitted portion of the buffer for
      *complete* Markdown blocks: paragraphs ended by a blank line, or
      code fences whose closing ``` has arrived.
   b. When a complete block is found, commit it permanently via a normal
      ``console.print()`` call (Rich renders this *above* the live area
      without cursor gymnastics).  The committed position advances.
   c. The Live area shows only the current *in-progress* block — always a
      small, bounded slice of the response.  Rich can always compute its
      line-count exactly, so cursor-up always lands correctly.
   d. ``transient=True`` on ``Live`` ensures the in-progress area is fully
      cleared when the context exits.  The uncommitted remainder is then
      printed once as ``Markdown`` — a normal print, no cursor movement.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from .utils import console   # ← shared singleton; never create Console() here


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

    def __init__(self, theme: str = "monokai", word_wrap: bool = True) -> None:
        self._theme = theme
        self._word_wrap = word_wrap
        self._buffer = ""       # full accumulated response text
        self._commit_pos = 0    # bytes of _buffer already printed permanently
        self._live: Live | None = None

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

    @asynccontextmanager
    async def live_display(
        self,
        model: str,
        *,
        show_thinking: bool = True,
    ) -> AsyncIterator["StreamingRenderer"]:
        """Async context manager owning the Rich Live session.

        On entry:  shows a "thinking" spinner as the initial renderable so
                   there is only ever one Live session active (no status+Live
                   overlap).
        On exit:   the live area (in-progress block) is cleared via
                   ``transient=True``; the uncommitted remainder is then
                   printed once as ``Markdown`` via a plain ``console.print``
                   — no cursor gymnastics possible.
        """
        self._buffer = ""
        self._commit_pos = 0

        initial_renderable = (
            Spinner("dots", text=Text(f" {model} is thinking…", style="dim italic"))
            if show_thinking
            else Text("")
        )

        # transient=True: Rich fully erases the live area when __exit__ is
        # called.  Combined with incremental block commits above the live
        # area, this guarantees that content is never double-printed.
        with Live(
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

        # Live has cleared its area.  Print the uncommitted remainder (the
        # last in-progress block, now complete) as Markdown exactly once.
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
        """Return the index up to which *text* can be safely committed.

        A position is safe to commit when:
        - It falls on a blank line *outside* any open code fence
          (paragraph boundary), or
        - It falls immediately after a line that closes an open code fence.

        The last line of *text* is never included (it may be incomplete —
        the stream may deliver the rest of it in the next chunk).

        Returns 0 if no safe commit boundary exists yet.
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


def render_message(role: str, content: str, theme: str = "monokai") -> None:
    """Render a single message with role label."""
    if role == "assistant":
        label = "[bold cyan]Assistant[/bold cyan]"
    elif role == "user":
        label = "[bold green]You[/bold green]"
    else:
        label = f"[bold yellow]{role.title()}[/bold yellow]"

    console.print(f"\n{label}")
    console.print(Markdown(content, code_theme=theme))


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


def render_token_usage(prompt: int, completion: int, total: int) -> None:
    console.print(
        f"\n[dim]Tokens — prompt: {prompt:,} · completion: {completion:,} · total: {total:,}[/dim]"
    )


def render_separator() -> None:
    console.print("[dim]" + "─" * 60 + "[/dim]")