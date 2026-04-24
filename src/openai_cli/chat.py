"""Chat session — history, streaming responses, tool calls, and the REPL."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, TypedDict

import openai
from openai import AsyncOpenAI

from .mcp_client import MCPManager
from .models import ModelManager
from .renderer import (
    StreamingRenderer,
    ToolChoice,
    confirm_tool_call,
    render_error,
    render_info,
    render_separator,
    render_tool_call,
    render_tool_result,
)
from .utils import console

if TYPE_CHECKING:
    from .commands import CommandRegistry
    from .config import Settings

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6


class _MessageRequired(TypedDict):
    role: str

class Message(_MessageRequired, total=False):
    content: str | None
    tool_calls: list[dict[str, Any]]
    tool_call_id: str


class ChatSession:
    def __init__(self, settings: "Settings") -> None:
        self.settings = settings
        self.model: str = settings.model
        self.history: list[Message] = []

        self._client: AsyncOpenAI | None = None
        self._registry: "CommandRegistry | None" = None
        self._mcp: MCPManager | None = None
        self._model_manager: ModelManager | None = None
        self._renderer = StreamingRenderer()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def registry(self) -> "CommandRegistry":
        assert self._registry is not None
        return self._registry

    @property
    def mcp_manager(self) -> MCPManager:
        assert self._mcp is not None
        return self._mcp

    @property
    def model_manager(self) -> ModelManager:
        assert self._model_manager is not None
        return self._model_manager

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        from .commands import build_registry

        self._client = AsyncOpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
        )
        self._model_manager = ModelManager(self._client)
        self._mcp = MCPManager(self.settings.mcp_file, disabled=self.settings.no_mcp)
        self._registry = build_registry()

        await self._mcp.initialize()

        # Populate model tab-completion
        model_cmd = self._registry.get("model")
        if model_cmd:
            model_cmd.completer_choices = await self._model_manager.list_models()

    # ── REPL ──────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import InMemoryHistory
        from .completer import SlashCommandCompleter

        session: PromptSession[str] = PromptSession(
            history=InMemoryHistory(),
            completer=SlashCommandCompleter(self._registry),  # type: ignore[arg-type]
            complete_while_typing=True,
        )

        console.print(self._welcome_banner())

        try:
            while True:
                try:
                    user_input = await session.prompt_async(f"\n[{self.model}] You: ")
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[dim]Goodbye![/dim]")
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    try:
                        await self._registry.execute(user_input, self)
                    except SystemExit:
                        break
                    continue

                await self.send_message(user_input)
        finally:
            # Always shut down MCP server processes/connections, even on
            # Ctrl-C or an unhandled exception, so no subprocesses are orphaned.
            if self._mcp:
                await self._mcp.aclose()

    # ── Message sending ───────────────────────────────────────────────────────

    async def send_message(self, content: str) -> str | None:
        assert self._client is not None
        self.history.append({"role": "user", "content": content})
        messages = self._build_messages()
        tools = self._mcp.tools if self._mcp and self._mcp.tools else None

        console.print()
        try:
            text = await self._run_with_tools(messages, tools)
        except openai.AuthenticationError:
            render_error("Authentication failed. Check your API key.")
            return None
        except openai.RateLimitError:
            render_error("Rate limit exceeded. Try again shortly.")
            return None
        except openai.BadRequestError as exc:
            render_error(f"Bad request: {exc}")
            return None
        except openai.APIConnectionError:
            render_error("Could not connect. Check your network and base_url.")
            return None
        except openai.APIStatusError as exc:
            render_error(f"API error {exc.status_code}: {exc.message}")
            return None

        if text:
            self.history.append({"role": "assistant", "content": text})
        return text

    # ── Streaming + tool loop ─────────────────────────────────────────────────

    async def _run_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
    ) -> str:
        assert self._client is not None
        local_messages: list[Message] = list(messages)

        for round_num in range(MAX_TOOL_ROUNDS + 1):
            round_tools = tools if (tools and round_num < MAX_TOOL_ROUNDS) else None
            kwargs: dict[str, Any] = {"model": self.model, "messages": local_messages}
            if round_tools:
                kwargs["tools"] = round_tools
                kwargs["tool_choice"] = "auto"

            text, tool_calls = await self._stream_response(kwargs)

            if not tool_calls:
                return text

            local_messages, cancelled = await self._execute_tool_calls(local_messages, text, tool_calls)
            if cancelled:
                return text
            render_separator()

        # MAX_TOOL_ROUNDS exhausted — return whatever the model last said.
        return text

    async def _stream_response(
        self, kwargs: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]]]:
        assert self._client is not None
        tool_call_accumulator: dict[int, dict[str, Any]] = {}
        interrupted = False

        async with self._renderer.live_display(self.model) as renderer:
            stream = await self._client.chat.completions.create(**kwargs, stream=True)
            try:
                async for chunk in stream:
                    choice = chunk.choices[0] if chunk.choices else None
                    if choice is None:
                        continue
                    delta = choice.delta
                    if delta.content:
                        renderer.push(delta.content)
                    if delta.tool_calls:
                        _accumulate_tool_calls(tool_call_accumulator, delta.tool_calls)
            except asyncio.CancelledError:
                # asyncio delivers Ctrl-C as CancelledError inside coroutines.
                # Treat it the same as KeyboardInterrupt here: flush partial
                # output and stop streaming, but keep the app alive.
                interrupted = True
                renderer.mark_interrupted()
            except KeyboardInterrupt:
                interrupted = True
                renderer.mark_interrupted()

        if interrupted:
            render_info("⚠  Response interrupted.")

        # Skip tool calls when interrupted — executing them against a
        # partial assistant message would produce incoherent results.
        tool_calls = (
            [] if interrupted
            else (list(tool_call_accumulator.values()) if tool_call_accumulator else [])
        )
        return renderer.text, tool_calls

    async def _execute_tool_calls(
        self,
        messages: list[Message],
        assistant_text: str,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[list[Message], bool]:
        """Execute each tool call after user confirmation.

        Returns ``(updated_messages, cancelled)`` where *cancelled* is ``True``
        when the user chose **Cancel**, signalling ``_run_with_tools`` to stop
        the tool loop immediately and return the current assistant text.
        """
        updated: list[Message] = list(messages)
        updated.append({
            "role": "assistant",
            "content": assistant_text or None,
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            fn = tc["function"]
            name: str = fn["name"]
            raw_args: str = fn.get("arguments", "")

            # ── Confirmation dialog ───────────────────────────────────────────
            choice = await confirm_tool_call(name, raw_args)

            if choice is ToolChoice.CANCEL:
                # Append a synthetic tool result so the message list stays
                # valid (the model expects a result for every tool_call_id),
                # then signal the caller to abort the loop.
                render_info(f"⊘  Cancelled — skipping {name} and all remaining tools.")
                updated.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": "Tool execution cancelled by user.",
                })
                # Fill refusals for any tools we haven't reached yet.
                remaining = tool_calls[tool_calls.index(tc) + 1:]
                for remaining_tc in remaining:
                    updated.append({
                        "role": "tool",
                        "tool_call_id": remaining_tc["id"],
                        "content": "Tool execution cancelled by user.",
                    })
                return updated, True

            if choice is ToolChoice.DENY:
                render_info(f"⊘  Denied — skipping {name}.")
                updated.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": "Tool call denied by user.",
                })
                continue

            # ── Allow — execute the tool ──────────────────────────────────────
            try:
                args: dict[str, Any] = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                args = {}

            render_tool_call(name, raw_args)
            result = await self._mcp.call_tool(name, args)  # type: ignore[union-attr]
            render_tool_result(name, result)
            updated.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

        return updated, False

    # ── History helpers ───────────────────────────────────────────────────────

    def _build_messages(self) -> list[Message]:
        return [{"role": "system", "content": self.settings.system_prompt}, *self.history]

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _welcome_banner(self) -> str:
        mcp_info = ""
        if self._mcp and self._mcp.tools:
            mcp_info = f" · [magenta]{len(self._mcp.tools)} MCP tool(s)[/magenta]"
        return (
            f"\n[bold cyan]OpenAI CLI Chat[/bold cyan]  "
            f"[dim]model:[/dim] [cyan]{self.model}[/cyan]{mcp_info}\n"
            "[dim]Type a message, or [bold]/help[/bold] for commands.[/dim]\n"
        )


# ── Helper ────────────────────────────────────────────────────────────────────


def _accumulate_tool_calls(
    accumulator: dict[int, dict[str, Any]], tc_deltas: Any
) -> None:
    for tc_delta in tc_deltas:
        idx: int = tc_delta.index
        if idx not in accumulator:
            accumulator[idx] = {
                "id": tc_delta.id or "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        acc = accumulator[idx]
        if tc_delta.id:
            acc["id"] = tc_delta.id
        if tc_delta.function:
            if tc_delta.function.name:
                acc["function"]["name"] += tc_delta.function.name
            if tc_delta.function.arguments:
                acc["function"]["arguments"] += tc_delta.function.arguments