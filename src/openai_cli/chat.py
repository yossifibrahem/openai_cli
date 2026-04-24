"""Chat session — history, streaming responses, tool calls, and the REPL."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import openai
from openai import AsyncOpenAI

from .mcp_client import MCPManager
from .models import ModelManager
from .renderer import (
    StreamingRenderer, ToolChoice,
    confirm_tool_call, render_error, render_info,
    render_separator, render_tool_call, render_tool_result,
)
from .utils import console

if TYPE_CHECKING:
    from .commands import CommandRegistry
    from .config import Settings

logger = logging.getLogger(__name__)
MAX_TOOL_ROUNDS = 6

Message = dict[str, Any]


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

    async def initialize(self) -> None:
        from .commands import build_registry
        self._client = AsyncOpenAI(api_key=self.settings.api_key, base_url=self.settings.base_url)
        self._model_manager = ModelManager(self._client)
        self._mcp = MCPManager(self.settings.mcp_file, disabled=self.settings.no_mcp)
        self._registry = build_registry()
        await self._mcp.initialize()

        model_cmd = self._registry.get("model")
        if model_cmd:
            model_cmd.completer_choices = await self._model_manager.list_models()

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
            if self._mcp:
                await self._mcp.aclose()

    async def send_message(self, content: str) -> str | None:
        assert self._client is not None
        self.history.append({"role": "user", "content": content})
        messages = [{"role": "system", "content": self.settings.system_prompt}, *self.history]
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

    async def _run_with_tools(self, messages: list[Message], tools: list | None) -> str:
        assert self._client is not None
        local_messages = list(messages)
        text = ""

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

        return text

    async def _stream_response(self, kwargs: dict[str, Any]) -> tuple[str, list]:
        assert self._client is not None
        tool_call_accumulator: dict[int, dict[str, Any]] = {}
        interrupted = False

        async with self._renderer.live_display(self.model) as renderer:
            stream = await self._client.chat.completions.create(**kwargs, stream=True)
            try:
                async for chunk in stream:
                    choice = chunk.choices[0] if chunk.choices else None
                    if not choice:
                        continue
                    if choice.delta.content:
                        renderer.push(choice.delta.content)
                    if choice.delta.tool_calls:
                        for tc in choice.delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_call_accumulator:
                                tool_call_accumulator[idx] = {
                                    "id": tc.id or "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            acc = tool_call_accumulator[idx]
                            if tc.id:
                                acc["id"] = tc.id
                            if tc.function:
                                acc["function"]["name"] += tc.function.name or ""
                                acc["function"]["arguments"] += tc.function.arguments or ""
            except (asyncio.CancelledError, KeyboardInterrupt):
                interrupted = True
                renderer.mark_interrupted()

        if interrupted:
            render_info("⚠  Response interrupted.")

        tool_calls = [] if interrupted else list(tool_call_accumulator.values())
        return renderer.text, tool_calls

    async def _execute_tool_calls(
        self, messages: list[Message], assistant_text: str, tool_calls: list
    ) -> tuple[list[Message], bool]:
        updated = list(messages)
        updated.append({"role": "assistant", "content": assistant_text or None, "tool_calls": tool_calls})

        for tc in tool_calls:
            fn = tc["function"]
            name: str = fn["name"]
            raw_args: str = fn.get("arguments", "")

            choice = await confirm_tool_call(name, raw_args)

            if choice is ToolChoice.CANCEL:
                render_info(f"⊘  Cancelled — skipping {name} and all remaining tools.")
                for remaining_tc in tool_calls[tool_calls.index(tc):]:
                    updated.append({"role": "tool", "tool_call_id": remaining_tc["id"],
                                    "content": "Tool execution cancelled by user."})
                return updated, True

            if choice is ToolChoice.DENY:
                render_info(f"⊘  Denied — skipping {name}.")
                updated.append({"role": "tool", "tool_call_id": tc["id"],
                                "content": "Tool call denied by user."})
                continue

            try:
                args = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                args = {}

            render_tool_call(name, raw_args)
            result = await self._mcp.call_tool(name, args)  # type: ignore[union-attr]
            render_tool_result(name, result)
            updated.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

        return updated, False

    def _welcome_banner(self) -> str:
        mcp_info = ""
        if self._mcp and self._mcp.tools:
            mcp_info = f" · [magenta]{len(self._mcp.tools)} MCP tool(s)[/magenta]"
        return (
            f"\n[bold cyan]OpenAI CLI Chat[/bold cyan]  "
            f"[dim]model:[/dim] [cyan]{self.model}[/cyan]{mcp_info}\n"
            "[dim]Type a message, or [bold]/help[/bold] for commands.[/dim]\n"
        )