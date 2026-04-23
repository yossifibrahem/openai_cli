"""Chat session: manages history, streaming responses, and the interactive REPL."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import openai
from openai import AsyncOpenAI

from .utils import console
from .mcp_client import MCPManager
from .models import ModelManager
from .renderer import (
    StreamingRenderer,
    render_error,
    render_info,
    render_tool_call,
    render_tool_result,
    render_token_usage,
    render_separator,
)

if TYPE_CHECKING:
    from .commands import CommandRegistry
    from .config import Settings

logger = logging.getLogger(__name__)

# Maximum tool-call rounds per user message (prevents infinite loops)
MAX_TOOL_ROUNDS = 6

# ── Types ─────────────────────────────────────────────────────────────────────

Message = dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────


class ChatSession:
    """Holds conversation state and drives the REPL."""

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings
        self.model: str = settings.model
        self.system_prompt: str = settings.system_prompt
        self.temperature: float = settings.temperature
        self.history: list[Message] = []
        self.total_tokens: int = 0

        self._client: AsyncOpenAI | None = None
        self._registry: "CommandRegistry | None" = None
        self._mcp: MCPManager | None = None
        self._model_manager: ModelManager | None = None
        self._renderer = StreamingRenderer(theme=settings.theme, word_wrap=settings.word_wrap)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def registry(self) -> "CommandRegistry":
        assert self._registry is not None, "Session not initialized"
        return self._registry

    @property
    def mcp_manager(self) -> MCPManager:
        assert self._mcp is not None, "Session not initialized"
        return self._mcp

    @property
    def model_manager(self) -> ModelManager:
        assert self._model_manager is not None, "Session not initialized"
        return self._model_manager

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Set up the OpenAI client, MCP manager, and command registry."""
        from .commands import build_registry

        self._client = AsyncOpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout=self.settings.timeout,
            max_retries=self.settings.max_retries,
        )
        self._model_manager = ModelManager(self.settings, self._client)
        self._mcp = MCPManager(self.settings.mcp_file, disabled=self.settings.no_mcp)
        self._registry = build_registry()

        # Initialize MCP — must happen before first send
        await self._mcp.initialize()

        # Populate model completer choices
        model_cmd = self._registry.get("model")
        if model_cmd:
            model_cmd.completer_choices = await self._model_manager.list_models()

    # ── REPL ──────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the interactive prompt loop."""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings

        from .completer import SlashCommandCompleter

        self.settings.history_dir.mkdir(parents=True, exist_ok=True)
        prompt_history_file = self.settings.history_dir / ".prompt_history"

        bindings = KeyBindings()

        @bindings.add("c-c")
        def _cancel(event: Any) -> None:  # noqa: ANN401
            event.app.exit(result="")

        session: PromptSession[str] = PromptSession(
            history=FileHistory(str(prompt_history_file)),
            completer=SlashCommandCompleter(self._registry),  # type: ignore[arg-type]
            complete_while_typing=True,
            key_bindings=bindings,
        )

        console.print(self._welcome_banner())

        while True:
            try:
                prompt_text = self._build_prompt()
                user_input = await session.prompt_async(prompt_text)
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

    # ── Message sending ───────────────────────────────────────────────────────

    async def send_message(self, content: str, *, from_retry: bool = False) -> str | None:
        """Append user message to history, stream response, handle tool calls."""
        assert self._client is not None

        if not from_retry:
            self.history.append({"role": "user", "content": content})
        self._trim_history()

        messages = self._build_messages()
        tools = self._mcp.tools if self._mcp and self._mcp.tools else None

        console.print()  # blank line before response

        try:
            response_text = await self._run_with_tools(messages, tools)
        except openai.AuthenticationError:
            render_error("Authentication failed. Check your API key.")
            return None
        except openai.RateLimitError:
            render_error("Rate limit exceeded. Wait a moment and try again.")
            return None
        except openai.BadRequestError as exc:
            render_error(f"Bad request: {exc}")
            return None
        except openai.APIConnectionError:
            render_error("Could not connect to the API. Check your network / base_url.")
            return None
        except openai.APIStatusError as exc:
            render_error(f"API error {exc.status_code}: {exc.message}")
            return None

        if response_text:
            self.history.append({"role": "assistant", "content": response_text})

        return response_text

    # ── Streaming + tool loop ─────────────────────────────────────────────────

    async def _run_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
    ) -> str:
        """Stream a response, handling tool calls for up to MAX_TOOL_ROUNDS."""
        assert self._client is not None
        local_messages = list(messages)

        for round_num in range(MAX_TOOL_ROUNDS + 1):
            # Last round: disable tools to force a final text response
            round_tools = tools if (tools and round_num < MAX_TOOL_ROUNDS) else None

            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": local_messages,
                "stream": self.settings.stream,
                "temperature": self.temperature,
            }
            if self.settings.max_tokens:
                kwargs["max_tokens"] = self.settings.max_tokens
            if round_tools:
                kwargs["tools"] = round_tools
                kwargs["tool_choice"] = "auto"

            if self.settings.stream:
                text, tool_calls, usage = await self._stream_response(kwargs)
            else:
                text, tool_calls, usage = await self._blocking_response(kwargs)

            # Track token usage
            if usage:
                self.total_tokens += usage.get("total_tokens", 0)
                if self.settings.show_token_usage and usage.get("total_tokens"):
                    render_token_usage(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        usage.get("total_tokens", 0),
                    )

            # If no tool calls, we have our final answer
            if not tool_calls:
                return text

            # Execute tool calls and add results to local message history
            local_messages.append(
                {"role": "assistant", "content": text or None, "tool_calls": tool_calls}
            )
            for tc in tool_calls:
                fn = tc["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                render_tool_call(name, fn.get("arguments", ""))
                result = await self._mcp.call_tool(name, args)
                render_tool_result(name, result)

                local_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            render_separator()

        return ""  # Should not reach here

    async def _stream_response(
        self, kwargs: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
        """Stream a response; collect text + any tool call chunks."""
        assert self._client is not None

        tool_call_accumulator: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] | None = None

        async with self._renderer.live_display(self.model, show_thinking=True) as renderer:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None

                if choice is None:
                    continue

                delta = choice.delta

                # ── Text token ────────────────────────────────────────────────
                if delta.content:
                    renderer.push(delta.content)

                # ── Tool call chunk ───────────────────────────────────────────
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_accumulator:
                            tool_call_accumulator[idx] = {
                                "id": tc_delta.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        acc = tool_call_accumulator[idx]
                        if tc_delta.id:
                            acc["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                acc["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                acc["function"]["arguments"] += tc_delta.function.arguments

                # ── Usage (last chunk) ────────────────────────────────────────
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

        tool_calls = list(tool_call_accumulator.values()) if tool_call_accumulator else []
        return renderer.text, tool_calls, usage

    async def _blocking_response(
        self, kwargs: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
        """Non-streaming response (stream=False)."""
        assert self._client is not None
        kwargs = {**kwargs, "stream": False}

        with console.status(f"[cyan]{self.model} is thinking…[/cyan]"):
            response = await self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        msg = choice.message
        text = msg.content or ""

        if text:
            from .renderer import render_message
            render_message("assistant", text, self.settings.theme)

        tool_calls: list[dict[str, Any]] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                })

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return text, tool_calls, usage

    # ── History helpers ───────────────────────────────────────────────────────

    def clear_history(self) -> None:
        self.history = []

    def _trim_history(self) -> None:
        """Keep history within context_window message count."""
        limit = self.settings.context_window
        if len(self.history) > limit:
            self.history = self.history[-limit:]

    def _build_messages(self) -> list[Message]:
        """Prepend system prompt to history."""
        system: list[Message] = [{"role": "system", "content": self.system_prompt}]
        return system + self.history

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _build_prompt(self) -> str:
        if self.settings.show_model_in_prompt:
            return f"\n[{self.model}] You: "
        return "\nYou: "

    def _welcome_banner(self) -> str:
        mcp_info = ""
        if self._mcp and self._mcp.tools:
            mcp_info = f" · [magenta]{len(self._mcp.tools)} MCP tool(s)[/magenta]"
        return (
            "\n[bold cyan]OpenAI CLI Chat[/bold cyan]  "
            f"[dim]model:[/dim] [cyan]{self.model}[/cyan]{mcp_info}\n"
            "[dim]Type a message, or [bold]/help[/bold] for commands.[/dim]\n"
        )