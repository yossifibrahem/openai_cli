"""Chat session: manages history, streaming responses, and the interactive REPL."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine, TypedDict

import openai
from openai import AsyncOpenAI

from .mcp_client import MCPManager
from .models import ModelManager
from .renderer import (
    StreamingRenderer,
    render_error,
    render_message,
    render_separator,
    render_token_usage,
    render_tool_call,
    render_tool_result,
)
from .utils import console

if TYPE_CHECKING:
    from .commands import CommandRegistry
    from .config import Settings

logger = logging.getLogger(__name__)

# Maximum tool-call rounds per user message (prevents infinite loops)
MAX_TOOL_ROUNDS = 6


# ── Types ─────────────────────────────────────────────────────────────────────


class Message(TypedDict, total=False):
    """A single chat message exchanged with the API."""

    role: str
    content: str | None
    tool_calls: list[dict[str, Any]]
    tool_call_id: str


class TokenUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# Error messages mapped by exception type
_API_ERROR_MESSAGES: dict[type[openai.APIError], str] = {
    openai.AuthenticationError: "Authentication failed. Check your API key.",
    openai.RateLimitError: "Rate limit exceeded. Wait a moment and try again.",
    openai.BadRequestError: "Bad request: {exc}",
    openai.APIConnectionError: "Could not connect to the API. Check your network / base_url.",
}


# ─────────────────────────────────────────────────────────────────────────────


class ChatSession:
    """Holds conversation state and drives the REPL."""

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings
        self.model: str = settings.model
        self.system_prompt: str = settings.system_prompt
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
        if self._registry is None:
            raise RuntimeError("ChatSession.initialize() must be called before use.")
        return self._registry

    @property
    def mcp_manager(self) -> MCPManager:
        if self._mcp is None:
            raise RuntimeError("ChatSession.initialize() must be called before use.")
        return self._mcp

    @property
    def model_manager(self) -> ModelManager:
        if self._model_manager is None:
            raise RuntimeError("ChatSession.initialize() must be called before use.")
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

        # MCP must be initialized before the first send
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
        if self._client is None:
            raise RuntimeError("ChatSession.initialize() must be called before sending messages.")

        if not from_retry:
            self.history.append({"role": "user", "content": content})
        self._trim_history()

        messages = self._build_messages()
        tools = self._mcp.tools if self._mcp and self._mcp.tools else None

        console.print()  # blank line before response

        response_text = await self._execute_api_call(self._run_with_tools(messages, tools))

        if response_text:
            self.history.append({"role": "assistant", "content": response_text})

        return response_text

    async def _execute_api_call(self, coro: Coroutine[Any, Any, str]) -> str | None:
        """Await an API coroutine with consistent error mapping."""
        try:
            return await coro
        except openai.APIStatusError as exc:
            render_error(f"API error {exc.status_code}: {exc.message}")
            return None
        except openai.APIError as exc:
            template = _API_ERROR_MESSAGES.get(type(exc))
            if template:
                render_error(template.format(exc=exc) if "{exc}" in template else template)
            else:
                render_error(f"API error: {exc}")
            return None

    # ── Streaming + tool loop ─────────────────────────────────────────────────

    async def _run_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
    ) -> str:
        """Stream a response, handling tool calls for up to MAX_TOOL_ROUNDS."""
        if self._client is None:
            raise RuntimeError("Client not initialized.")

        local_messages: list[Message] = list(messages)

        for round_num in range(MAX_TOOL_ROUNDS + 1):
            # On the final round, disable tools to force a text response
            round_tools = tools if (tools and round_num < MAX_TOOL_ROUNDS) else None

            request_kwargs = self._build_request_kwargs(local_messages, round_tools)

            if self.settings.stream:
                text, tool_calls, usage = await self._stream_response(request_kwargs)
            else:
                text, tool_calls, usage = await self._blocking_response(request_kwargs)

            self._record_usage(usage)

            # No tool calls → final answer
            if not tool_calls:
                return text

            local_messages = await self._execute_tool_calls(local_messages, text, tool_calls)
            render_separator()

        return ""  # Unreachable; satisfies the type checker

    def _build_request_kwargs(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Assemble keyword arguments for the chat completions API call."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": self.settings.stream,
        }
        if self.settings.max_tokens is not None:
            kwargs["max_tokens"] = self.settings.max_tokens
        if self.settings.temperature is not None:
            kwargs["temperature"] = self.settings.temperature
        if self.settings.top_p is not None:
            kwargs["top_p"] = self.settings.top_p
        if self.settings.presence_penalty is not None:
            kwargs["presence_penalty"] = self.settings.presence_penalty
        if self.settings.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.settings.frequency_penalty
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return kwargs

    def _record_usage(self, usage: TokenUsage | None) -> None:
        """Update the running token total and optionally render usage stats."""
        if not usage:
            return
        self.total_tokens += usage.get("total_tokens", 0)
        if self.settings.show_token_usage and usage.get("total_tokens"):
            render_token_usage(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage.get("total_tokens", 0),
            )

    async def _execute_tool_calls(
        self,
        messages: list[Message],
        assistant_text: str,
        tool_calls: list[dict[str, Any]],
    ) -> list[Message]:
        """Run each tool call, append results, return the extended message list."""
        updated = list(messages)
        updated.append({"role": "assistant", "content": assistant_text or None, "tool_calls": tool_calls})

        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}

            render_tool_call(name, fn.get("arguments", ""))
            result = await self._mcp.call_tool(name, args)  # type: ignore[union-attr]
            render_tool_result(name, result)

            updated.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        return updated

    async def _stream_response(
        self, kwargs: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]], TokenUsage | None]:
        """Stream a response; collect text and any tool-call chunks."""
        if self._client is None:
            raise RuntimeError("Client not initialized.")

        tool_call_accumulator: dict[int, dict[str, Any]] = {}
        usage: TokenUsage | None = None

        async with self._renderer.live_display(self.model, show_thinking=True) as renderer:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue

                delta = choice.delta

                if delta.content:
                    renderer.push(delta.content)

                if delta.tool_calls:
                    self._accumulate_tool_call_chunks(tool_call_accumulator, delta.tool_calls)

                if hasattr(chunk, "usage") and chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

        tool_calls = list(tool_call_accumulator.values()) if tool_call_accumulator else []
        return renderer.text, tool_calls, usage

    @staticmethod
    def _accumulate_tool_call_chunks(
        accumulator: dict[int, dict[str, Any]],
        tc_deltas: Any,
    ) -> None:
        """Merge streaming tool-call delta chunks into the accumulator."""
        for tc_delta in tc_deltas:
            idx = tc_delta.index
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

    async def _blocking_response(
        self, kwargs: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]], TokenUsage | None]:
        """Non-streaming response (stream=False)."""
        if self._client is None:
            raise RuntimeError("Client not initialized.")

        non_stream_kwargs = {**kwargs, "stream": False}

        with console.status(f"[cyan]{self.model} is thinking…[/cyan]"):
            response = await self._client.chat.completions.create(**non_stream_kwargs)

        choice = response.choices[0]
        msg = choice.message
        text = msg.content or ""

        if text:
            render_message("assistant", text, self.settings.theme)

        tool_calls: list[dict[str, Any]] = []
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]

        usage: TokenUsage | None = None
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
        """Prepend the system prompt to conversation history."""
        return [{"role": "system", "content": self.system_prompt}, *self.history]

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