"""MCP (Model Context Protocol) client.

Reads ``mcp.json``, manages server processes/connections, and exposes
tools as OpenAI-compatible function definitions.

Supported transports:
  • ``stdio``  — launch a local subprocess
  • ``sse``    — connect to an HTTP SSE endpoint

To add a new transport, subclass ``Transport`` and register it in
``_make_transport`` — no other code needs to change (OCP).

mcp.json example::

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
          "env": { "MY_VAR": "value" }
        },
        "weather": {
          "url": "http://localhost:3001/sse"
        }
      }
    }

Requires the optional ``mcp`` package (``pip install openai-cli-chat[mcp]``).
If the package is not installed the module loads cleanly but ``MCPManager``
returns an empty tool list and logs a warning.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from rich.table import Table

from .utils import console

logger = logging.getLogger(__name__)

# Set to True to let MCP subprocess stderr pass through to the terminal.
# Useful when developing or debugging a server; should stay False in normal use
# so that chatty servers (e.g. mcp-remote) don't pollute the UI.
MCP_SUBPROCESS_STDERR: bool = False

# ── Optional mcp import ───────────────────────────────────────────────────────
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.types import Tool as MCPTool

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    logger.debug("mcp package not installed — MCP support disabled")


# ── Data types ────────────────────────────────────────────────────────────────


class ServerConfig:
    """Parsed entry from mcp.json."""

    def __init__(self, name: str, raw: dict[str, Any]) -> None:
        self.name = name
        # SSE transport
        self.url: str | None = raw.get("url")
        # Stdio transport
        self.command: str | None = raw.get("command")
        self.args: list[str] = raw.get("args", [])
        # Merge parent environment so subprocesses inherit PATH etc.
        self.env: dict[str, str] = {**os.environ, **raw.get("env", {})}
        self.enabled: bool = raw.get("enabled", True)

    def __repr__(self) -> str:
        if self.url:
            return f"ServerConfig(name={self.name!r}, url={self.url!r})"
        return f"ServerConfig(name={self.name!r}, command={self.command!r})"


# ── Transport protocol (OCP) ──────────────────────────────────────────────────


class Transport(ABC):
    """Abstract transport.  Subclass to add new connection mechanisms without
    modifying any existing code."""

    @abstractmethod
    @asynccontextmanager
    async def connect(self) -> AsyncIterator["ClientSession"]:
        """Yield a fully-initialised ``ClientSession``."""
        ...  # pragma: no cover


class StdioTransport(Transport):
    """Spawns a local subprocess and communicates over stdin/stdout.

    Subprocess stderr is suppressed by default (``MCP_SUBPROCESS_STDERR =
    False``) so that debug chatter from servers like ``mcp-remote`` never
    reaches the terminal.  Flip the constant to ``True`` at the top of this
    module to let stderr through while developing or debugging a server.

    The ``mcp`` library routes subprocess stderr through ``stdio_client``'s
    ``errlog`` parameter (not through ``StdioServerParameters``), so that is
    the correct place to apply suppression.
    """

    def __init__(self, config: ServerConfig) -> None:
        self._config = config

    @asynccontextmanager
    async def connect(self) -> AsyncIterator["ClientSession"]:
        params = StdioServerParameters(
            command=self._config.command,
            args=self._config.args,
            env=self._config.env,
        )
        # errlog is where the mcp library sends subprocess stderr.
        # Open /dev/null once for the lifetime of this connection so the
        # server's debug output never reaches the terminal.
        errlog = (
            open(os.devnull, "w")
            if not MCP_SUBPROCESS_STDERR
            else None          # None → mcp defaults to sys.stderr
        )
        try:
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        finally:
            if errlog is not None:
                errlog.close()


class SSETransport(Transport):
    """Connects to an HTTP Server-Sent Events endpoint."""

    def __init__(self, config: ServerConfig) -> None:
        self._config = config

    @asynccontextmanager
    async def connect(self) -> AsyncIterator["ClientSession"]:
        async with sse_client(self._config.url) as (read, write):  # type: ignore[arg-type]
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


def _make_transport(config: ServerConfig) -> Transport:
    """Return the correct ``Transport`` for *config*.

    Add new transport types here — callers don't need to change.
    """
    if config.url:
        return SSETransport(config)
    return StdioTransport(config)


# ── Main manager ──────────────────────────────────────────────────────────────


class MCPManager:
    """Manages a collection of MCP servers and their tools.

    Sessions are kept alive for the lifetime of the manager (held open via
    ``AsyncExitStack``) so every tool call reuses the existing connection
    rather than spawning a new subprocess or HTTP session per call.

    Call ``await manager.aclose()`` when the application exits to cleanly
    shut down all server processes and connections.
    """

    def __init__(self, mcp_file: Path, disabled: bool = False) -> None:
        self._mcp_file = mcp_file
        self._disabled = disabled
        self._servers: dict[str, ServerConfig] = {}
        self._tools: list[dict[str, Any]] = []          # OpenAI-format tool defs
        self._tool_server_map: dict[str, str] = {}      # tool_name → server_name
        self._sessions: dict[str, "ClientSession"] = {} # server_name → live session
        self._exit_stack = AsyncExitStack()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Load mcp.json, connect to all enabled servers, and discover tools."""
        if self._disabled:
            logger.debug("MCP disabled via --no-mcp")
            return

        if not _MCP_AVAILABLE:
            logger.warning(
                "mcp package not installed. Install with: pip install openai-cli-chat[mcp]"
            )
            return

        self._servers = self._load_mcp_json()
        if not self._servers:
            return

        enabled = {name: srv for name, srv in self._servers.items() if srv.enabled}
        logger.info("Loading tools from %d MCP server(s): %s", len(enabled), list(enabled))

        await self._exit_stack.__aenter__()
        for name, server in enabled.items():
            await self._connect_server(name, server)

        if self._tools:
            logger.info("MCP: loaded %d tool(s) total", len(self._tools))

    async def aclose(self) -> None:
        """Shut down all server connections and processes cleanly."""
        await self._exit_stack.aclose()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a named tool on its persistent session."""
        if not _MCP_AVAILABLE:
            return "Error: mcp package not installed."

        server_name = self._tool_server_map.get(tool_name)
        if not server_name:
            return f"Error: unknown tool {tool_name!r}"

        session = self._sessions.get(server_name)
        if session is None:
            return f"Error: server {server_name!r} is not connected."

        logger.debug("Calling tool %r on server %r", tool_name, server_name)
        try:
            result = await session.call_tool(tool_name, arguments)
            return _extract_tool_result(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("Tool call failed: %s", exc)
            return f"Error calling {tool_name}: {exc}"

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def tools(self) -> list[dict[str, Any]]:
        """OpenAI-compatible tool definitions."""
        return self._tools

    # ── Display ───────────────────────────────────────────────────────────────

    def display_servers(self) -> None:
        """Print a rich table of servers and their tools."""
        if not self._servers:
            console.print("[dim]No MCP servers configured.[/dim]")
            return

        table = Table(title="MCP Servers", show_header=True, header_style="bold magenta")
        table.add_column("Server", style="magenta")
        table.add_column("Transport", style="cyan")
        table.add_column("Endpoint / Command", style="white")
        table.add_column("Tools", style="green", justify="right")
        table.add_column("Status", justify="center")

        for name, srv in self._servers.items():
            transport_name = "sse" if srv.url else "stdio"
            endpoint = srv.url or f"{srv.command} {' '.join(srv.args)}"
            tool_count = sum(1 for s in self._tool_server_map.values() if s == name)
            connected = name in self._sessions
            status = "✓" if connected else ("○" if srv.enabled else "–")
            row_style = "" if srv.enabled else "dim"
            table.add_row(
                name, transport_name, endpoint, str(tool_count), status, style=row_style
            )

        console.print(table)

        if self._tools:
            console.print(f"\n[dim]Total tools available: {len(self._tools)}[/dim]")
            for t in self._tools:
                fn = t["function"]
                console.print(f"  [cyan]{fn['name']}[/cyan] — {fn.get('description', '')[:70]}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_mcp_json(self) -> dict[str, ServerConfig]:
        """Parse mcp.json; return an empty dict on any error."""
        if not self._mcp_file.exists():
            logger.debug("No mcp.json at %s", self._mcp_file)
            return {}
        try:
            raw = json.loads(self._mcp_file.read_text())
            servers_raw: dict[str, Any] = raw.get("mcpServers", {})
            return {name: ServerConfig(name, cfg) for name, cfg in servers_raw.items()}
        except (json.JSONDecodeError, OSError) as exc:
            console.print(f"[yellow]Warning:[/yellow] Could not parse mcp.json: {exc}")
            return {}

    async def _connect_server(self, name: str, server: ServerConfig) -> None:
        """Open a persistent session for *server* and register its tools."""
        try:
            transport = _make_transport(server)
            session: ClientSession = await self._exit_stack.enter_async_context(
                transport.connect()
            )
            result = await session.list_tools()
            tools: list[MCPTool] = result.tools

            self._sessions[name] = session
            for tool in tools:
                self._tools.append(_mcp_tool_to_openai(tool))
                self._tool_server_map[tool.name] = name
            logger.debug("Server %r: loaded %d tool(s)", name, len(tools))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not connect to MCP server %r: %s", name, exc)
            console.print(
                f"[yellow]Warning:[/yellow] MCP server [bold]{name}[/bold] unavailable: {exc}"
            )


# ── Conversion helpers ────────────────────────────────────────────────────────


def _mcp_tool_to_openai(tool: Any) -> dict[str, Any]:
    """Convert an MCP Tool to an OpenAI function-calling tool dict."""
    schema: dict[str, Any] = {}
    if hasattr(tool, "inputSchema") and tool.inputSchema:
        schema = (
            tool.inputSchema
            if isinstance(tool.inputSchema, dict)
            else tool.inputSchema.model_dump()
        )
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": getattr(tool, "description", "") or "",
            "parameters": schema or {"type": "object", "properties": {}},
        },
    }


def _extract_tool_result(result: Any) -> str:
    """Turn an MCP CallToolResult into a plain string."""
    if not result or not result.content:
        return ""
    parts: list[str] = []
    for item in result.content:
        if hasattr(item, "text"):
            parts.append(item.text)
        elif hasattr(item, "data"):
            parts.append(f"[binary data, {len(item.data)} bytes]")
        else:
            parts.append(str(item))
    return "\n".join(parts)