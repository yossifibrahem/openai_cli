"""MCP (Model Context Protocol) client.

Reads ``mcp.json``, manages server processes/connections, and exposes
tools as OpenAI-compatible function definitions.

Supported transports:
  • ``stdio``  — launch a local subprocess
  • ``sse``    — connect to an HTTP SSE endpoint

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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from rich.table import Table

from .utils import console

logger = logging.getLogger(__name__)

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
        self.env: dict[str, str] = {**os.environ, **raw.get("env", {})}
        self.enabled: bool = raw.get("enabled", True)

    @property
    def transport(self) -> str:
        return "sse" if self.url else "stdio"

    def __repr__(self) -> str:
        if self.url:
            return f"ServerConfig(name={self.name!r}, url={self.url!r})"
        return f"ServerConfig(name={self.name!r}, command={self.command!r})"


# ── Main manager ──────────────────────────────────────────────────────────────


class MCPManager:
    """Manages a collection of MCP servers and their tools."""

    def __init__(self, mcp_file: Path, disabled: bool = False) -> None:
        self._mcp_file = mcp_file
        self._disabled = disabled
        self._servers: dict[str, ServerConfig] = {}
        self._tools: list[dict[str, Any]] = []          # OpenAI-format tools
        self._tool_server_map: dict[str, str] = {}      # tool_name → server_name
        self._loaded = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Load mcp.json and discover tools from all servers."""
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

        enabled = {n: s for n, s in self._servers.items() if s.enabled}
        logger.info("Loading tools from %d MCP server(s): %s", len(enabled), list(enabled))

        for name, server in enabled.items():
            await self._load_server_tools(name, server)

        self._loaded = True
        if self._tools:
            logger.info("MCP: loaded %d tool(s) total", len(self._tools))

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool and return the result as a string."""
        if not _MCP_AVAILABLE:
            return "Error: mcp package not installed."

        server_name = self._tool_server_map.get(tool_name)
        if not server_name:
            return f"Error: unknown tool {tool_name!r}"

        server = self._servers[server_name]
        logger.debug("Calling tool %r on server %r", tool_name, server_name)

        try:
            async with _server_session(server) as session:
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

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def server_count(self) -> int:
        return len(self._servers)

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
            endpoint = srv.url or f"{srv.command} {' '.join(srv.args)}"
            tool_count = sum(1 for s in self._tool_server_map.values() if s == name)
            status = "✓" if srv.enabled else "○"
            style = "" if srv.enabled else "dim"
            table.add_row(name, srv.transport, endpoint, str(tool_count), status, style=style)

        console.print(table)

        if self._tools:
            console.print(f"\n[dim]Total tools available: {len(self._tools)}[/dim]")
            for t in self._tools:
                fn = t["function"]
                console.print(f"  [cyan]{fn['name']}[/cyan] — {fn.get('description', '')[:70]}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_mcp_json(self) -> dict[str, ServerConfig]:
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

    async def _load_server_tools(self, name: str, server: ServerConfig) -> None:
        """Connect to a server, list its tools, then disconnect."""
        try:
            async with _server_session(server) as session:
                result = await session.list_tools()
                tools: list[MCPTool] = result.tools

            for tool in tools:
                self._tools.append(_mcp_tool_to_openai(tool))
                self._tool_server_map[tool.name] = name
            logger.debug("Server %r: loaded %d tool(s)", name, len(tools))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not connect to MCP server %r: %s", name, exc)
            console.print(
                f"[yellow]Warning:[/yellow] MCP server [bold]{name}[/bold] unavailable: {exc}"
            )


# ── Transport abstraction ─────────────────────────────────────────────────────


@asynccontextmanager
async def _server_session(server: ServerConfig) -> AsyncIterator["ClientSession"]:
    """Open a short-lived MCP ClientSession for any supported transport.

    Centralises the stdio/SSE branching so callers never duplicate it.
    Yields a fully-initialised ``ClientSession`` and cleans up on exit.
    """
    if server.transport == "sse":
        async with sse_client(server.url) as (read, write):  # type: ignore[arg-type]
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    else:
        params = StdioServerParameters(
            command=server.command,
            args=server.args,
            env=server.env,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


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