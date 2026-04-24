"""MCP (Model Context Protocol) client.

Reads ``mcp.json``, manages server processes/connections, and exposes
tools as OpenAI-compatible function definitions.

mcp.json example::

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        },
        "weather": {
          "url": "http://localhost:3001/sse"
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from rich.table import Table

from .utils import console

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    logger.debug("mcp package not installed — MCP support disabled")


@dataclass
class ServerConfig:
    name: str
    url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> "ServerConfig":
        return cls(
            name=name,
            url=raw.get("url"),
            command=raw.get("command"),
            args=raw.get("args", []),
            env={**os.environ, **raw.get("env", {})},
            enabled=raw.get("enabled", True),
        )


@asynccontextmanager
async def _connect(config: ServerConfig) -> AsyncIterator["ClientSession"]:
    """Open an MCP ClientSession for stdio or SSE transport."""
    if config.url:
        async with sse_client(config.url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    else:
        params = StdioServerParameters(
            command=config.command, args=config.args, env=config.env
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


class MCPManager:
    def __init__(self, mcp_file: Path, disabled: bool = False) -> None:
        self._mcp_file = mcp_file
        self._disabled = disabled
        self._servers: dict[str, ServerConfig] = {}
        self._tools: list[dict[str, Any]] = []
        self._tool_server_map: dict[str, str] = {}
        self._sessions: dict[str, "ClientSession"] = {}
        self._exit_stack = AsyncExitStack()

    async def initialize(self) -> None:
        if self._disabled or not _MCP_AVAILABLE:
            if not self._disabled and not _MCP_AVAILABLE:
                logger.warning("mcp package not installed. Install with: pip install openai-cli-chat[mcp]")
            return

        self._servers = self._load_mcp_json()
        if not self._servers:
            return

        await self._exit_stack.__aenter__()
        for name, server in self._servers.items():
            if server.enabled:
                await self._connect_server(name, server)

        if self._tools:
            logger.info("MCP: loaded %d tool(s) total", len(self._tools))

    async def aclose(self) -> None:
        await self._exit_stack.aclose()

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if not _MCP_AVAILABLE:
            return "Error: mcp package not installed."
        server_name = self._tool_server_map.get(tool_name)
        session = self._sessions.get(server_name) if server_name else None
        if not session:
            return f"Error: tool {tool_name!r} not available."
        try:
            result = await session.call_tool(tool_name, arguments)
            return _extract_tool_result(result)
        except Exception as exc:
            logger.error("Tool call failed: %s", exc)
            return f"Error calling {tool_name}: {exc}"

    @property
    def tools(self) -> list[dict[str, Any]]:
        return self._tools

    def display_servers(self) -> None:
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
            transport = "sse" if srv.url else "stdio"
            endpoint = srv.url or f"{srv.command} {' '.join(srv.args)}"
            tool_count = sum(1 for s in self._tool_server_map.values() if s == name)
            status = "✓" if name in self._sessions else ("○" if srv.enabled else "–")
            table.add_row(name, transport, endpoint, str(tool_count), status,
                          style="" if srv.enabled else "dim")

        console.print(table)
        if self._tools:
            console.print(f"\n[dim]Total tools available: {len(self._tools)}[/dim]")
            for t in self._tools:
                fn = t["function"]
                console.print(f"  [cyan]{fn['name']}[/cyan] — {fn.get('description', '')[:70]}")

    def _load_mcp_json(self) -> dict[str, ServerConfig]:
        if not self._mcp_file.exists():
            return {}
        try:
            raw = json.loads(self._mcp_file.read_text())
            return {
                name: ServerConfig.from_dict(name, cfg)
                for name, cfg in raw.get("mcpServers", {}).items()
            }
        except (json.JSONDecodeError, OSError) as exc:
            console.print(f"[yellow]Warning:[/yellow] Could not parse mcp.json: {exc}")
            return {}

    async def _connect_server(self, name: str, server: ServerConfig) -> None:
        try:
            session = await self._exit_stack.enter_async_context(_connect(server))
            result = await session.list_tools()
            self._sessions[name] = session
            for tool in result.tools:
                self._tools.append(_mcp_tool_to_openai(tool))
                self._tool_server_map[tool.name] = name
            logger.debug("Server %r: loaded %d tool(s)", name, len(result.tools))
        except Exception as exc:
            logger.warning("Could not connect to MCP server %r: %s", name, exc)
            console.print(f"[yellow]Warning:[/yellow] MCP server [bold]{name}[/bold] unavailable: {exc}")


def _mcp_tool_to_openai(tool: Any) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    if hasattr(tool, "inputSchema") and tool.inputSchema:
        schema = (
            tool.inputSchema if isinstance(tool.inputSchema, dict)
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
    if not result or not result.content:
        return ""
    parts = []
    for item in result.content:
        if hasattr(item, "text"):
            parts.append(item.text)
        elif hasattr(item, "data"):
            parts.append(f"[binary data, {len(item.data)} bytes]")
        else:
            parts.append(str(item))
    return "\n".join(parts)