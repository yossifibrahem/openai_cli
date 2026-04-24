"""CLI entry point — argument parsing and application startup."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("openai-cli-chat")
except Exception:
    __version__ = "0.1.0"


# ── Argument parser ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai",
        description="🤖  OpenAI CLI Chat — chat with AI models from your terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
EXAMPLES
  ai                              Start interactive chat with default model
  ai -m gpt-4o-mini               Use a specific model
  ai -m 4o                        Use an alias  (4o → gpt-4o)
  ai -s "You are a pirate"        Override the system prompt
  ai -t 0.2                       Lower temperature (more focused)
  ai --base-url http://localhost:11434/v1   Use a local / Ollama endpoint
  ai --list-models                List available models and exit
  ai "What is 2+2?"               Single non-interactive message

SLASH COMMANDS (inside the chat)
  /help                           Show all slash commands
  /model [name]                   View or switch model
  /models                         List all available models
  /clear                          Clear conversation history
  /system [prompt]                View or set system prompt
  /save [file]                    Save conversation as JSON
  /load [file]                    Load a saved conversation
  /history [n]                    Show recent history
  /retry                          Retry last message
  /copy                           Copy last response to clipboard
  /tokens                         Show token usage
  /temp [value]                   View or set temperature
  /mcp                            Show MCP servers & tools
  /export [file]                  Export as Markdown
  /multi                          Enter multi-line input mode
  /exit                           Exit

MCP (Model Context Protocol)
  Place an mcp.json file in the current directory (or use --mcp-file).
  See mcp.json.example for format. Requires: pip install openai-cli-chat[mcp]

CONFIGURATION
  Default config file: ~/.config/openai-cli/config.json
  Environment variables: OPENAI_API_KEY, AI_MODEL, AI_BASE_URL, AI_TEMPERATURE, …
  .env file in the current directory is loaded automatically.
  Run `ai --setup` to reconfigure the wizard.
""",
    )

    # ── API / connection ──────────────────────────────────────────────────────
    conn = parser.add_argument_group("Connection")
    conn.add_argument("--api-key", metavar="KEY", help="OpenAI API key (overrides OPENAI_API_KEY)")
    conn.add_argument("--base-url", metavar="URL", help="API base URL (e.g. for local/Ollama)")
    conn.add_argument("--timeout", type=float, metavar="SECS", help="Request timeout (default: 60)")

    # ── Model / generation ────────────────────────────────────────────────────
    model_grp = parser.add_argument_group("Model & generation")
    model_grp.add_argument("-m", "--model", metavar="MODEL", help="Model to use (alias or full ID)")
    model_grp.add_argument("-s", "--system", metavar="PROMPT", dest="system",
                           help="System prompt")
    model_grp.add_argument("-t", "--temperature", type=float, metavar="T",
                           help="Sampling temperature (0.0–2.0)")
    model_grp.add_argument("--max-tokens", type=int, metavar="N", help="Max tokens per response")
    model_grp.add_argument("--no-stream", action="store_true",
                           help="Disable streaming (wait for full response)")

    # ── MCP ───────────────────────────────────────────────────────────────────
    mcp_grp = parser.add_argument_group("MCP")
    mcp_grp.add_argument("--mcp-file", type=Path, metavar="FILE",
                         help="Path to mcp.json (default: ./mcp.json)")
    mcp_grp.add_argument("--no-mcp", action="store_true", help="Disable MCP servers")

    # ── Config / info ─────────────────────────────────────────────────────────
    info_grp = parser.add_argument_group("Info & config")
    info_grp.add_argument("--config-file", type=Path, metavar="FILE",
                          help="Config file (default: ~/.config/openai-cli/config.json)")
    info_grp.add_argument("--list-models", action="store_true",
                          help="List available models and exit")
    info_grp.add_argument("--setup", action="store_true",
                          help="Run the setup wizard to reconfigure")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_grp = parser.add_argument_group("Logging")
    log_grp.add_argument("--log-level",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                         default="WARNING",
                         help="Log verbosity (default: WARNING)")
    log_grp.add_argument("--log-file", type=Path, metavar="FILE",
                         help="Write logs to file")

    # ── Version ───────────────────────────────────────────────────────────────
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")

    # ── Positional (non-interactive mode) ─────────────────────────────────────
    parser.add_argument("message", nargs="?", metavar="MESSAGE",
                        help="Send a single message and exit (non-interactive)")

    return parser


# ── Helpers ───────────────────────────────────────────────────────────────────


def _normalize_stream_arg(args: argparse.Namespace) -> None:
    """Translate --no-stream flag into args.stream = False for config loader.

    The config loader maps args.stream → Settings.stream (see _CLI_TO_SETTINGS
    in config.py).  argparse only gives us ``no_stream=True``; we convert that
    to an explicit ``stream=False`` so the mapping picks it up.
    """
    if getattr(args, "no_stream", False):
        args.stream = False  # type: ignore[attr-defined]


# ── Async entrypoints ─────────────────────────────────────────────────────────


async def _list_models(settings: "Settings") -> None:  # noqa: F821
    from .models import ModelManager
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
    manager = ModelManager(settings, client)
    await manager.list_and_display()


async def _single_message(settings: "Settings", message: str) -> None:  # noqa: F821
    from .chat import ChatSession

    session = ChatSession(settings)
    await session.initialize()
    await session.send_message(message)


async def _interactive(settings: "Settings") -> None:  # noqa: F821
    from .chat import ChatSession

    session = ChatSession(settings)
    await session.initialize()
    await session.run()


# ── Public entry point ────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point — called by the ``ai`` script."""
    from .config import config_exists, load_config
    from .utils import console, setup_logging

    parser = _build_parser()
    args = parser.parse_args()

    setup_logging(args.log_level, getattr(args, "log_file", None))

    # Translate --no-stream into args.stream before config loading
    _normalize_stream_arg(args)

    # ── Check for first run or --setup flag ────────────────────────────────────────────
    if args.setup or not config_exists():
        from .wizard import run_wizard
        import os

        os.environ.setdefault("TERM", "xterm-color")
        settings = asyncio.run(run_wizard())
        if args.setup:
            return
    else:
        try:
            settings = load_config(args)
        except ValueError:
            sys.exit(1)

    # ── Non-interactive modes ─────────────────────────────────────────────────
    if args.list_models:
        asyncio.run(_list_models(settings))
        return

    if args.message:
        asyncio.run(_single_message(settings, args.message))
        return

    # ── Interactive REPL ──────────────────────────────────────────────────────
    try:
        asyncio.run(_interactive(settings))
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    main()