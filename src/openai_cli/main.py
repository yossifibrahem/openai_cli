"""CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai",
        description="🤖  OpenAI CLI Chat",
        epilog="""
EXAMPLES
  ai                            Start interactive chat
  ai -m gpt-4o-mini             Use a specific model
  ai --base-url http://localhost:11434/v1   Use Ollama
  ai "What is 2+2?"             Single non-interactive message
  ai --setup                    Reconfigure

SLASH COMMANDS
  /help   /model   /models   /clear   /mcp   /multi   /exit
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api-key",  metavar="KEY", help="OpenAI API key")
    parser.add_argument("--base-url", metavar="URL", help="API base URL")
    parser.add_argument("-m", "--model", metavar="MODEL", help="Model to use")
    parser.add_argument("-s", "--system", metavar="PROMPT", dest="system_prompt",
                        help="System prompt")
    parser.add_argument("--mcp-file", type=Path, metavar="FILE",
                        help="Path to mcp.json (default: ./mcp.json)")
    parser.add_argument("--no-mcp", action="store_true", help="Disable MCP")
    parser.add_argument("--setup", action="store_true", help="Run setup wizard")
    parser.add_argument("message", nargs="?", metavar="MESSAGE",
                        help="Send a single message and exit")
    return parser


async def _run_interactive(settings: "Settings") -> None:  # noqa: F821
    from .chat import ChatSession
    session = ChatSession(settings)
    await session.initialize()
    await session.run()


async def _run_single(settings: "Settings", message: str) -> None:  # noqa: F821
    from .chat import ChatSession
    session = ChatSession(settings)
    await session.initialize()
    await session.send_message(message)


def main() -> None:
    from .config import Settings, config_exists, run_setup
    from .utils import console

    args = _build_parser().parse_args()

    if args.setup or not config_exists():
        settings = run_setup()
        if args.setup:
            return
    else:
        overrides = {
            "api_key": args.api_key,
            "base_url": args.base_url,
            "model": args.model,
            "system_prompt": args.system_prompt,
            "mcp_file": args.mcp_file,
            "no_mcp": args.no_mcp or None,
        }
        try:
            settings = Settings.load(overrides)
        except SystemExit:
            sys.exit(1)

    if args.message:
        asyncio.run(_run_single(settings, args.message))
        return

    try:
        asyncio.run(_run_interactive(settings))
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    main()
