"""Configuration management.

Priority (highest → lowest):
  1. CLI arguments
  2. Environment variables (OPENAI_* or AI_*)
  3. Config file (~/.config/openai-cli/config.json)
  4. Hard-coded defaults
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from .utils import console

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "openai-cli"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.json"
DEFAULT_MCP_FILE = Path.cwd() / "mcp.json"

AVAILABLE_THEMES = ["monokai", "dracula", "github-dark", "one-dark", "solarized-dark"]


class Settings(BaseSettings):
    """Application settings loaded from env, config file, and CLI args."""

    model_config = SettingsConfigDict(
        env_prefix="AI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API connection ───────────────────────────────────────────────────────
    api_key: str = Field(default="", alias="OPENAI_API_KEY")
    base_url: str = Field(default="https://api.openai.com/v1")
    timeout: float = Field(default=60.0, ge=1.0, le=600.0)
    max_retries: int = Field(default=2, ge=0, le=5)

    # ── Model defaults ───────────────────────────────────────────────────────
    model: str = Field(default="gpt-4o")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)

    # ── Chat behaviour ───────────────────────────────────────────────────────
    system_prompt: str = Field(default="You are a helpful assistant.")
    stream: bool = Field(default=True)
    context_window: int = Field(default=20, ge=1, le=200, description="Max messages to keep in history")
    save_history: bool = Field(default=True)

    # ── Appearance ───────────────────────────────────────────────────────────
    theme: str = Field(default="monokai")
    show_token_usage: bool = Field(default=True)
    show_model_in_prompt: bool = Field(default=True)
    word_wrap: bool = Field(default=True)

    # ── Paths ────────────────────────────────────────────────────────────────
    config_file: Path = Field(default=DEFAULT_CONFIG_FILE)
    mcp_file: Path = Field(default=DEFAULT_MCP_FILE)
    history_dir: Path = Field(default=DEFAULT_CONFIG_DIR / "history")
    log_file: Path | None = Field(default=None)

    # ── Runtime (not persisted) ──────────────────────────────────────────────
    no_mcp: bool = Field(default=False)
    log_level: str = Field(default="WARNING")

    @field_validator("api_key", mode="before")
    @classmethod
    def _resolve_api_key(cls, v: Any) -> str:
        """Also check OPENAI_API_KEY directly."""
        if not v:
            return os.environ.get("OPENAI_API_KEY", "")
        return str(v)

    @field_validator("theme")
    @classmethod
    def _validate_theme(cls, v: str) -> str:
        if v not in AVAILABLE_THEMES:
            logger.warning("Unknown theme %r, falling back to 'monokai'", v)
            return "monokai"
        return v

    def to_persist_dict(self) -> dict[str, Any]:
        """Return only user-configurable fields for serialisation."""
        skip = {"api_key", "no_mcp", "log_level", "log_file"}
        return {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in self.model_dump().items()
            if k not in skip and v is not None
        }


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_config_file(path: Path) -> dict[str, Any]:
    """Load JSON config file; return empty dict if not found."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        logger.debug("Loaded config from %s", path)
        return data
    except json.JSONDecodeError as exc:
        console.print(f"[yellow]Warning:[/yellow] Could not parse config file {path}: {exc}")
        return {}


def _args_to_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Convert non-None CLI args to a settings override dict."""
    mapping: dict[str, str] = {
        "model": "model",
        "system": "system_prompt",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "base_url": "base_url",
        "api_key": "api_key",
        "timeout": "timeout",
        "config_file": "config_file",
        "mcp_file": "mcp_file",
        "no_mcp": "no_mcp",
        "log_level": "log_level",
        "log_file": "log_file",
    }
    return {
        settings_key: getattr(args, arg_key)
        for arg_key, settings_key in mapping.items()
        if getattr(args, arg_key, None) is not None
    }


def load_config(args: argparse.Namespace | None = None) -> Settings:
    """Build Settings by merging file config + env + CLI overrides."""
    # Determine config file path early (CLI > default)
    config_file = DEFAULT_CONFIG_FILE
    if args and getattr(args, "config_file", None):
        config_file = Path(args.config_file)

    file_data = _load_config_file(config_file)
    cli_overrides = _args_to_overrides(args) if args else {}

    # Merge: file_data is base, CLI overrides win
    merged = {**file_data, **cli_overrides, "config_file": config_file}

    settings = Settings(**merged)

    if not settings.api_key:
        console.print(
            "[red]Error:[/red] No API key found.\n"
            "Set [bold]OPENAI_API_KEY[/bold] environment variable or add "
            "[bold]api_key[/bold] to your config file.\n"
            f"Config file: [dim]{settings.config_file}[/dim]"
        )
        raise ValueError("Missing API key")

    return settings


def save_config(settings: Settings) -> None:
    """Persist current settings to the config file."""
    settings.config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.config_file, "w") as f:
        json.dump(settings.to_persist_dict(), f, indent=2)
    logger.debug("Config saved to %s", settings.config_file)