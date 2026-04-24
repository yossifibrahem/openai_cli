# OpenAI CLI Chat

A feature-rich terminal chat client for OpenAI-compatible APIs. Supports streaming, MCP tool-calling, conversation history, slash commands, and rich markdown rendering.

## Installation

```bash
pip install openai-cli-chat

# With MCP support:
pip install openai-cli-chat[mcp]
```

## Quick Start

```bash
export OPENAI_API_KEY=sk-...

ai                          # Interactive chat
ai "What is 2+2?"           # Single message, non-interactive
ai -m gpt-4o-mini           # Use a specific model
ai -s "You are a pirate"    # Override system prompt
ai -t 0.2                   # Lower temperature (more focused)
ai --no-stream              # Disable streaming
ai --list-models            # List available models
```

## Configuration

Settings are resolved in priority order (highest wins):

1. **CLI arguments** (`--model`, `--temperature`, etc.)
2. **Environment variables** (`AI_MODEL`, `AI_TEMPERATURE`, `OPENAI_API_KEY`, …)
3. **Config file** (`~/.config/openai-cli/config.json`)
4. **Built-in defaults**

A `.env` file in the current working directory is loaded automatically.

### Config File Format

```json
{
  "model": "gpt-4o",
  "temperature": 0.7,
  "system_prompt": "You are a helpful assistant.",
  "stream": true,
  "theme": "monokai",
  "context_window": 20,
  "show_token_usage": true
}
```

### Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your API key |
| `AI_MODEL` | Default model |
| `AI_BASE_URL` | API base URL (e.g. for Ollama) |
| `AI_TEMPERATURE` | Sampling temperature |
| `AI_SYSTEM_PROMPT` | System prompt |
| `AI_STREAM` | `true` / `false` |
| `AI_CONTEXT_WINDOW` | Max messages kept in history |
| `AI_THEME` | Syntax highlight theme |

## Slash Commands

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/model [name]` | View or switch model |
| `/models` | List all available models |
| `/clear` | Clear conversation history |
| `/system [prompt]` | View or set system prompt |
| `/save [file]` | Save conversation as JSON |
| `/load [file]` | Load a saved conversation |
| `/history [n]` | Show recent history |
| `/retry` | Retry last message |
| `/copy` | Copy last response to clipboard |
| `/tokens` | Show token usage |
| `/temp [value]` | View or set temperature |
| `/mcp` | Show MCP servers & tools |
| `/export [file]` | Export as Markdown |
| `/multi` | Multi-line input mode |
| `/exit` | Exit |

## MCP (Model Context Protocol)

Place an `mcp.json` in your working directory (or point to one with `--mcp-file`):

```json
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
```

Supported transports: **stdio** (subprocess) and **SSE** (HTTP).

Requires: `pip install openai-cli-chat[mcp]`

## Using with Ollama / Local Models

```bash
ai --base-url http://localhost:11434/v1 -m llama3
```

No API key is required for local endpoints; set a dummy value if needed:
```bash
export OPENAI_API_KEY=ollama
```

## Themes

Available syntax highlighting themes: `monokai` (default), `dracula`, `github-dark`, `one-dark`, `solarized-dark`.

Set via config file, `AI_THEME` env var, or the config JSON.

## Architecture

```
main.py          CLI entry point, argument parsing
config.py        Settings (pydantic-settings, env + file + CLI merge)
chat.py          ChatSession — REPL loop, message sending, tool-call loop
commands.py      Slash command registry and handlers
completer.py     prompt_toolkit autocompleter for slash commands
mcp_client.py    MCP server manager (stdio + SSE transports)
models.py        Model listing and validation
renderer.py      Rich streaming markdown renderer (incremental block commits)
utils.py         Shared Rich Console singleton, logging setup
```

## Development

```bash
git clone https://github.com/your-org/openai-cli-chat
cd openai-cli-chat
pip install -e ".[dev,mcp]"
```

Run tests:
```bash
pytest
```