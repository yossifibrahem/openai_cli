# OpenAI CLI Chat

A minimal terminal chat app for OpenAI-compatible APIs — streaming markdown, MCP tool support, and slash commands.

```
$ ai

OpenAI CLI Chat  model: gpt-4o

[gpt-4o] You: Explain async/await in Python

async/await lets you write non-blocking code that looks synchronous…
```

---

## Installation

```bash
pip install -e .
export OPENAI_API_KEY=sk-...
ai
```

First run launches a setup wizard automatically. Re-run it any time with `ai --setup`.

### With MCP support

```bash
pip install -e ".[mcp]"
```

---

## Usage

```bash
ai                                        # Interactive chat
ai -m gpt-4o-mini                         # Specific model
ai -s "You are a pirate"                  # Override system prompt
ai --base-url http://localhost:11434/v1   # Ollama / local endpoint
ai "What is 2+2?"                         # Single message, non-interactive
ai --setup                                # Reconfigure
```

---

## Configuration

Settings are loaded in this priority order:

```
CLI flags > environment variables > config file > defaults
```

**Environment variables**

```bash
OPENAI_API_KEY=sk-...
AI_MODEL=gpt-4o
AI_BASE_URL=https://api.openai.com/v1
AI_SYSTEM_PROMPT="You are a helpful assistant."
```

**Config file** — `~/.config/openai-cli/config.json` (created on first run)

```json
{
  "api_key": "sk-...",
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4o",
  "system_prompt": "You are a helpful assistant."
}
```

---

## Slash Commands

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/model [name]` | View or switch model |
| `/models` | List available models |
| `/clear` | Clear conversation history |
| `/mcp` | Show MCP servers & tools |
| `/multi` | Enter multi-line input mode |
| `/exit` | Exit |

Tab-completion is available for all slash commands.

---

## MCP (Tool Use)

Place an `mcp.json` in the current directory:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    },
    "my-sse-server": {
      "url": "http://localhost:3001/sse"
    }
  }
}
```

Requires `pip install ".[mcp]"`. Use `/mcp` to inspect connected servers and tools.

---

## Project Structure

```
src/openai_cli/
├── main.py        # Entry point & argument parsing
├── config.py      # Settings (load/save, setup wizard)
├── chat.py        # REPL loop, streaming, tool calls
├── renderer.py    # Rich streaming markdown renderer
├── commands.py    # Slash command registry
├── completer.py   # Tab-autocomplete (prompt_toolkit)
├── mcp_client.py  # MCP server manager
├── models.py      # Model listing
└── utils.py       # Shared console
```
