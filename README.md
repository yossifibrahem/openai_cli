# OpenAI CLI Chat

A feature-rich command-line chat app for OpenAI-compatible APIs — built with streaming markdown, slash commands, MCP tool support, and flexible configuration.

```
$ ai

OpenAI CLI Chat  model: gpt-4o

[gpt-4o] You: Explain async/await in Python with an example

 ┌─────────────────────────────────────────────────────────────┐
 │ async/await lets you write non-blocking code that looks     │
 │ synchronous. Here's a simple example:                       │
 │                                                             │
 │ ```python                                                   │
 │ import asyncio                                              │
 │                                                             │
 │ async def fetch_data():                                     │
 │     await asyncio.sleep(1)   # non-blocking wait           │
 │     return {"result": "ok"}                                 │
 │                                                             │
 │ asyncio.run(fetch_data())                                   │
 │ ```                                                         │
 └─────────────────────────────────────────────────────────────┘

Tokens — prompt: 24 · completion: 87 · total: 111
```

---

## Features

- **Streaming markdown** — responses render live with Rich, including syntax-highlighted code blocks, tables, and lists
- **Slash commands** with tab-autocomplete — `/model`, `/clear`, `/save`, `/retry`, `/export`, and more
- **MCP server support** — connect any [Model Context Protocol](https://modelcontextprotocol.io) server via `mcp.json` for tool use
- **Connection settings** — custom base URL (Ollama, LM Studio, Azure), timeout, retries
- **Model management** — list models, switch mid-session, type aliases like `4o` instead of `gpt-4o`
- **Config file** — persist defaults in `~/.config/openai-cli/config.json`
- **History** — save/load conversations as JSON, export as Markdown
- **Progress indicators** — spinner while waiting, live token counter
- **Non-interactive mode** — pipe a single message: `ai "Summarise this: $(cat file.txt)"`

---

## Installation

### Requirements

- Python ≥ 3.10
- `pip`

### Quick install (recommended)

```bash
# Clone the repo
git clone https://github.com/yourname/openai-cli.git
cd openai-cli

# Install (editable so you can tweak the code)
pip install -e .

# Set your API key
export OPENAI_API_KEY=sk-...

# Start chatting
ai
```

### With MCP support

```bash
pip install -e ".[mcp]"
```

### With development tools

```bash
pip install -e ".[dev]"
```

---

## Configuration

### Environment variables

The fastest way — set in your shell or in a `.env` file in the working directory:

```bash
OPENAI_API_KEY=sk-...          # Required
AI_MODEL=gpt-4o                # Default model
AI_BASE_URL=https://...        # Custom endpoint
AI_TEMPERATURE=0.7
AI_SYSTEM_PROMPT="You are a helpful assistant."
AI_CONTEXT_WINDOW=20           # Messages to keep in memory
AI_THEME=monokai               # Code highlighting theme
AI_SHOW_TOKEN_USAGE=true
```

Copy `.env.example` to `.env` and fill in your values.

### Config file

`~/.config/openai-cli/config.json` — created automatically after first run. Edit it directly:

```json
{
  "model": "gpt-4o",
  "temperature": 0.7,
  "system_prompt": "You are a helpful assistant.",
  "theme": "monokai",
  "context_window": 20,
  "show_token_usage": true,
  "base_url": "https://api.openai.com/v1"
}
```

### Priority order

```
CLI flags > environment variables (.env / shell) > config file > defaults
```

---

## Usage

### Interactive mode

```bash
ai                         # Start with defaults
ai -m gpt-4o-mini          # Specify model
ai -m 4o                   # Use alias
ai -s "You are a pirate"   # Override system prompt
ai -t 0.2                  # Lower temperature
ai --no-stream             # Wait for full response
```

### Non-interactive (single message)

```bash
ai "What is the capital of France?"
echo "Explain this code:" | ai
ai "Summarise: $(cat report.txt)"
```

### Local / compatible endpoints

```bash
# Ollama
ai --base-url http://localhost:11434/v1 -m llama3

# LM Studio
ai --base-url http://localhost:1234/v1

# Azure OpenAI
ai --base-url https://YOUR_RESOURCE.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT \
   --api-key YOUR_AZURE_KEY

# Any OpenAI-compatible API
ai --base-url https://your-api.example.com/v1 --api-key your-key
```

---

## Slash Commands

Type any command inside the chat prompt. Tab-completion is available.

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/model [name]` | View or switch model |
| `/models` | List all available models |
| `/clear` | Clear conversation history |
| `/system [prompt]` | View or set system prompt |
| `/save [file]` | Save conversation as JSON |
| `/load [file]` | Load a saved conversation |
| `/history [n]` | Show recent messages |
| `/retry` | Retry last message |
| `/copy` | Copy last response to clipboard |
| `/tokens` | Show token usage |
| `/temp [value]` | View or set temperature |
| `/mcp` | Show MCP servers & tools |
| `/config` | Show current configuration |
| `/export [file]` | Export as Markdown |
| `/multi` | Enter multi-line input mode |
| `/exit` | Exit |

### Model aliases

| Alias | Full model ID |
|---|---|
| `4o` | gpt-4o |
| `4o-mini` | gpt-4o-mini |
| `4` | gpt-4 |
| `4-turbo` | gpt-4-turbo |
| `3.5` | gpt-3.5-turbo |
| `o1` | o1 |
| `o3` | o3 |
| `o4-mini` | o4-mini |

---

## MCP Servers (Tool Use)

[Model Context Protocol](https://modelcontextprotocol.io) lets the model call external tools — file system access, web search, database queries, and more.

### 1. Create `mcp.json` in your working directory

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "enabled": true
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": { "BRAVE_API_KEY": "your-key" },
      "enabled": true
    },
    "my-sse-server": {
      "url": "http://localhost:3001/sse",
      "enabled": false
    }
  }
}
```

### 2. Install the `mcp` extra

```bash
pip install "openai-cli-chat[mcp]"
```

### 3. Start chatting — the model picks up the tools automatically

```
[gpt-4o] You: List the files in /tmp

⚙ Tool Call
filesystem / list_directory
{"path": "/tmp"}

✓ filesystem
file1.txt
file2.py
...
```

Use `/mcp` to inspect connected servers and available tools.

---

## Code Themes

Set via `AI_THEME` env var or config file. Available themes:

- `monokai` (default)
- `dracula`
- `github-dark`
- `one-dark`
- `solarized-dark`

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint & format
ruff check src/
ruff format src/

# Type check
mypy src/
```

---

## Project Structure

```
openai-cli/
├── pyproject.toml          # Package config & dependencies
├── .env.example            # Environment variable template
├── mcp.json.example        # MCP server config template
└── src/
    └── openai_cli/
        ├── __init__.py
        ├── __main__.py     # python -m openai_cli
        ├── main.py         # CLI entry point & argparse
        ├── config.py       # Settings (pydantic-settings)
        ├── chat.py         # REPL loop, streaming, tool calls
        ├── renderer.py     # Rich markdown rendering
        ├── commands.py     # Slash command registry
        ├── completer.py    # Tab-autocomplete (prompt_toolkit)
        ├── mcp_client.py   # MCP server manager
        ├── models.py       # Model list & aliases
        └── utils.py        # Logging setup
```

---

## Troubleshooting

**`Error: No API key found`**
Set `OPENAI_API_KEY` in your environment or `.env` file.

**`Could not connect to the API`**
Check `--base-url` is correct and reachable. For Ollama, make sure `ollama serve` is running.

**MCP servers not connecting**
- Ensure `pip install openai-cli-chat[mcp]` was run
- Check the server `command`/`args` are valid (test manually in a terminal)
- Use `--log-level DEBUG` to see detailed connection logs
- Set `"enabled": false` to disable a problematic server

**Streaming looks glitchy**
Try `--no-stream` for a cleaner non-streaming experience, or reduce your terminal width.

---

## License

MIT
