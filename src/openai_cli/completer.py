"""prompt_toolkit autocompletion for slash commands."""

from __future__ import annotations

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from .commands import CommandRegistry


class SlashCommandCompleter(Completer):
    """Suggests slash command names and their argument choices."""

    def __init__(self, registry: CommandRegistry) -> None:
        self._registry = registry

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> list[Completion]:  # type: ignore[override]
        text = document.text_before_cursor

        # Only offer completions when the line starts with /
        if not text.startswith("/"):
            return []

        # Strip leading slash and split
        stripped = text[1:]
        parts = stripped.split(None, 1)

        # ── Complete the command name ─────────────────────────────────────────
        if len(parts) <= 1 and " " not in stripped:
            partial = stripped.lower()
            completions: list[Completion] = []
            seen: set[str] = set()
            for name in self._registry.all_names():
                if name not in seen and name.startswith(partial):
                    cmd = self._registry.get(name)
                    display_meta = cmd.description if cmd else ""
                    completions.append(
                        Completion(
                            text=name,
                            start_position=-len(partial),
                            display=f"/{name}",
                            display_meta=display_meta,
                        )
                    )
                    seen.add(name)
            return completions

        # ── Complete command arguments ────────────────────────────────────────
        cmd_name = parts[0].lower()
        cmd = self._registry.get(cmd_name)
        if cmd is None or not cmd.completer_choices:
            return []

        partial_arg = parts[1] if len(parts) > 1 else ""
        return [
            Completion(
                text=choice,
                start_position=-len(partial_arg),
                display=choice,
            )
            for choice in cmd.completer_choices
            if choice.startswith(partial_arg)
        ]
