"""`MemoryProvider` ABC for pluggable external memory backends.

A provider:

- Declares its tool schemas (`get_tool_schemas`) — the model sees these alongside
  the built-in `memory` tool and can call them.
- Handles tool calls routed by `MemoryManager` (`handle_tool_call`).
- Optionally implements lifecycle hooks (`sync_turn`, `on_session_end`,
  `on_memory_write`, `shutdown`). All hooks default to no-op so a provider only
  overrides what it actually uses.

Mini-memory **intentionally drops** hermes' per-turn ``prefetch`` /
``queue_prefetch`` / ``on_pre_compress`` hooks: mini-swe-agent has no context
compression and we avoid mutating cached prompt layers mid-session. Providers may
return a one-shot ``initial_context`` overlay for the first user task; after that
recall is tool-driven.

Concrete providers (Hindsight local, Mem0 OSS local) live in
`minisweagent.memory.providers`. See `notes/hermes-memory-digest.md` §6.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class MemoryProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap synchronous availability check. **No network calls.**"""
        ...

    @abstractmethod
    def initialize(self, session_id: str, *, home: Path, **kwargs: Any) -> None:
        """Called once per session. ``home`` is the per-instance state dir.

        Accepts ``**kwargs`` so the manager / agent can grow extra context
        (``platform``, ``user_id``, …) without breaking provider implementations.
        Providers should ignore unknown kwargs.
        """
        ...

    @abstractmethod
    def get_tool_schemas(self) -> list[dict]:
        """Return tool schemas in **OpenAI tool-call wrapper format**::

            {
                "type": "function",
                "function": {
                    "name": "<tool_name>",
                    "description": "<...>",
                    "parameters": {"type": "object", "properties": {...}, "required": [...]},
                },
            }

        This matches `BASH_TOOL` in `minisweagent.models.utils.actions_toolcall`,
        so provider tools sit alongside built-in tools without re-wrapping.
        Hermes plugins use the *unwrapped* form (top-level ``name``/``parameters``);
        when porting one, wrap it before returning here.
        """

    def handle_tool_call(self, name: str, args: dict) -> dict:
        """Route a tool call. Return a JSON-serializable result ``dict``.

        Convention (matches `BuiltinMemory`): include ``"success": bool`` and either
        ``"error": "..."`` (on failure) or domain-specific keys on success.

        Default implementation surfaces an error so context-only providers (no
        tools) don't have to override — but must not be reached when
        ``get_tool_schemas`` is non-empty. Manager only routes tool names this
        provider actually advertised, so the default catches programming errors.
        """
        return {"success": False, "error": f"Provider {self.name!r} does not handle tool {name!r}"}

    def system_prompt_block(self) -> str:
        """Static block injected into system prompt at session start."""
        return ""

    def initial_context(self, query: str) -> str:
        """Optional context prepended to the first user task message."""
        return ""

    def sync_turn(self, user: str, assistant: str) -> None:
        """Persist a completed turn. **Must not block** (run on a daemon thread)."""

    def on_session_end(self, messages: list[dict]) -> None:
        """Final extraction / flush at session boundary."""

    def on_memory_write(self, action: str, content: str) -> None:
        """Mirror a built-in MEMORY.md write to the provider backend."""

    def shutdown(self) -> None:
        """Clean up connections / threads at process exit."""
