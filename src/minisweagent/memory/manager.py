"""`MemoryManager` — composes the built-in MEMORY.md store with an optional
external provider, an optional FTS-backed session store, and an optional
LLM consolidation pass; routes tool calls and dispatches lifecycle hooks.

Single-provider rule (see digest §5.1): only one external provider may be
registered at a time. Built-in memory is always active. Sessions / consolidation
are independent opt-ins controlled via ``MemoryManagerConfig``.

Diverges from hermes intentionally: builtin store is **owned** by the manager
(simpler), not registered as a provider with name=='builtin'. Single external
provider is enforced strictly (hermes accepts builtin + 1 external).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path

from minisweagent.memory.builtin import BuiltinMemory, BuiltinMemoryConfig
from minisweagent.memory.consolidation import consolidate_memory
from minisweagent.memory.filesystem import FileSystemMemory, FileSystemMemoryConfig
from minisweagent.memory.provider import MemoryProvider
from minisweagent.memory.session_store import SessionStore, summarize_session

logger = logging.getLogger(__name__)

MEMORY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "memory",
        "description": (
            "Save durable engineering knowledge to MEMORY.md so it survives across trials and "
            "instances. Memory is injected into the system prompt at session start as a frozen "
            "snapshot — writes during this session take effect in the *next* session, but every "
            "tool response shows you the live entries list so you can plan consolidations.\n\n"
            "WHEN TO SAVE (be proactive, do not wait to be asked):\n"
            "- Project environment facts not obvious from the task input (language version, "
            "framework, test runner, build command, container/runtime quirks).\n"
            "- Build/test gotchas you verified ('pytest must run from repo root with PYTHONPATH=.', "
            "'this repo's CI uses tox -e py311', 'tests/integration/* are flaky — retry').\n"
            "- Concrete bug-fix idioms specific to this codebase that you confirmed work.\n"
            "- Failed approaches you already ruled out so future trials don't repeat them.\n"
            "- Repo conventions (lint config, docstring style, type-checking rules, line width).\n\n"
            "PRIORITY: build/test infrastructure > stable bug-fix patterns > coding conventions. "
            "The most valuable entries prevent the next trial from re-discovering the same gotcha.\n\n"
            "DO NOT SAVE:\n"
            "- The current issue's text (already in the task input).\n"
            "- Raw logs, diffs, command output dumps, or stack traces.\n"
            "- Trial-local state (a path you cd'd to, a temp file you created).\n"
            "- Things easy to re-discover (file locations findable by ripgrep).\n\n"
            "ACTIONS: 'add' (new entry), 'replace' (rewrite the entry uniquely identified by "
            "old_text), 'remove' (drop the entry uniquely identified by old_text). "
            "Keep entries compact and information-dense; consolidate via 'replace' when memory "
            "is over 80% full."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "replace", "remove"]},
                "content": {"type": "string", "description": "New entry text. Required for add/replace."},
                "old_text": {
                    "type": "string",
                    "description": "Unique substring of an existing entry. Required for replace/remove.",
                },
            },
            "required": ["action"],
        },
    },
}

SESSION_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "session_search",
        "description": (
            "Full-text search over transcripts of *past* sessions stored locally in a SQLite "
            "FTS5 index. This is recall for previous coding trials; past sessions are NOT "
            "injected into the prompt, so you have to query for what you want.\n\n"
            "USE THIS PROACTIVELY when the current repo, file, failing test, stack trace, "
            "or dependency looks like something a previous trial may have touched. At the "
            "start of an issue, if there is a clear cross-session signal (same repo, similar "
            "error, same test framework, or familiar file path), search before deep "
            "investigation so you can reuse prior lessons. Skip it when the task looks new "
            "or the signal is weak; noisy searches are not useful.\n\n"
            "Returns up to `limit` past trial/session results. Each result includes "
            "the session_id, a synchronous extractive summary, and up to three matching "
            "snippets with nearby transcript context. Use the summary to decide whether "
            "the past trial is relevant, then inspect the snippets for concrete commands, "
            "files, tests, and errors.\n\n"
            "Query syntax is SQLite FTS5: bare words AND together; use `OR`, prefix `term*`, "
            "or quoted phrases `\"...\"` for precision. Keep queries short (1-4 keywords)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "FTS5 query string."},
                "limit": {
                    "type": "integer",
                    "description": "Max past sessions to return (default 5, hard cap 20).",
                },
            },
            "required": ["query"],
        },
    },
}


@dataclass
class ConsolidationConfig:
    """LLM-driven memory consolidation triggers (digest §5 / "Memory Flush").

    Both triggers are off by default — each one costs an extra ``model.query``.
    """

    on_session_end: bool = False
    every_n_steps: int = 0
    max_actions: int = 3
    summary_max_chars: int = 4000


@dataclass
class MemoryManagerConfig:
    home: Path = field(default_factory=lambda: Path.home() / ".mini-memory")
    char_limit: int = 48_000
    builtin_enabled: bool = True
    sessions_enabled: bool = True
    sessions_path: Path | None = None
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    filesystem: FileSystemMemoryConfig = field(default_factory=FileSystemMemoryConfig)


class MemoryManager:
    def __init__(
        self,
        *,
        config: MemoryManagerConfig | None = None,
        builtin: BuiltinMemory | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self.config = config or MemoryManagerConfig()
        if not self.config.builtin_enabled:
            if builtin is not None:
                raise ValueError("builtin_enabled is false but a BuiltinMemory was passed")
            self.builtin = None
        elif builtin is not None:
            if builtin.config.path.parent != self.config.home:
                raise ValueError(
                    f"BuiltinMemory.path.parent ({builtin.config.path.parent}) must equal "
                    f"MemoryManagerConfig.home ({self.config.home}); providers derive subdirs from `home` "
                    "and would otherwise diverge from the built-in store's location."
                )
            self.builtin = builtin
        else:
            self.builtin = BuiltinMemory(
                BuiltinMemoryConfig(path=self.config.home / "MEMORY.md", char_limit=self.config.char_limit)
            )
        self.session_store: SessionStore | None
        if session_store is not None:
            self.session_store = session_store
        elif self.config.sessions_enabled:
            self.session_store = SessionStore(self.config.sessions_path or self.config.home / "sessions.db")
        else:
            self.session_store = None
        self.filesystem_memory: FileSystemMemory | None
        if self.config.filesystem.enabled:
            fs_config = self.config.filesystem
            if fs_config.home is None:
                fs_config = FileSystemMemoryConfig(
                    home=self.config.home,
                    enabled=fs_config.enabled,
                    chain_id=fs_config.chain_id,
                )
            elif fs_config.home != self.config.home:
                fs_config = FileSystemMemoryConfig(
                    home=fs_config.home,
                    enabled=fs_config.enabled,
                    chain_id=fs_config.chain_id,
                )
            self.filesystem_memory = FileSystemMemory(fs_config)
        else:
            self.filesystem_memory = None
        self.provider: MemoryProvider | None = None
        self._provider_tool_names: set[str] = set()
        self._session_id: str = ""
        self._last_memory_write_n_calls = 0
        self._last_consolidation_n_calls = 0
        self._memory_write_pending = False

    @classmethod
    def from_config(cls, cfg: dict) -> MemoryManager:
        """Build a manager from a yaml-style dict.

        Recognized keys::

            home: ~/.mini-memory               # str, expanded with ~
            char_limit: 48000                  # int
            builtin_enabled: true              # bool, default true — false disables MEMORY.md tool + snapshot
            sessions_enabled: true             # bool, default true
            sessions_path: ~/.mini-memory/sessions.db  # optional override
            consolidation:                     # opt-in LLM consolidation
              on_session_end: false
              every_n_steps: 0
              max_actions: 3
              summary_max_chars: 4000
            filesystem:
              enabled: false
              chain_id: default
            provider: hindsight | mem0         # optional, default None
            hindsight: {...}                   # passed to HindsightConfig(**...)
            mem0: {...}                        # passed to Mem0Config(**...)

        Provider sub-configs are imported lazily — installing the package for the
        chosen provider (`mini-swe-agent[hindsight]` / `[mem0]`) is enough.
        """
        mgr_kwargs: dict = {"char_limit": cfg.get("char_limit", 48_000)}
        if cfg.get("home"):
            mgr_kwargs["home"] = Path(str(cfg["home"])).expanduser()
        if "builtin_enabled" in cfg:
            mgr_kwargs["builtin_enabled"] = bool(cfg["builtin_enabled"])
        if "sessions_enabled" in cfg:
            mgr_kwargs["sessions_enabled"] = bool(cfg["sessions_enabled"])
        if cfg.get("sessions_path"):
            mgr_kwargs["sessions_path"] = Path(str(cfg["sessions_path"])).expanduser()
        if (consol := cfg.get("consolidation")) is not None:
            mgr_kwargs["consolidation"] = ConsolidationConfig(**consol)
        if (filesystem := cfg.get("filesystem")) is not None:
            valid_fs_keys = {item.name for item in fields(FileSystemMemoryConfig)}
            fs_kwargs = {key: value for key, value in dict(filesystem).items() if key in valid_fs_keys}
            fs_kwargs["home"] = Path(str(fs_kwargs.get("home") or mgr_kwargs.get("home") or Path.home() / ".mini-memory")).expanduser()
            mgr_kwargs["filesystem"] = FileSystemMemoryConfig(**fs_kwargs)
        manager = cls(config=MemoryManagerConfig(**mgr_kwargs))
        provider = cfg.get("provider")
        if provider == "hindsight":
            from minisweagent.memory.providers import HindsightConfig, HindsightProvider

            manager.register(HindsightProvider(HindsightConfig(**cfg.get("hindsight", {}))))
        elif provider == "mem0":
            from minisweagent.memory.providers import Mem0Config, Mem0Provider

            manager.register(Mem0Provider(Mem0Config(**cfg.get("mem0", {}))))
        elif provider is not None:
            raise ValueError(f"Unknown memory provider: {provider!r}. Use 'hindsight' or 'mem0'.")
        return manager

    def register(self, provider: MemoryProvider) -> None:
        if self.provider is not None:
            raise RuntimeError(
                f"A memory provider {self.provider.name!r} is already registered (single-provider rule)."
            )
        self.provider = provider
        self._provider_tool_names = {s["function"]["name"] for s in provider.get_tool_schemas()}

    def initialize(self, session_id: str, **kwargs) -> None:
        """Refresh the frozen snapshot and initialize the provider for this session.

        Extra kwargs (``platform``, ``user_id``, …) are forwarded to the provider
        so future signatures can grow without changing the manager.
        """
        self._session_id = session_id
        self._last_memory_write_n_calls = 0
        self._last_consolidation_n_calls = 0
        self._memory_write_pending = False
        if self.builtin is not None:
            self.builtin.load_snapshot()
        if self.provider is not None:
            self.provider.initialize(session_id, home=self.config.home, **kwargs)
        if self.filesystem_memory is not None:
            self.filesystem_memory.initialize(session_id, **kwargs)

    def system_prompt_block(self) -> str:
        parts = []
        if self.builtin is not None:
            parts.append(self.builtin.render_snapshot())
        if self.provider is not None:
            parts.append(self.provider.system_prompt_block())
        if self.filesystem_memory is not None:
            parts.append(self.filesystem_memory.system_prompt_block())
        return "\n\n".join(p for p in parts if p)

    def initial_context(self, query: str) -> str:
        if self.provider is None:
            return ""
        try:
            return self.provider.initial_context(query)
        except Exception:
            return ""

    def get_tool_schemas(self) -> list[dict]:
        schemas: list[dict] = []
        if self.builtin is not None:
            schemas.append(MEMORY_TOOL_SCHEMA)
        if self.session_store is not None:
            schemas.append(SESSION_SEARCH_TOOL_SCHEMA)
        if self.provider is not None:
            schemas.extend(self.provider.get_tool_schemas())
        return schemas

    @property
    def tool_names(self) -> set[str]:
        names: set[str] = set()
        if self.builtin is not None:
            names.add("memory")
        if self.session_store is not None:
            names.add("session_search")
        return names | self._provider_tool_names

    def handle_tool_call(self, name: str, args: dict) -> dict:
        if name == "memory":
            return self._handle_builtin(args)
        if name == "session_search" and self.session_store is not None:
            return self._handle_session_search(args)
        if name in self._provider_tool_names:
            return self.provider.handle_tool_call(name, args)  # type: ignore[union-attr]
        return {"success": False, "error": f"Unknown memory tool: {name!r}"}

    def _handle_builtin(self, args: dict) -> dict:
        if self.builtin is None:
            return {"success": False, "error": "Built-in MEMORY.md is disabled."}
        action = args.get("action", "")
        before = self.builtin.load()
        if action == "add":
            mirror = args.get("content", "")
            result = self.builtin.add(mirror)
        elif action == "replace":
            mirror = args.get("content", "")
            result = self.builtin.replace(args.get("old_text", ""), mirror)
        elif action == "remove":
            mirror = args.get("old_text", "")
            result = self.builtin.remove(mirror)
        else:
            return {"success": False, "error": f"Unknown memory action: {action!r}. Use 'add', 'replace', or 'remove'."}
        changed = result.get("success") and (result.get("entries") or self.builtin.load()) != before
        if changed:
            self._memory_write_pending = True
        if changed and self.provider is not None:
            try:
                self.provider.on_memory_write(action, mirror)
            except Exception as exc:
                logger.warning("%s on_memory_write failed: %s", self.provider.name, exc)
        return result

    def _handle_session_search(self, args: dict) -> dict:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"success": False, "error": "query cannot be empty."}
        try:
            limit = _coerce_limit(args.get("limit"))
            sessions = self.session_store.search(query, limit=limit)  # type: ignore[union-attr]
        except Exception as e:
            return {"success": False, "error": f"session_search failed: {e}"}
        return {"success": True, "query": query, "session_count": len(sessions), "sessions": sessions}

    def sync_turn(self, user: str, assistant: str) -> None:
        if self.provider is not None:
            try:
                self.provider.sync_turn(user, assistant)
            except Exception as exc:
                logger.warning("%s sync_turn failed: %s", self.provider.name, exc)

    def maybe_consolidate(self, model, messages: list[dict], *, n_calls: int) -> dict | None:
        """Run consolidation after ``every_n_steps`` model calls without a memory write.

        Returns ``None`` if not triggered, otherwise the consolidation result.
        """
        every = self.config.consolidation.every_n_steps
        if every <= 0 or n_calls <= 0:
            return None
        if self._memory_write_pending:
            self._last_memory_write_n_calls = n_calls
            self._memory_write_pending = False
            return None
        if self.builtin is None or n_calls - max(self._last_memory_write_n_calls, self._last_consolidation_n_calls) < every:
            return None
        result = consolidate_memory(
            model,
            self.builtin,
            messages,
            max_actions=self.config.consolidation.max_actions,
            summary_max_chars=self.config.consolidation.summary_max_chars,
        )
        self._last_consolidation_n_calls = n_calls
        if result.get("applied", 0) > 0:
            self._last_memory_write_n_calls = n_calls
        return result

    def on_session_end(self, messages: list[dict], *, model=None) -> None:
        if self.provider is not None:
            try:
                self.provider.on_session_end(messages)
            except Exception as exc:
                logger.warning("%s on_session_end failed: %s", self.provider.name, exc)
        if self.session_store is not None and self._session_id:
            try:
                self.session_store.record_session(self._session_id, messages, summary=summarize_session(messages))
            except Exception:  # session indexing must never break agent shutdown
                pass
        if self.filesystem_memory is not None:
            try:
                self.filesystem_memory.on_session_end(messages, model=model)
            except Exception as exc:
                logger.warning("filesystem memory on_session_end failed: %s", exc)
        if model is not None and self.builtin is not None and self.config.consolidation.on_session_end:
            consolidate_memory(
                model,
                self.builtin,
                messages,
                max_actions=self.config.consolidation.max_actions,
                summary_max_chars=self.config.consolidation.summary_max_chars,
            )

    def shutdown(self) -> None:
        if self.session_store is not None:
            try:
                self.session_store.close()
            except Exception:
                pass
        if self.provider is not None:
            self.provider.shutdown()


def _coerce_limit(value) -> int:
    try:
        return min(20, max(1, int(value or 5)))
    except (TypeError, ValueError):
        return 5
