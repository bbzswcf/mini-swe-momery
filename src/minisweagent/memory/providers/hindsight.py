"""Hindsight memory provider — `local_embedded` mode only.

Wraps the `hindsight` package's embedded daemon (local PostgreSQL + LLM-driven
extraction) behind our `MemoryProvider` ABC. Tools exposed to the agent:
``hindsight_retain`` / ``hindsight_recall`` / ``hindsight_reflect``.

Engineering patterns ported from `hermes-agent/plugins/memory/hindsight`
(digest §13):

- Lazy thread-safe client init.
- Task-end synchronous transcript retain; per-step hooks only buffer locally.
- Background asyncio loop dedicated to hindsight's async client API.
- Profile env file materialized at the path the daemon expects
  (``~/.hindsight/profiles/<profile>.env``) — that path is hard-coded inside
  the embedded daemon; we override it here only via env file *content*.

Out of scope (vs hermes, intentional):

- No cloud / `local_external` modes.
- No bank_id templating, no multi-bank, no full Hermes ``auto_retain`` /
  ``retain_every_n_turns`` knobs.
- No per-turn prefetch / on_pre_compress; mini-memory only supports a one-shot
  initial recall overlay plus recall-on-demand via the ``hindsight_recall`` tool.
- No gateway / profile / ``agent_identity`` context.
- No per-turn or per-tool-call Hindsight transcript writes; task transcripts are
  retained synchronously once at session end.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minisweagent.memory.provider import MemoryProvider

logger = logging.getLogger(__name__)

_DEFAULT_REQUEST_TIMEOUT = 120.0
_DEFAULT_IDLE_TIMEOUT = 300
_DEFAULT_LLM_PROVIDER = "openai_compatible"
_DEFAULT_LLM_MODEL = "gpt-5.4-2026-03-05"
_DEFAULT_LLM_BASE_URL = "https://aidp.bytedance.net/api/modelhub/online/v2/crawl/openai/deployments/gpt_openapi"
_VALID_BUDGETS = ("low", "mid", "high")
_VALID_PROVIDERS = (
    "openai",
    "anthropic",
    "gemini",
    "groq",
    "openrouter",
    "minimax",
    "ollama",
    "lmstudio",
    "openai_compatible",
)


def _daemon_profile_root() -> Path:
    """Where the embedded daemon looks for ``<profile>.env``. Hard-coded by hindsight-embed."""
    return Path.home() / ".hindsight" / "profiles"


# ---------------------------------------------------------------------------
# Tool schemas — OpenAI tool-call wrapper format (see `provider.py`).
# ---------------------------------------------------------------------------

_RETAIN_DESC_WITH_BUILTIN = (
    "Store a piece of code-task knowledge to long-term memory. The Hindsight "
    "backend extracts entities, builds a knowledge graph, and indexes it for "
    "later recall/reflect. Prefer the built-in `memory` tool for compact, frozen "
    "facts; use this for richer narrative context (e.g. 'when I tried X on this "
    "repo the test runner crashed with Y; root cause was Z') that benefits from "
    "synthesis across many entries."
)
_RETAIN_DESC_STANDALONE = (
    "Store durable code-task knowledge to long-term memory. The Hindsight backend "
    "extracts entities, builds a knowledge graph, and indexes it for later "
    "recall/reflect. Use for build/test commands, repo conventions, verified "
    "gotchas, and failed approaches to avoid — not the current issue text, raw "
    "logs, or diffs."
)

RETAIN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "hindsight_retain",
        "description": _RETAIN_DESC_WITH_BUILTIN,
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Information to store."},
                "context": {
                    "type": "string",
                    "description": "Short label, e.g. 'pytest config' or 'failed approach'.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags merged with configured retain_tags.",
                },
            },
            "required": ["content"],
        },
    },
}

RECALL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "hindsight_recall",
        "description": (
            "Search long-term memory for relevant code-task knowledge. Returns ranked "
            "memories using semantic search + entity-graph traversal. Use when you "
            "suspect you've seen a similar issue / pattern / fix before."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look up."}},
            "required": ["query"],
        },
    },
}

REFLECT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "hindsight_reflect",
        "description": (
            "Synthesize a narrative answer across all stored memories using the "
            "configured LLM. More expensive than recall but produces reasoning "
            "(e.g. 'have I tried fix X for this kind of bug before? if so, why didn't "
            "it work?')."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Question to reflect on."}},
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HindsightConfig:
    profile: str = "mini-memory"
    bank_id: str = "mini-memory"
    budget: str = "mid"
    llm_provider: str = _DEFAULT_LLM_PROVIDER
    llm_model: str = _DEFAULT_LLM_MODEL
    llm_api_key: str = ""
    llm_base_url: str = _DEFAULT_LLM_BASE_URL
    database_url: str = ""
    request_timeout: float = _DEFAULT_REQUEST_TIMEOUT
    drain_timeout: float = 30.0
    idle_timeout: int = _DEFAULT_IDLE_TIMEOUT
    recall_max_tokens: int = 4096
    recall_max_input_chars: int = 800
    recall_tags: str | list[str] = ""
    recall_tags_match: str = "any"
    retain_context: str = "code-task trial in mini-memory"
    retain_tags: str | list[str] = ""
    retain_async: bool = True
    mirror_builtin_writes: bool = True
    auto_recall_on_init: bool = True
    fail_fast: bool = False

    def __post_init__(self) -> None:
        self.llm_api_key = str(
            self.llm_api_key
            or os.getenv("HINDSIGHT_API_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()
        self.profile = self.profile.strip()
        self.bank_id = self.bank_id.strip()
        self.recall_tags = _normalize_tags(self.recall_tags)
        self.retain_tags = _normalize_tags(self.retain_tags)
        if not self.profile:
            raise ValueError("profile must be a non-empty string")
        if not self.bank_id:
            raise ValueError("bank_id must be a non-empty string")
        if self.budget not in _VALID_BUDGETS:
            raise ValueError(f"budget must be one of {_VALID_BUDGETS}, got {self.budget!r}")
        if self.llm_provider not in _VALID_PROVIDERS:
            raise ValueError(f"llm_provider must be one of {_VALID_PROVIDERS}, got {self.llm_provider!r}")
        if self.recall_tags_match not in ("any", "all", "any_strict", "all_strict"):
            raise ValueError("recall_tags_match must be one of any, all, any_strict, all_strict")


def _check_local_runtime() -> tuple[bool, str | None]:
    """Probe that the local Hindsight stack imports cleanly. No side effects."""
    try:
        importlib.import_module("hindsight")
        importlib.import_module("hindsight_embed.daemon_embed_manager")
    except Exception as exc:
        return False, str(exc)
    return True, None


def _is_retriable_embedded_connection_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "cannot connect to host",
            "connection refused",
            "connect call failed",
            "clientconnectorerror",
            "server disconnected",
        )
    )


def _build_daemon_env(cfg: HindsightConfig) -> dict[str, str]:
    """Env vars the standalone hindsight-embed daemon expects on startup."""
    daemon_provider = "openai" if cfg.llm_provider in ("openai_compatible", "openrouter") else cfg.llm_provider
    env = {
        "HINDSIGHT_API_LLM_PROVIDER": daemon_provider,
        "HINDSIGHT_API_LLM_API_KEY": cfg.llm_api_key,
        "HINDSIGHT_API_LLM_MODEL": cfg.llm_model,
        "HINDSIGHT_API_LOG_LEVEL": "info",
        "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT": str(cfg.idle_timeout),
    }
    if cfg.llm_base_url:
        env["HINDSIGHT_API_LLM_BASE_URL"] = cfg.llm_base_url
    if cfg.database_url:
        env["HINDSIGHT_EMBED_API_DATABASE_URL"] = cfg.database_url
    return env


def _derive_default_scoped_name(name: str, home: Path) -> str:
    if name != "mini-memory":
        return name
    digest = hashlib.sha1(str(home.expanduser().resolve()).encode()).hexdigest()[:12]
    return f"{name}-{digest}"


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    raw_items: list[Any]
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            raw_items = parsed if isinstance(parsed, list) else text.split(",")
        else:
            raw_items = text.split(",")
    else:
        raw_items = [value]
    tags = []
    for item in raw_items:
        tag = str(item).strip()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _merge_tags(*values: Any) -> list[str]:
    tags: list[str] = []
    for value in values:
        for tag in _normalize_tags(value):
            if tag not in tags:
                tags.append(tag)
    return tags


def _materialize_daemon_env(cfg: HindsightConfig, *, profile: str | None = None) -> Path:
    """Write the per-profile env file the embedded daemon reads on startup."""
    path = _daemon_profile_root() / f"{profile or cfg.profile}.env"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{k}={v}\n" for k, v in _build_daemon_env(cfg).items()), encoding="utf-8")
    return path


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# HindsightProvider
# ---------------------------------------------------------------------------


class HindsightProvider(MemoryProvider):
    """Long-term memory backed by a local Hindsight embedded daemon.

    Lifecycle:

    - `initialize` materializes the daemon env file and mints a fresh
      ``document_id`` (per process, per session). The first tool call lazy-starts
      the daemon; we don't warm it up eagerly.
    - `sync_turn` accumulates a JSON snippet per turn in memory only.
    - `on_session_end` synchronously flushes the whole task transcript under the
      current document_id before the runner can move to the next issue.
    - `shutdown` closes the client and stops the asyncio loop.
    """

    def __init__(self, config: HindsightConfig | None = None) -> None:
        self.config = config or HindsightConfig()

        self._client: Any = None
        self._client_lock = threading.Lock()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()

        self._shutting_down = threading.Event()

        self._session_id: str = ""
        self._document_id: str = ""
        self._session_turns: list[str] = []
        self._flushed_turns = 0
        self._profile = self.config.profile
        self._bank_id = self.config.bank_id

    # ------------------------------------------------------------------ ABC

    @property
    def name(self) -> str:
        return "hindsight"

    def is_available(self) -> bool:
        return _check_local_runtime()[0]

    def initialize(self, session_id: str, *, home: Path, **_: Any) -> None:
        self._session_id = (str(session_id or "").strip()) or "default"
        self._document_id = f"{self._session_id}-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        self._session_turns = []
        self._flushed_turns = 0
        self._profile = _derive_default_scoped_name(self.config.profile, home)
        self._bank_id = _derive_default_scoped_name(self.config.bank_id, home)
        self._shutting_down.clear()
        _materialize_daemon_env(self.config, profile=self._profile)
        if self.config.fail_fast:
            client = self._get_client()
            ensure_started = getattr(client, "_ensure_started", None)
            if callable(ensure_started):
                ensure_started()

    def get_tool_schemas(self) -> list[dict]:
        retain = {
            "type": "function",
            "function": {
                **RETAIN_SCHEMA["function"],
                "description": (
                    _RETAIN_DESC_WITH_BUILTIN
                    if self.config.mirror_builtin_writes
                    else _RETAIN_DESC_STANDALONE
                ),
            },
        }
        return [retain, RECALL_SCHEMA, REFLECT_SCHEMA]

    def system_prompt_block(self) -> str:
        cfg = self.config
        return (
            "# Hindsight long-term memory\n"
            f"Active. Bank: {self._bank_id}, budget: {cfg.budget}.\n"
            "Use `hindsight_retain` to record narrative context, `hindsight_recall` "
            "to search, `hindsight_reflect` for cross-memory synthesis."
        )

    # ---------------------------------------------------------------- tools

    def handle_tool_call(self, name: str, args: dict) -> dict:
        try:
            if name == "hindsight_retain":
                return self._tool_retain(args)
            if name == "hindsight_recall":
                return self._tool_recall(args)
            if name == "hindsight_reflect":
                return self._tool_reflect(args)
            return {"success": False, "error": f"Unknown hindsight tool: {name!r}"}
        except Exception as exc:
            logger.warning("hindsight tool call failed: %s", exc)
            return {"success": False, "error": f"hindsight call failed: {exc}"}

    def _tool_retain(self, args: dict) -> dict:
        content = (args.get("content") or "").strip()
        if not content:
            return {"success": False, "error": "Missing required parameter: content."}
        kwargs = self._retain_kwargs(content, context=args.get("context"), tags=args.get("tags"))
        self._run_op(lambda c: c.aretain(**kwargs))
        return {"success": True, "message": "Memory stored."}

    def _tool_recall(self, args: dict) -> dict:
        query = (args.get("query") or "").strip()
        if not query:
            return {"success": False, "error": "Missing required parameter: query."}
        resp = self._run_op(lambda c: c.arecall(**self._recall_kwargs(query)))
        results = list(getattr(resp, "results", None) or [])
        texts = [t for t in (getattr(r, "text", "") for r in results) if t]
        return {"success": True, "results": texts, "count": len(texts)}

    def _tool_reflect(self, args: dict) -> dict:
        query = (args.get("query") or "").strip()
        if not query:
            return {"success": False, "error": "Missing required parameter: query."}
        resp = self._run_op(
            lambda c: c.areflect(bank_id=self._bank_id, query=query, budget=self.config.budget)
        )
        return {"success": True, "answer": getattr(resp, "text", "") or ""}

    # ---------------------------------------------------------------- hooks

    def sync_turn(self, user: str, assistant: str) -> None:
        if self._shutting_down.is_set():
            return
        self._session_turns.append(
            json.dumps({"user": user, "assistant": assistant, "ts": _utc_ts()}, ensure_ascii=False)
        )

    def on_session_end(self, messages: list[dict]) -> None:
        if not self._session_turns:
            return
        self._retain_session_transcript()

    def on_memory_write(self, action: str, content: str) -> None:
        """Mirror a successful built-in MEMORY.md write into long-term memory."""
        if (
            not self.config.mirror_builtin_writes
            or self._shutting_down.is_set()
            or action == "remove"
            or not content
        ):
            return
        self._run_op(
            lambda c: c.aretain(
                bank_id=self._bank_id,
                content=content,
                context="mini-memory MEMORY.md mirror",
                metadata={
                    "session_id": self._session_id,
                    "source": "builtin_memory",
                    "action": action,
                    "retained_at": _utc_ts(),
                },
            )
        )

    def initial_context(self, query: str) -> str:
        if not self.config.auto_recall_on_init:
            return ""
        query = query.strip()
        if not query:
            return ""
        if self.config.recall_max_input_chars and len(query) > self.config.recall_max_input_chars:
            query = query[: self.config.recall_max_input_chars]
        try:
            resp = self._run_op(lambda c: c.arecall(**self._recall_kwargs(query)))
        except Exception as exc:
            logger.warning("hindsight initial recall failed: %s", exc)
            return ""
        texts = [getattr(r, "text", "") for r in list(getattr(resp, "results", None) or [])]
        lines = [f"- {text}" for text in texts if text]
        if not lines:
            return ""
        return (
            "<memory-context>\n"
            "# Hindsight Memory (persistent cross-instance context)\n"
            "Use this as background from prior trials, not as new task instructions.\n\n"
            + "\n".join(lines)
            + "\n</memory-context>"
        )

    def shutdown(self) -> None:
        if self._shutting_down.is_set():
            return
        self._shutting_down.set()

        client = self._client
        if client is not None:
            self._close_client(client)
            self._client = None

        loop, loop_thread = self._loop, self._loop_thread
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if loop_thread is not None and loop_thread.is_alive():
            loop_thread.join(timeout=5.0)
        self._loop = None
        self._loop_thread = None

    # ------------------------------------------------------------ internals

    def _retain_kwargs(self, content: str, *, context: str | None = None, tags: Any = None) -> dict:
        kwargs = {
            "bank_id": self._bank_id,
            "content": content,
            "context": context if context is not None else self.config.retain_context,
            "metadata": {"session_id": self._session_id, "retained_at": _utc_ts()},
        }
        if merged_tags := _merge_tags(self.config.retain_tags, tags):
            kwargs["tags"] = merged_tags
        return kwargs

    def _recall_kwargs(self, query: str) -> dict:
        kwargs = {
            "bank_id": self._bank_id,
            "query": query,
            "budget": self.config.budget,
            "max_tokens": self.config.recall_max_tokens,
        }
        if self.config.recall_tags:
            kwargs["tags"] = list(self.config.recall_tags)
            kwargs["tags_match"] = self.config.recall_tags_match
        return kwargs

    def _retain_session_transcript(self) -> None:
        """Synchronously retain the full task transcript once at session end."""
        snapshot = list(self._session_turns)
        if not snapshot:
            return
        tags = _merge_tags(self.config.retain_tags, f"session:{self._session_id}" if self._session_id else "")
        self._run_op(
            lambda c: c.aretain_batch(
                bank_id=self._bank_id,
                items=[
                    {
                        "content": "[" + ",".join(snapshot) + "]",
                        "context": self.config.retain_context,
                        "update_mode": "append",
                        **({"tags": tags} if tags else {}),
                        "metadata": {
                            "session_id": self._session_id,
                            "turn_count": str(len(snapshot)),
                            "phase": "session_end",
                            "retained_at": _utc_ts(),
                        },
                    }
                ],
                document_id=self._document_id,
                retain_async=False,
            )
        )
        self._flushed_turns = len(self._session_turns)
        self._session_turns = []

    def _get_client(self) -> Any:
        """Lazy thread-safe client init. Raises if the local runtime is unavailable."""
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            available, reason = _check_local_runtime()
            if not available:
                raise RuntimeError(
                    "Hindsight local runtime unavailable" + (f": {reason}" if reason else "")
                )
            from hindsight import HindsightEmbedded  # type: ignore[import-not-found]

            HindsightEmbedded.__del__ = lambda self: None
            cfg = self.config
            provider = "openai" if cfg.llm_provider in ("openai_compatible", "openrouter") else cfg.llm_provider
            kwargs: dict[str, Any] = {
                "profile": self._profile,
                "llm_provider": provider,
                "llm_api_key": cfg.llm_api_key,
                "llm_model": cfg.llm_model,
                "idle_timeout": cfg.idle_timeout,
            }
            if cfg.llm_base_url:
                kwargs["llm_base_url"] = cfg.llm_base_url
            if cfg.database_url:
                kwargs["database_url"] = cfg.database_url
            self._client = HindsightEmbedded(**kwargs)
            return self._client

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._loop is not None and self._loop.is_running():
                return self._loop
            loop = asyncio.new_event_loop()

            def _run() -> None:
                asyncio.set_event_loop(loop)
                loop.run_forever()

            t = threading.Thread(target=_run, daemon=True, name="hindsight-loop")
            t.start()
            self._loop = loop
            self._loop_thread = t
            return loop

    def _run_sync(self, coro: Any) -> Any:
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=self.config.request_timeout)

    def _run_op(self, op: Callable[[Any], Any]) -> Any:
        client = self._get_client()
        try:
            return self._run_sync(op(client))
        except Exception as exc:
            if not _is_retriable_embedded_connection_error(exc):
                raise
            logger.info("hindsight embedded daemon unreachable; recreating client and retrying once: %s", exc)
            self._client = None
            return self._run_sync(op(self._get_client()))

    def _close_client(self, client: Any) -> None:
        # Catch shutdown-path noise so it never masks the agent's exit code.
        try:
            aclose = getattr(client, "aclose", None)
        except Exception:
            logger.debug("hindsight client.aclose lookup failed", exc_info=True)
            aclose = None
        if aclose is not None:
            try:
                self._run_sync(aclose())
                return
            except Exception:
                logger.debug("hindsight client.aclose failed", exc_info=True)
        try:
            close = getattr(client, "close", None)
        except Exception:
            logger.debug("hindsight client.close lookup failed", exc_info=True)
            return
        if callable(close):
            try:
                close()
            except Exception:
                logger.debug("hindsight client.close failed", exc_info=True)
