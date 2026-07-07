"""Mem0 memory provider — OSS local mode (`from mem0 import Memory`).

Server-side LLM fact extraction with semantic search and dedup, running
**locally** against a chroma-backed vector store. *Does not* use the cloud
`MemoryClient` (the `hermes-agent` mem0 plugin is cloud-only — we adopt its
threading / circuit-breaker / filter model but swap the backend for OSS).

Design (digest §13 + §10.14):

- Lazy thread-safe client (`Memory.from_config(...)`).
- Circuit breaker: 5 consecutive *runtime* failures → 120 s cooldown to avoid
  hammering a stuck embedder / LLM endpoint. Structural failures (mem0 package
  not installed) bypass the breaker entirely so reinstalling the package never
  leaves the breaker stuck-open.
- Single-writer thread shared by `sync_turn` and `on_memory_write`: a new write
  waits for the previous one to finish before starting (Mem0's server-side
  extraction is the bottleneck and concurrent writes against a local chroma
  store can clog).
- Read / write filter separation: search uses ``user_id`` only (cross-trial
  recall), `add` carries ``user_id + agent_id`` (attribution).
- Vector store path derived from the ``home`` passed to `initialize`, so
  per-instance / per-trial isolation works without changing the manager.

Tools exposed (3 — semantics match digest §10.14):

- ``mem0_search`` — semantic search.
- ``mem0_note`` — store verbatim (``infer=False``).
- ``mem0_observe`` — feed a paragraph; let Mem0's LLM extract facts (``infer=True``).

Note: ``sync_turn`` writes the (user, assistant) pair with ``infer=False`` to
keep raw transcripts available without polluting the fact store. Fact
extraction is **only** triggered when the model explicitly calls ``mem0_observe``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minisweagent.memory.provider import MemoryProvider

logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Tool schemas — OpenAI tool-call wrapper format.
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mem0_search",
        "description": (
            "Search past Mem0 memories by meaning. Returns ranked facts (semantic match). "
            "Use this when you suspect prior trials on similar repos / issues already "
            "discovered something relevant — repo conventions, build/test gotchas, "
            "fixes that worked or didn't."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
                "top_k": {
                    "type": "integer",
                    "description": "Max results to return (default 5, max 50).",
                },
            },
            "required": ["query"],
        },
    },
}

NOTE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mem0_note",
        "description": (
            "Store a fact **verbatim** (no LLM extraction). Use for explicit, durable "
            "code-task conclusions: 'pytest must run with PYTHONPATH=. on this repo', "
            "'CI pins black==24.3.0', etc. Cheaper than `mem0_observe` and ensures the "
            "exact wording is preserved."
        ),
        "parameters": {
            "type": "object",
            "properties": {"fact": {"type": "string", "description": "The fact to store, as a single sentence."}},
            "required": ["fact"],
        },
    },
}

OBSERVE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mem0_observe",
        "description": (
            "Feed a free-form observation (a paragraph of trial notes, a tool output "
            "summary, etc.) and let Mem0's server-side LLM extract atomic facts from "
            "it. Use when you have rich text to file but don't want to manually "
            "distill it; prefer `mem0_note` for things you can already state crisply."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Observation text — a sentence or short paragraph.",
                },
            },
            "required": ["content"],
        },
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Mem0Config:
    user_id: str = "mini-memory"
    agent_id: str = "mini-memory-agent"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = ""
    embedder_provider: str = "openai"
    embedder_model: str = "text-embedding-3-small"
    embedder_api_key: str = ""  # falls back to llm_api_key if empty
    vector_store_provider: str = "chroma"
    vector_store_path: Path | None = None  # auto-filled from `home` in `initialize`
    vector_store_collection: str = "mini_memory"
    search_top_k: int = 5
    search_query_max_chars: int = 800
    mirror_builtin_writes: bool = True
    extra: dict[str, Any] = field(default_factory=dict)  # escape hatch for advanced mem0 keys

    def build_mem0_dict(self) -> dict[str, Any]:
        """Render the nested config dict that ``Memory.from_config(...)`` expects."""
        if self.vector_store_path is None:
            raise RuntimeError(
                "Mem0Config.vector_store_path is unset — call MemoryManager.initialize() first "
                "(or set it explicitly)."
            )
        llm_cfg: dict[str, Any] = {"model": self.llm_model}
        if self.llm_api_key:
            llm_cfg["api_key"] = self.llm_api_key
        if self.llm_base_url:
            llm_cfg["openai_base_url" if self.llm_provider == "openai" else "base_url"] = self.llm_base_url

        emb_cfg: dict[str, Any] = {"model": self.embedder_model}
        emb_key = self.embedder_api_key or self.llm_api_key
        if emb_key:
            emb_cfg["api_key"] = emb_key

        cfg: dict[str, Any] = {
            "llm": {"provider": self.llm_provider, "config": llm_cfg},
            "embedder": {"provider": self.embedder_provider, "config": emb_cfg},
            "vector_store": {
                "provider": self.vector_store_provider,
                "config": {
                    "collection_name": self.vector_store_collection,
                    "path": str(self.vector_store_path),
                },
            },
        }
        cfg.update(self.extra)
        return cfg


# ---------------------------------------------------------------------------
# Mem0Provider
# ---------------------------------------------------------------------------


def _unwrap_results(response: Any) -> list[dict[str, Any]]:
    """Normalize a Mem0 search/get_all response into a flat list of dicts."""
    if isinstance(response, dict):
        return list(response.get("results", []))
    if isinstance(response, list):
        return list(response)
    return []


_PACKAGE_UNAVAILABLE_ERROR = {
    "success": False,
    "error": "mem0 package not installed. Run: pip install 'mini-swe-agent[mem0]'",
}


class Mem0Provider(MemoryProvider):
    """Mem0 OSS-local provider with circuit breaker + lazy client."""

    def __init__(self, config: Mem0Config | None = None) -> None:
        self.config = config or Mem0Config()
        self._client: Any = None
        self._client_lock = threading.Lock()
        # Single shared writer thread — both `sync_turn` and `on_memory_write` go
        # through `_spawn_serialized()` so writes against the local chroma store
        # never run concurrently.
        self._writer_thread: threading.Thread | None = None
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        self._shutting_down = threading.Event()
        self._session_id = ""

    # ------------------------------------------------------------------ ABC

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        # Cheap cached import check — must not raise (digest §6.1).
        try:
            import mem0  # noqa: F401
        except Exception:
            return False
        return True

    def initialize(self, session_id: str, *, home: Path, **_: Any) -> None:
        self._session_id = (str(session_id or "").strip()) or "default"
        self._shutting_down.clear()
        if self.config.vector_store_path is None:
            self.config.vector_store_path = home / "mem0"
        self.config.vector_store_path.mkdir(parents=True, exist_ok=True)

    def get_tool_schemas(self) -> list[dict]:
        return [SEARCH_SCHEMA, NOTE_SCHEMA, OBSERVE_SCHEMA]

    def system_prompt_block(self) -> str:
        return (
            "# Mem0 long-term memory\n"
            f"Active. user_id={self.config.user_id!r}, agent_id={self.config.agent_id!r}.\n"
            "Use `mem0_search` to look up past trial knowledge, `mem0_note` to file a "
            "verbatim fact, `mem0_observe` to record a paragraph for server-side fact "
            "extraction."
        )

    # ---------------------------------------------------------------- tools

    def handle_tool_call(self, name: str, args: dict) -> dict:
        # Structural unavailability bypasses the breaker so reinstalling the
        # package later doesn't leave the breaker stuck open.
        if not self.is_available():
            return dict(_PACKAGE_UNAVAILABLE_ERROR)
        if self._is_breaker_open():
            return {
                "success": False,
                "error": "Mem0 circuit breaker open (recent failures). Skipping this call; will reset after cooldown.",
            }
        if name == "mem0_search":
            return self._guard(lambda: self._tool_search(args))
        if name == "mem0_note":
            return self._guard(lambda: self._tool_note(args))
        if name == "mem0_observe":
            return self._guard(lambda: self._tool_observe(args))
        return {"success": False, "error": f"Unknown mem0 tool: {name!r}"}

    def _tool_search(self, args: dict) -> dict:
        query = (args.get("query") or "").strip()
        if not query:
            return {"success": False, "error": "Missing required parameter: query."}
        top_k = max(1, min(int(args.get("top_k") or self.config.search_top_k), 50))
        if self.config.search_query_max_chars and len(query) > self.config.search_query_max_chars:
            query = query[: self.config.search_query_max_chars]
        results = _unwrap_results(self._get_client().search(query=query, limit=top_k, **self._read_filters()))
        items = [
            {"memory": r.get("memory", ""), "score": r.get("score", 0)}
            for r in results
            if r.get("memory")
        ]
        return {"success": True, "results": items, "count": len(items)}

    def _tool_note(self, args: dict) -> dict:
        fact = (args.get("fact") or "").strip()
        if not fact:
            return {"success": False, "error": "Missing required parameter: fact."}
        self._get_client().add([{"role": "user", "content": fact}], infer=False, **self._write_filters())
        return {"success": True, "message": "Fact stored verbatim."}

    def _tool_observe(self, args: dict) -> dict:
        content = (args.get("content") or "").strip()
        if not content:
            return {"success": False, "error": "Missing required parameter: content."}
        self._get_client().add([{"role": "user", "content": content}], infer=True, **self._write_filters())
        return {"success": True, "message": "Observation queued for fact extraction."}

    # ---------------------------------------------------------------- hooks

    def sync_turn(self, user: str, assistant: str) -> None:
        """Persist the (user, assistant) pair verbatim (``infer=False``).

        Fact extraction is owned by the explicit ``mem0_observe`` tool — sync_turn
        only stores raw transcripts so they're searchable without inflating the
        extracted-fact corpus on every turn.
        """
        if self._shutting_down.is_set() or self._is_breaker_open() or not self.is_available():
            return
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        write_filters = self._write_filters()

        def _run() -> None:
            try:
                self._get_client().add(messages, infer=False, **write_filters)
            except Exception as exc:
                self._record_failure()
                logger.warning("mem0 sync_turn failed: %s", exc)
                return
            self._record_success()

        self._spawn_serialized(_run, name="mem0-sync")

    def on_session_end(self, messages: list[dict]) -> None:
        # Drain the single writer so the buffer is flushed when the session ends.
        thread = self._writer_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)

    def on_memory_write(self, action: str, content: str) -> None:
        """Mirror a successful built-in MEMORY.md write into Mem0 as a verbatim note."""
        if (
            not self.config.mirror_builtin_writes
            or self._shutting_down.is_set()
            or self._is_breaker_open()
            or not self.is_available()
            or action == "remove"
            or not content
        ):
            return
        write_filters = self._write_filters()
        session_id = self._session_id

        def _run() -> None:
            try:
                self._get_client().add(
                    [{"role": "user", "content": content}],
                    infer=False,
                    metadata={
                        "source": "builtin_memory",
                        "action": action,
                        "session_id": session_id,
                    },
                    **write_filters,
                )
            except Exception as exc:
                self._record_failure()
                logger.warning("mem0 mirror write failed: %s", exc)
                return
            self._record_success()

        self._spawn_serialized(_run, name="mem0-mirror")

    def shutdown(self) -> None:
        if self._shutting_down.is_set():
            return
        self._shutting_down.set()
        thread = self._writer_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)
        with self._client_lock:
            self._client = None

    # ------------------------------------------------------------ internals

    def _spawn_serialized(self, target: Callable[[], None], *, name: str) -> None:
        """Wait for the previous writer to finish, then start a new one. Single-writer guarantee."""
        prev = self._writer_thread
        if prev is not None and prev.is_alive():
            prev.join(timeout=10.0)
        thread = threading.Thread(target=target, daemon=True, name=name)
        self._writer_thread = thread
        thread.start()

    def _read_filters(self) -> dict[str, Any]:
        """Search uses ``user_id`` only — recall across all agents/sessions for this user."""
        return {"user_id": self.config.user_id}

    def _write_filters(self) -> dict[str, Any]:
        """Add carries both IDs so the agent's writes stay attributable."""
        return {"user_id": self.config.user_id, "agent_id": self.config.agent_id}

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            # Convert ImportError → RuntimeError so callers get a single, install-aware
            # diagnostic instead of a stray ImportError far from the install hint.
            try:
                from mem0 import Memory  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "mem0 package not installed. Run: pip install 'mini-swe-agent[mem0]'"
                ) from exc
            self._client = Memory.from_config(self.config.build_mem0_dict())
            return self._client

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self) -> None:
        self._consecutive_failures = 0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures == _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "mem0 circuit breaker tripped after %d consecutive failures; pausing %ds",
                self._consecutive_failures,
                _BREAKER_COOLDOWN_SECS,
            )

    def _guard(self, fn: Callable[[], dict]) -> dict:
        """Run a synchronous tool call, mark success/failure for the breaker, surface errors as dicts."""
        try:
            result = fn()
        except Exception as exc:
            self._record_failure()
            logger.warning("mem0 tool call failed: %s", exc)
            return {"success": False, "error": f"mem0 call failed: {exc}"}
        if result.get("success"):
            self._record_success()
        return result
