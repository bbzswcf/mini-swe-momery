from __future__ import annotations

import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from minisweagent.memory.providers.mem0 import Mem0Config, Mem0Provider

# ---------------------------------------------------------------------------
# Fake mem0.Memory — only the methods we actually call.
# ---------------------------------------------------------------------------


@dataclass
class FakeMemory:
    add_calls: list[tuple[list[dict], dict]] = field(default_factory=list)
    search_calls: list[dict] = field(default_factory=list)
    search_results: list[dict] = field(
        default_factory=lambda: [{"memory": "alpha", "score": 0.9}, {"memory": "beta", "score": 0.7}]
    )
    add_should_raise: Exception | None = None
    search_should_raise: Exception | None = None

    def add(self, messages, **kwargs):
        if self.add_should_raise is not None:
            raise self.add_should_raise
        self.add_calls.append((messages, kwargs))

    def search(self, **kwargs):
        if self.search_should_raise is not None:
            raise self.search_should_raise
        self.search_calls.append(kwargs)
        return {"results": list(self.search_results)}


@pytest.fixture
def _fake_mem0_module(monkeypatch):
    """Make `is_available()` return True even when the real `mem0` isn't installed."""
    monkeypatch.setitem(sys.modules, "mem0", types.ModuleType("mem0"))


@pytest.fixture
def provider(tmp_path, monkeypatch, _fake_mem0_module):
    cfg = Mem0Config(
        user_id="u1",
        agent_id="a1",
        llm_api_key="sk-test",
        vector_store_path=tmp_path / "mem0",
    )
    p = Mem0Provider(cfg)
    fake = FakeMemory()
    monkeypatch.setattr(p, "_get_client", lambda: fake)
    yield p, fake
    p.shutdown()


def _wait_writer(p: Mem0Provider, *, timeout: float = 5.0) -> None:
    t = p._writer_thread
    if t is not None and t.is_alive():
        t.join(timeout=timeout)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_build_dict_requires_vector_store_path():
    with pytest.raises(RuntimeError, match="vector_store_path"):
        Mem0Config().build_mem0_dict()


def test_config_build_dict_emits_expected_nested_structure(tmp_path):
    cfg = Mem0Config(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_api_key="sk-test",
        embedder_model="text-embedding-3-small",
        vector_store_path=tmp_path / "store",
        vector_store_collection="trial_42",
    )
    d = cfg.build_mem0_dict()
    assert d["llm"] == {"provider": "openai", "config": {"model": "gpt-4o-mini", "api_key": "sk-test"}}
    assert d["embedder"]["config"] == {"model": "text-embedding-3-small", "api_key": "sk-test"}
    assert d["vector_store"]["provider"] == "chroma"
    assert d["vector_store"]["config"] == {"collection_name": "trial_42", "path": str(tmp_path / "store")}


def test_config_build_dict_falls_back_to_llm_api_key_for_embedder(tmp_path):
    """Most setups reuse the OpenAI key for both LLM and embeddings — make that the default."""
    d = Mem0Config(llm_api_key="shared", vector_store_path=tmp_path).build_mem0_dict()
    assert d["embedder"]["config"]["api_key"] == "shared"


def test_config_extra_overrides_default_keys(tmp_path):
    """`extra` is the escape hatch for advanced mem0 keys (history_db_path, ...)."""
    d = Mem0Config(vector_store_path=tmp_path, extra={"history_db_path": "/tmp/h.db"}).build_mem0_dict()
    assert d["history_db_path"] == "/tmp/h.db"


# ---------------------------------------------------------------------------
# Runtime probe
# ---------------------------------------------------------------------------


def test_is_available_returns_false_when_mem0_missing(monkeypatch):
    """In our test env the `mem0` package is not installed."""
    monkeypatch.delitem(sys.modules, "mem0", raising=False)
    assert Mem0Provider().is_available() is False


def test_is_available_returns_true_when_mem0_imports(_fake_mem0_module):
    assert Mem0Provider().is_available() is True


def test_get_client_raises_clear_error_when_package_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "mem0", raising=False)
    p = Mem0Provider(Mem0Config(vector_store_path=Path("/tmp")))
    with pytest.raises(RuntimeError, match="not installed"):
        p._get_client()


def test_handle_tool_call_short_circuits_when_package_missing(monkeypatch, tmp_path):
    """Structural unavailability must bypass the breaker entirely."""
    monkeypatch.delitem(sys.modules, "mem0", raising=False)
    p = Mem0Provider(Mem0Config(vector_store_path=tmp_path))
    res = p.handle_tool_call("mem0_search", {"query": "x"})
    assert res["success"] is False and "not installed" in res["error"]
    # Repeated unavailability must not trip the circuit breaker — reinstall recovery.
    for _ in range(10):
        p.handle_tool_call("mem0_search", {"query": "x"})
    assert p._consecutive_failures == 0
    assert p._is_breaker_open() is False


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


def test_initialize_creates_vector_store_dir_under_home(tmp_path):
    p = Mem0Provider()
    p.initialize("session-1", home=tmp_path)
    assert p.config.vector_store_path == tmp_path / "mem0"
    assert (tmp_path / "mem0").is_dir() and p._session_id == "session-1"


def test_initialize_keeps_explicit_vector_store_path_intact(tmp_path):
    explicit = tmp_path / "elsewhere"
    p = Mem0Provider(Mem0Config(vector_store_path=explicit))
    p.initialize("s", home=tmp_path / "home")
    assert p.config.vector_store_path == explicit and explicit.is_dir()


def test_initialize_accepts_extra_kwargs(tmp_path):
    """Forward-compat: ignore unknown kwargs from manager so future args don't break us."""
    p = Mem0Provider()
    p.initialize("s", home=tmp_path, platform="cli", agent_role="reviewer")
    assert p._session_id == "s"


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def test_tool_schemas_match_openai_wrapper_format_with_expected_names():
    schemas = Mem0Provider().get_tool_schemas()
    assert {s["function"]["name"] for s in schemas} == {"mem0_search", "mem0_note", "mem0_observe"}
    for s in schemas:
        assert s["type"] == "function" and s["function"]["parameters"]["type"] == "object"


def test_search_uses_read_filter_and_returns_ranked_results(provider):
    p, fake = provider
    res = p.handle_tool_call("mem0_search", {"query": "pytest config"})
    assert res == {
        "success": True,
        "results": [{"memory": "alpha", "score": 0.9}, {"memory": "beta", "score": 0.7}],
        "count": 2,
    }
    assert fake.search_calls[0] == {"query": "pytest config", "limit": 5, "user_id": "u1"}


def test_search_clamps_top_k_to_max_50(provider):
    p, fake = provider
    p.handle_tool_call("mem0_search", {"query": "x", "top_k": 9999})
    assert fake.search_calls[0]["limit"] == 50


def test_search_truncates_long_query(provider):
    p, fake = provider
    p.handle_tool_call("mem0_search", {"query": "x" * 5000})
    assert len(fake.search_calls[0]["query"]) == p.config.search_query_max_chars


def test_note_uses_infer_false_and_write_filter(provider):
    p, fake = provider
    res = p.handle_tool_call("mem0_note", {"fact": "this repo runs tests via tox -e py311"})
    assert res["success"] and "verbatim" in res["message"]
    messages, kwargs = fake.add_calls[0]
    assert messages == [{"role": "user", "content": "this repo runs tests via tox -e py311"}]
    assert kwargs == {"infer": False, "user_id": "u1", "agent_id": "a1"}


def test_observe_uses_infer_true(provider):
    p, fake = provider
    p.handle_tool_call("mem0_observe", {"content": "I tried fix X but it broke Y"})
    _, kwargs = fake.add_calls[0]
    assert kwargs["infer"] is True


@pytest.mark.parametrize(
    ("tool", "args", "missing"),
    [
        ("mem0_search", {}, "query"),
        ("mem0_search", {"query": "  "}, "query"),
        ("mem0_note", {}, "fact"),
        ("mem0_observe", {"content": ""}, "content"),
    ],
)
def test_handle_tool_call_validates_required_args(provider, tool, args, missing):
    p, fake = provider
    res = p.handle_tool_call(tool, args)
    assert res["success"] is False and missing in res["error"]
    assert fake.add_calls == [] and fake.search_calls == []


def test_unknown_tool_returns_error_dict(provider):
    p, _ = provider
    res = p.handle_tool_call("mem0_nope", {})
    assert res["success"] is False and "Unknown" in res["error"]


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------


def test_sync_turn_writes_in_background_with_infer_false(provider):
    """sync_turn stores raw transcripts (`infer=False`) — fact extraction is `mem0_observe`'s job."""
    p, fake = provider
    p.initialize("s", home=p.config.vector_store_path.parent)
    p.sync_turn("user said X", "assistant did Y")
    _wait_writer(p)
    assert len(fake.add_calls) == 1
    messages, kwargs = fake.add_calls[0]
    assert messages == [
        {"role": "user", "content": "user said X"},
        {"role": "assistant", "content": "assistant did Y"},
    ]
    assert kwargs == {"infer": False, "user_id": "u1", "agent_id": "a1"}


def test_sync_turn_serializes_writes_via_single_shared_thread(provider):
    """Two sync_turns in a row → second waits for first; both land on the client in order."""
    p, fake = provider
    p.initialize("s", home=p.config.vector_store_path.parent)
    p.sync_turn("u1", "a1")
    p.sync_turn("u2", "a2")
    _wait_writer(p)
    assert len(fake.add_calls) == 2
    assert [m[0]["content"] for m, _ in fake.add_calls] == ["u1", "u2"]


def test_on_memory_write_serializes_through_same_writer_as_sync_turn(provider):
    """sync_turn + on_memory_write must share the writer — no concurrent client.add() races."""
    p, fake = provider
    p.initialize("s", home=p.config.vector_store_path.parent)
    p.sync_turn("u", "a")
    p.on_memory_write("add", "fact A")
    p.on_memory_write("replace", "fact B")
    p.on_memory_write("remove", "fact A")  # no-op
    p.on_memory_write("add", "")  # empty — skipped
    _wait_writer(p)
    assert len(fake.add_calls) == 3  # sync_turn + add + replace
    contents = [m[0]["content"] for m, _ in fake.add_calls]
    assert contents == ["u", "fact A", "fact B"]
    metadata_actions = [k.get("metadata", {}).get("action") for _, k in fake.add_calls[1:]]
    assert metadata_actions == ["add", "replace"]
    assert all(k.get("infer") is False for _, k in fake.add_calls)


def test_mirror_can_be_disabled(tmp_path, monkeypatch, _fake_mem0_module):
    p = Mem0Provider(Mem0Config(vector_store_path=tmp_path, mirror_builtin_writes=False))
    fake = FakeMemory()
    monkeypatch.setattr(p, "_get_client", lambda: fake)
    p.initialize("s", home=tmp_path)
    p.on_memory_write("add", "fact")
    time.sleep(0.05)
    assert fake.add_calls == []
    p.shutdown()


def test_on_session_end_drains_pending_sync(provider):
    p, fake = provider
    p.initialize("s", home=p.config.vector_store_path.parent)
    p.sync_turn("u", "a")
    p.on_session_end(messages=[])
    assert len(fake.add_calls) == 1


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


def test_circuit_breaker_trips_after_threshold_failures(provider):
    p, fake = provider
    fake.search_should_raise = RuntimeError("upstream down")
    for _ in range(5):
        res = p.handle_tool_call("mem0_search", {"query": "q"})
        assert res["success"] is False and "mem0 call failed" in res["error"]
    res = p.handle_tool_call("mem0_search", {"query": "q"})
    assert res["success"] is False and "circuit breaker" in res["error"]
    assert p._is_breaker_open()


def test_circuit_breaker_resets_on_success(provider):
    p, fake = provider
    fake.search_should_raise = RuntimeError("transient")
    for _ in range(3):
        p.handle_tool_call("mem0_search", {"query": "q"})
    assert p._consecutive_failures == 3
    fake.search_should_raise = None
    p.handle_tool_call("mem0_search", {"query": "q"})
    assert p._consecutive_failures == 0


def test_circuit_breaker_reopens_after_cooldown(provider):
    p, fake = provider
    fake.search_should_raise = RuntimeError("down")
    for _ in range(5):
        p.handle_tool_call("mem0_search", {"query": "q"})
    assert p._is_breaker_open()
    p._breaker_open_until = time.monotonic() - 1  # simulate cooldown elapsed
    assert p._is_breaker_open() is False and p._consecutive_failures == 0


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def test_shutdown_drops_post_shutdown_writes(provider):
    p, fake = provider
    p.initialize("s", home=p.config.vector_store_path.parent)
    p.shutdown()
    p.sync_turn("u", "a")
    p.on_memory_write("add", "x")
    time.sleep(0.05)
    assert fake.add_calls == [] and fake.search_calls == []


def test_double_shutdown_is_safe(tmp_path, monkeypatch, _fake_mem0_module):
    p = Mem0Provider(Mem0Config(vector_store_path=tmp_path))
    monkeypatch.setattr(p, "_get_client", lambda: FakeMemory())
    p.initialize("s", home=tmp_path)
    p.shutdown()
    p.shutdown()
