from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from minisweagent.memory.providers import hindsight as hs
from minisweagent.memory.providers.hindsight import (
    HindsightConfig,
    HindsightProvider,
    _build_daemon_env,
)

# ---------------------------------------------------------------------------
# Fake hindsight client — imitates only the methods we actually call.
# ---------------------------------------------------------------------------


@dataclass
class _FakeRecallResult:
    text: str


@dataclass
class _FakeRecallResp:
    results: list[_FakeRecallResult]


@dataclass
class _FakeReflectResp:
    text: str


@dataclass
class FakeClient:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    closed: bool = False
    recall_results: tuple[str, ...] = ("alpha", "beta")
    reflect_text: str = "synthesized answer"

    async def aretain(self, **kwargs):
        self.calls.append(("aretain", kwargs))

    async def aretain_batch(self, **kwargs):
        self.calls.append(("aretain_batch", kwargs))

    async def arecall(self, **kwargs):
        self.calls.append(("arecall", kwargs))
        return _FakeRecallResp(results=[_FakeRecallResult(text=t) for t in self.recall_results])

    async def areflect(self, **kwargs):
        self.calls.append(("areflect", kwargs))
        return _FakeReflectResp(text=self.reflect_text)

    async def aclose(self):
        self.calls.append(("aclose", {}))
        self.closed = True


class FailOnceClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.failures_left = 1

    async def aretain_batch(self, **kwargs):
        self.calls.append(("aretain_batch_attempt", kwargs))
        if self.failures_left:
            self.failures_left -= 1
            raise RuntimeError("temporary retain failure")
        self.calls.append(("aretain_batch", kwargs))


@pytest.fixture
def home_redirect(tmp_path, monkeypatch):
    """Redirect Path.home() so daemon profile env lands in tmp_path."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def provider(home_redirect, monkeypatch):
    cfg = HindsightConfig(
        profile="test-profile",
        bank_id="test-bank",
        llm_api_key="sk-test",
        llm_model="gpt-test",
        idle_timeout=60,
    )
    p = HindsightProvider(cfg)
    fake = FakeClient()
    monkeypatch.setattr(p, "_get_client", lambda: fake)
    yield p, fake
    p.shutdown()


# ---------------------------------------------------------------------------
# Config & runtime probe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"budget": "max"}, "budget"),
        ({"llm_provider": "deepseek"}, "llm_provider"),
    ],
)
def test_config_validates_known_choices(kwargs, match):
    with pytest.raises(ValueError, match=match):
        HindsightConfig(**kwargs)


def test_config_defaults_to_main_experiment_modelhub_chat_endpoint():
    cfg = HindsightConfig()
    assert cfg.llm_provider == "openai_compatible"
    assert cfg.llm_model == "gpt-5.4-2026-03-05"
    assert cfg.llm_base_url == "https://aidp.bytedance.net/api/modelhub/online/v2/crawl/openai/deployments/gpt_openapi"


def test_config_reads_hindsight_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "hindsight-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert HindsightConfig().llm_api_key == "hindsight-key"


def test_config_falls_back_to_openai_api_key(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_API_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert HindsightConfig().llm_api_key == "openai-key"


def test_is_available_returns_false_when_runtime_missing(monkeypatch):
    """`hindsight` is not installed in our test env — probe must report False without raising."""
    monkeypatch.setattr(hs, "_check_local_runtime", lambda: (False, "missing"))
    assert HindsightProvider().is_available() is False


def test_is_available_returns_true_when_runtime_imports(monkeypatch):
    monkeypatch.setattr(hs.importlib, "import_module", lambda name: object())
    assert HindsightProvider().is_available() is True


def test_get_client_raises_when_runtime_unavailable(monkeypatch):
    monkeypatch.setattr(hs, "_check_local_runtime", lambda: (False, "missing"))
    p = HindsightProvider()
    with pytest.raises(RuntimeError, match="local runtime unavailable"):
        p._get_client()


def test_get_client_passes_openai_compatible_as_openai_to_daemon(monkeypatch):
    """The HindsightEmbedded client wants `openai` even when we configure `openai_compatible`."""
    import sys
    import types

    captured: dict[str, Any] = {}

    class FakeEmbedded:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __del__(self):  # pragma: no cover - replaced by provider
            pass

    fake_module = types.ModuleType("hindsight")
    fake_module.HindsightEmbedded = FakeEmbedded
    monkeypatch.setitem(sys.modules, "hindsight", fake_module)
    monkeypatch.setattr(hs, "_check_local_runtime", lambda: (True, None))

    cfg = HindsightConfig(
        llm_provider="openai_compatible",
        llm_base_url="http://x:1/v1",
        llm_api_key="k",
        database_url="postgresql://u:p@db:5432/hindsight",
    )
    HindsightProvider(cfg)._get_client()

    assert captured["llm_provider"] == "openai"
    assert captured["llm_base_url"] == "http://x:1/v1"
    assert captured["database_url"] == "postgresql://u:p@db:5432/hindsight"


# ---------------------------------------------------------------------------
# initialize + daemon env materialization
# ---------------------------------------------------------------------------


def test_initialize_writes_daemon_env_file_with_expected_vars(home_redirect):
    cfg = HindsightConfig(
        profile="trial-7",
        llm_provider="openrouter",
        llm_model="qwen/qwen3.5-9b",
        llm_api_key="sk-or-1",
        llm_base_url="https://openrouter.ai/api/v1",
        database_url="postgresql://u:p@db:5432/hindsight",
        idle_timeout=120,
    )
    p = HindsightProvider(cfg)
    p.initialize("session-A", home=home_redirect)

    env_path = home_redirect / ".hindsight" / "profiles" / "trial-7.env"
    assert env_path.exists()
    body = dict(line.split("=", 1) for line in env_path.read_text().splitlines() if "=" in line)
    assert body == {
        "HINDSIGHT_API_LLM_PROVIDER": "openai",  # openrouter → openai wire format
        "HINDSIGHT_API_LLM_API_KEY": "sk-or-1",
        "HINDSIGHT_API_LLM_MODEL": "qwen/qwen3.5-9b",
        "HINDSIGHT_API_LLM_BASE_URL": "https://openrouter.ai/api/v1",
        "HINDSIGHT_EMBED_API_DATABASE_URL": "postgresql://u:p@db:5432/hindsight",
        "HINDSIGHT_API_LOG_LEVEL": "info",
        "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT": "120",
    }


def test_initialize_resets_session_state(home_redirect):
    p = HindsightProvider(HindsightConfig(llm_api_key="k"))
    p._session_turns = ["leftover"]
    p.initialize("s1", home=home_redirect)
    assert p._session_turns == [] and p._session_id == "s1" and p._document_id.startswith("s1-")
    p.initialize("s2", home=home_redirect)
    assert p._session_id == "s2" and not p._document_id.startswith("s1-")


def test_initialize_derives_default_profile_and_bank_from_home(home_redirect):
    p1 = HindsightProvider(HindsightConfig(llm_api_key="k"))
    p2 = HindsightProvider(HindsightConfig(llm_api_key="k"))
    p1.initialize("s", home=home_redirect / "chain-a")
    p2.initialize("s", home=home_redirect / "chain-b")
    assert p1._profile != p2._profile
    assert p1._bank_id != p2._bank_id
    assert p1._profile.startswith("mini-memory-")
    assert p1._bank_id.startswith("mini-memory-")


def test_initialize_accepts_extra_kwargs(home_redirect):
    """Provider.initialize must swallow unknown kwargs from manager forward-compat."""
    p = HindsightProvider(HindsightConfig(llm_api_key="k"))
    p.initialize("s", home=home_redirect, platform="cli", agent_role="reviewer")
    assert p._session_id == "s"


def test_initialize_fail_fast_starts_embedded_client(home_redirect, monkeypatch):
    class WarmupClient:
        def __init__(self):
            self.started = False

        def _ensure_started(self):
            self.started = True

    p = HindsightProvider(HindsightConfig(llm_api_key="k", fail_fast=True))
    client = WarmupClient()
    monkeypatch.setattr(p, "_get_client", lambda: client)
    p.initialize("s", home=home_redirect)
    assert client.started is True


def test_build_daemon_env_includes_idle_timeout_zero():
    """idle_timeout=0 means 'never auto-shut-down', daemon expects the literal '0'."""
    env = _build_daemon_env(HindsightConfig(idle_timeout=0))
    assert env["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] == "0"


# ---------------------------------------------------------------------------
# Tool schemas / dispatch
# ---------------------------------------------------------------------------


def test_tool_schemas_match_openai_wrapper_format_with_expected_names():
    schemas = HindsightProvider().get_tool_schemas()
    assert {s["function"]["name"] for s in schemas} == {
        "hindsight_retain",
        "hindsight_recall",
        "hindsight_reflect",
    }
    for s in schemas:
        assert s["type"] == "function" and s["function"]["parameters"]["type"] == "object"


def test_handle_tool_call_routes_three_tools_and_rejects_unknown(provider):
    p, fake = provider
    p.initialize("s1", home=Path.home())

    assert p.handle_tool_call("hindsight_retain", {"content": "fact about repo X"}) == {
        "success": True,
        "message": "Memory stored.",
    }
    recall = p.handle_tool_call("hindsight_recall", {"query": "anything"})
    assert recall == {"success": True, "results": ["alpha", "beta"], "count": 2}
    reflect = p.handle_tool_call("hindsight_reflect", {"query": "why?"})
    assert reflect == {"success": True, "answer": "synthesized answer"}
    bad = p.handle_tool_call("hindsight_unknown", {})
    assert bad["success"] is False and "Unknown" in bad["error"]

    names = [c[0] for c in fake.calls]
    assert names == ["aretain", "arecall", "areflect"]


@pytest.mark.parametrize(
    ("tool", "args", "missing"),
    [
        ("hindsight_retain", {}, "content"),
        ("hindsight_retain", {"content": "   "}, "content"),
        ("hindsight_recall", {}, "query"),
        ("hindsight_reflect", {"query": ""}, "query"),
    ],
)
def test_handle_tool_call_validates_required_args(provider, tool, args, missing):
    p, fake = provider
    p.initialize("s1", home=Path.home())
    res = p.handle_tool_call(tool, args)
    assert res["success"] is False and missing in res["error"]
    assert fake.calls == []  # nothing reached the client


def test_handle_tool_call_returns_error_dict_when_runtime_fails(home_redirect, monkeypatch):
    p = HindsightProvider(HindsightConfig(llm_api_key="k"))
    monkeypatch.setattr(p, "_get_client", lambda: (_ for _ in ()).throw(RuntimeError("daemon down")))
    p.initialize("s", home=home_redirect)
    res = p.handle_tool_call("hindsight_recall", {"query": "x"})
    assert res["success"] is False and "daemon down" in res["error"]


def test_retries_once_after_stale_embedded_connection(home_redirect, monkeypatch):
    class FailingRecallClient(FakeClient):
        async def arecall(self, **kwargs):
            self.calls.append(("arecall", kwargs))
            raise RuntimeError("Connection refused")

    p = HindsightProvider(HindsightConfig(llm_api_key="k"))
    first, second = FailingRecallClient(), FakeClient()
    clients = [first, second]
    monkeypatch.setattr(p, "_get_client", lambda: clients.pop(0))
    p.initialize("s", home=home_redirect)
    res = p.handle_tool_call("hindsight_recall", {"query": "x"})
    assert res == {"success": True, "results": ["alpha", "beta"], "count": 2}


def test_recall_drops_empty_text_results(provider):
    p, fake = provider
    fake.recall_results = ("hit", "", "another")
    p.initialize("s1", home=Path.home())
    res = p.handle_tool_call("hindsight_recall", {"query": "x"})
    assert res["results"] == ["hit", "another"] and res["count"] == 2


def test_retain_merges_config_and_tool_tags(home_redirect, monkeypatch):
    p = HindsightProvider(HindsightConfig(llm_api_key="k", retain_tags=["repo:django", "chain:1"]))
    fake = FakeClient()
    monkeypatch.setattr(p, "_get_client", lambda: fake)
    p.initialize("s", home=home_redirect)
    p.handle_tool_call("hindsight_retain", {"content": "fact", "tags": ["chain:1", "task:abc"]})
    assert fake.calls[0][1]["tags"] == ["repo:django", "chain:1", "task:abc"]


def test_recall_uses_configured_tags_and_match_mode(home_redirect, monkeypatch):
    p = HindsightProvider(
        HindsightConfig(llm_api_key="k", recall_tags="repo:django,chain:1", recall_tags_match="all")
    )
    fake = FakeClient()
    monkeypatch.setattr(p, "_get_client", lambda: fake)
    p.initialize("s", home=home_redirect)
    p.handle_tool_call("hindsight_recall", {"query": "x"})
    kwargs = fake.calls[0][1]
    assert kwargs["tags"] == ["repo:django", "chain:1"]
    assert kwargs["tags_match"] == "all"


def test_initial_context_recalls_relevant_memories(home_redirect, monkeypatch):
    p = HindsightProvider(HindsightConfig(llm_api_key="k", auto_recall_on_init=True, recall_tags=["repo:django"]))
    fake = FakeClient()
    fake.recall_results = ("use pytest -q", "set PYTHONPATH=.")
    monkeypatch.setattr(p, "_get_client", lambda: fake)
    p.initialize("s", home=home_redirect)
    block = p.initial_context("django failing tests")
    assert "# Hindsight Memory" in block
    assert "- use pytest -q" in block
    assert "- set PYTHONPATH=." in block
    kwargs = fake.calls[0][1]
    assert kwargs["query"] == "django failing tests"
    assert kwargs["tags"] == ["repo:django"]


# ---------------------------------------------------------------------------
# Lifecycle hooks: sync_turn / on_session_end / on_memory_write
# ---------------------------------------------------------------------------


def test_sync_turn_buffers_without_writing_to_hindsight(provider):
    p, fake = provider
    p.initialize("session-99", home=Path.home())
    p.sync_turn("user msg", "assistant msg")
    assert fake.calls == []


def test_session_end_writes_all_buffered_turns_once_synchronously(provider):
    p, fake = provider
    p.initialize("session-99", home=Path.home())
    p.sync_turn("u1", "a1")
    p.sync_turn("u2", "a2")
    p.on_session_end(messages=[])

    assert len(fake.calls) == 1 and fake.calls[0][0] == "aretain_batch"
    kwargs = fake.calls[0][1]
    assert kwargs["bank_id"] == "test-bank" and kwargs["document_id"].startswith("session-99-")
    assert kwargs["retain_async"] is False
    item = kwargs["items"][0]
    assert "u1" in item["content"] and "a1" in item["content"]
    assert "u2" in item["content"] and "a2" in item["content"]
    assert item["metadata"]["session_id"] == "session-99"
    assert item["metadata"]["phase"] == "session_end"
    assert item["metadata"]["turn_count"] == "2"
    assert item["update_mode"] == "append"


def test_session_end_clears_buffer_after_success(provider):
    p, fake = provider
    p.initialize("s", home=Path.home())
    p.sync_turn("u1", "a1")
    p.on_session_end(messages=[])
    p.on_session_end(messages=[])
    assert len([c for c in fake.calls if c[0] == "aretain_batch"]) == 1


def test_session_end_propagates_retain_failures(home_redirect, monkeypatch):
    p = HindsightProvider(HindsightConfig(llm_api_key="k"))
    fake = FailOnceClient()
    monkeypatch.setattr(p, "_get_client", lambda: fake)
    p.initialize("s", home=home_redirect)
    p.sync_turn("u1", "a1")
    with pytest.raises(RuntimeError, match="temporary retain failure"):
        p.on_session_end(messages=[])


def test_session_end_noop_when_no_buffered_turns(provider):
    p, fake = provider
    p.initialize("s", home=Path.home())
    p.on_session_end(messages=[])
    assert fake.calls == []


def test_on_memory_write_mirrors_add_replace_and_skips_remove_and_empty(provider):
    p, fake = provider
    p.initialize("s", home=Path.home())
    p.on_memory_write("add", "fact A")
    p.on_memory_write("replace", "fact B")
    p.on_memory_write("remove", "fact A")  # no-op
    p.on_memory_write("add", "")  # no-op
    actions = [c[1]["metadata"]["action"] for c in fake.calls if c[0] == "aretain"]
    contents = [c[1]["content"] for c in fake.calls if c[0] == "aretain"]
    assert actions == ["add", "replace"] and contents == ["fact A", "fact B"]


def test_mirror_builtin_writes_can_be_disabled(home_redirect, monkeypatch):
    p = HindsightProvider(HindsightConfig(llm_api_key="k", mirror_builtin_writes=False))
    fake = FakeClient()
    monkeypatch.setattr(p, "_get_client", lambda: fake)
    p.initialize("s", home=home_redirect)
    p.on_memory_write("add", "fact")
    assert fake.calls == []
    p.shutdown()


def test_session_end_forces_synchronous_hindsight_retain(home_redirect, monkeypatch):
    """Task-end writes must finish before the next issue starts."""
    p = HindsightProvider(HindsightConfig(llm_api_key="k", retain_async=True))
    fake = FakeClient()
    monkeypatch.setattr(p, "_get_client", lambda: fake)
    p.initialize("s", home=home_redirect)
    p.sync_turn("u", "a")
    p.on_session_end(messages=[])
    assert fake.calls[0][1]["retain_async"] is False
    p.shutdown()


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def test_shutdown_drains_writer_and_closes_client(home_redirect, monkeypatch):
    p = HindsightProvider(HindsightConfig(llm_api_key="k"))
    fake = FakeClient()
    monkeypatch.setattr(p, "_get_client", lambda: fake)
    p._client = fake
    p.initialize("s", home=home_redirect)
    p.sync_turn("u", "a")
    p.on_session_end(messages=[])
    p.shutdown()
    assert fake.closed is True
    # Post-shutdown writes are dropped silently.
    pre = [(c[0], c[1].get("items", [{}])[0].get("content", "")) for c in fake.calls]
    p.sync_turn("u2", "a2")
    p.on_memory_write("add", "z")
    post = [(c[0], c[1].get("items", [{}])[0].get("content", "")) for c in fake.calls]
    assert pre == post


def test_session_end_waits_for_retain_before_shutdown_closes_client(home_redirect, monkeypatch):
    class SlowClient(FakeClient):
        async def aretain_batch(self, **kwargs):
            await asyncio.sleep(0.05)
            self.calls.append(("aretain_batch", kwargs))

    p = HindsightProvider(HindsightConfig(llm_api_key="k", drain_timeout=1))
    fake = SlowClient()
    monkeypatch.setattr(p, "_get_client", lambda: fake)
    p._client = fake
    p.initialize("s", home=home_redirect)
    p.sync_turn("u", "a")
    p.on_session_end(messages=[])
    p.shutdown()
    assert [c[0] for c in fake.calls] == ["aretain_batch", "aclose"]


def test_shutdown_ignores_failed_lazy_client_start():
    class FailedLazyClient:
        def __getattr__(self, _name):
            raise RuntimeError("Failed to start daemon")

    p = HindsightProvider(HindsightConfig(llm_api_key="k"))
    p._client = FailedLazyClient()
    p.shutdown()
    assert p._client is None


def test_double_shutdown_is_safe(home_redirect, monkeypatch):
    p = HindsightProvider(HindsightConfig(llm_api_key="k"))
    monkeypatch.setattr(p, "_get_client", lambda: FakeClient())
    p.initialize("s", home=home_redirect)
    p.shutdown()
    p.shutdown()  # second call must not raise
