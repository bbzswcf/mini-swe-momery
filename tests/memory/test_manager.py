from __future__ import annotations

import pytest

from minisweagent.memory import (
    SESSION_SEARCH_TOOL_SCHEMA,
    BuiltinMemory,
    BuiltinMemoryConfig,
    ConsolidationConfig,
    FileSystemMemoryConfig,
    MemoryManager,
    MemoryManagerConfig,
    MemoryProvider,
)


class MinimalProvider(MemoryProvider):
    """Implements only abstract methods — exercises default no-op hooks."""

    name = "minimal"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id, *, home, **_):
        pass

    def get_tool_schemas(self):
        return []


class FakeProvider(MemoryProvider):
    name = "fake"

    def __init__(self):
        self.calls: list = []

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id, *, home, **kwargs):
        self.calls.append(("initialize", session_id, home, kwargs))

    def get_tool_schemas(self):
        self.calls.append(("get_tool_schemas",))
        return [
            {
                "type": "function",
                "function": {"name": "fake_search", "description": "x", "parameters": {}},
            }
        ]

    def handle_tool_call(self, name, args):
        self.calls.append(("handle_tool_call", name, args))
        return {"success": True, "result": "fake"}

    def system_prompt_block(self):
        return "[fake provider block]"

    def on_memory_write(self, action, content):
        self.calls.append(("on_memory_write", action, content))

    def sync_turn(self, user, assistant):
        self.calls.append(("sync_turn", user, assistant))

    def on_session_end(self, messages):
        self.calls.append(("on_session_end", len(messages)))

    def shutdown(self):
        self.calls.append(("shutdown",))


class FailingHookProvider(FakeProvider):
    def on_memory_write(self, action, content):
        raise RuntimeError("mirror down")

    def sync_turn(self, user, assistant):
        raise RuntimeError("sync down")

    def on_session_end(self, messages):
        raise RuntimeError("flush down")


@pytest.fixture
def manager(tmp_path):
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=200))
    return MemoryManager(config=MemoryManagerConfig(home=tmp_path), builtin=builtin)


def test_default_manager_exposes_builtin_and_session_search_tools(manager):
    """Sessions are enabled by default → both `memory` and `session_search` are wired."""
    assert manager.tool_names == {"memory", "session_search"}
    manager.initialize("s1")  # captures snapshot (empty so far)
    assert manager.system_prompt_block() == ""
    manager.builtin.add("a fact about the project")
    manager.initialize("s2")  # next session re-captures with the new entry
    assert "a fact about the project" in manager.system_prompt_block()


def test_initialize_refreshes_snapshot_between_sessions(manager):
    manager.builtin.add("session-1 fact")
    manager.initialize("s1")
    assert "session-1 fact" in manager.system_prompt_block()
    manager.handle_tool_call("memory", {"action": "add", "content": "session-1 in-flight"})
    # mid-session write must not appear in the frozen snapshot
    assert "session-1 in-flight" not in manager.system_prompt_block()
    manager.initialize("s2")
    assert "session-1 in-flight" in manager.system_prompt_block()


def test_initialize_forwards_extra_kwargs_to_provider(manager, tmp_path):
    fake = FakeProvider()
    manager.register(fake)
    manager.initialize("session-x", platform="cli", agent_role="reviewer")
    init_call = next(c for c in fake.calls if c[0] == "initialize")
    assert init_call[1] == "session-x"
    assert init_call[2] == tmp_path
    assert init_call[3] == {"platform": "cli", "agent_role": "reviewer"}


def test_handle_builtin_routes_three_actions_and_rejects_unknown(manager):
    assert manager.handle_tool_call("memory", {"action": "add", "content": "first"})["success"]
    assert manager.handle_tool_call("memory", {"action": "replace", "old_text": "first", "content": "second"})["success"]
    assert manager.builtin.load() == ["second"]
    assert manager.handle_tool_call("memory", {"action": "remove", "old_text": "second"})["success"]
    assert manager.builtin.load() == []
    assert not manager.handle_tool_call("memory", {"action": "noop"})["success"]
    assert not manager.handle_tool_call("memory", {"action": "remove", "old_text": "absent"})["success"]


def test_register_caches_provider_tool_names(manager):
    """Tool-name lookup must not call provider.get_tool_schemas() per dispatch."""
    fake = FakeProvider()
    manager.register(fake)
    assert manager._provider_tool_names == {"fake_search"}
    schema_calls_before = sum(1 for c in fake.calls if c == ("get_tool_schemas",))
    for _ in range(5):
        manager.handle_tool_call("fake_search", {})
        manager.handle_tool_call("memory", {"action": "add", "content": f"x{_}"})
    assert sum(1 for c in fake.calls if c == ("get_tool_schemas",)) == schema_calls_before


def test_register_enforces_single_provider_rule(manager):
    manager.register(FakeProvider())
    with pytest.raises(RuntimeError, match="single-provider"):
        manager.register(FakeProvider())


def test_provider_registration_extends_schemas_and_routes(manager):
    fake = FakeProvider()
    manager.register(fake)
    assert manager.tool_names == {"memory", "session_search", "fake_search"}
    assert "[fake provider block]" in manager.system_prompt_block()
    assert manager.handle_tool_call("fake_search", {"q": "x"}) == {"success": True, "result": "fake"}
    assert ("handle_tool_call", "fake_search", {"q": "x"}) in fake.calls
    assert not manager.handle_tool_call("ghost", {})["success"]


def test_on_memory_write_fires_only_on_successful_builtin_writes(manager):
    fake = FakeProvider()
    manager.register(fake)
    manager.handle_tool_call("memory", {"action": "add", "content": "hello"})
    manager.handle_tool_call("memory", {"action": "remove", "old_text": "absent"})  # should fail
    assert ("on_memory_write", "add", "hello") in fake.calls
    assert not any(c[0] == "on_memory_write" and c[1] == "remove" for c in fake.calls)


def test_provider_memory_write_failure_does_not_break_builtin_write(manager):
    manager.register(FailingHookProvider())
    res = manager.handle_tool_call("memory", {"action": "add", "content": "keep this fact"})
    assert res["success"] is True
    assert manager.builtin.load() == ["keep this fact"]


def test_provider_lifecycle_failures_do_not_break_local_memory_paths(tmp_path):
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=400))
    mgr = MemoryManager(
        config=MemoryManagerConfig(
            home=tmp_path,
            consolidation=ConsolidationConfig(on_session_end=True),
        ),
        builtin=builtin,
    )
    mgr.register(FailingHookProvider())
    mgr.initialize("trial-1")
    mgr.sync_turn("user", "assistant")
    mgr.on_session_end(
        [{"role": "user", "content": "pytest gotcha"}],
        model=_RecordingModel({"extra": {"actions": [{"tool_name": "memory", "args": {"action": "add", "content": "fact"}}]}}),
    )
    assert builtin.load() == ["fact"]
    assert mgr.handle_tool_call("session_search", {"query": "pytest"})["session_count"] == 1


def test_lifecycle_methods_delegate_to_provider(manager, tmp_path):
    fake = FakeProvider()
    manager.register(fake)
    manager.initialize("session-42")
    manager.sync_turn("u", "a")
    manager.on_session_end([{}, {}, {}])
    manager.shutdown()
    lifecycle_calls = [c for c in fake.calls if c[0] != "get_tool_schemas"]
    assert lifecycle_calls == [
        ("initialize", "session-42", tmp_path, {}),
        ("sync_turn", "u", "a"),
        ("on_session_end", 3),
        ("shutdown",),
    ]


def test_minimal_provider_uses_default_noop_hooks(manager):
    """A provider that overrides only the abstract methods still cooperates with all manager hooks."""
    manager.register(MinimalProvider())
    assert manager.system_prompt_block() == ""
    manager.initialize("s1")
    manager.sync_turn("u", "a")
    manager.on_session_end([])
    manager.shutdown()


def test_handle_tool_call_returns_default_error_for_unrouted_provider_tool(manager):
    """When a tool name is not registered with any provider, manager rejects it cleanly."""
    res = manager.handle_tool_call("not_registered", {})
    assert res["success"] is False


# ---------------------------------------------------------------------------
# Session search wiring + consolidation triggers
# ---------------------------------------------------------------------------


def test_sessions_disabled_drops_tool_and_store(tmp_path):
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=200))
    mgr = MemoryManager(
        config=MemoryManagerConfig(home=tmp_path, sessions_enabled=False),
        builtin=builtin,
    )
    assert mgr.session_store is None
    assert "session_search" not in mgr.tool_names
    res = mgr.handle_tool_call("session_search", {"query": "x"})
    assert res["success"] is False


def test_session_search_schema_encourages_code_task_recall_without_forcing_it():
    description = SESSION_SEARCH_TOOL_SCHEMA["function"]["description"]
    assert "USE THIS PROACTIVELY" in description
    assert "current repo, file, failing test, stack trace" in description
    assert "clear cross-session signal" in description
    assert "Skip it when the task looks new" in description


def test_session_search_round_trip_via_tool_dispatch(tmp_path):
    """`on_session_end` indexes the trace; `session_search` tool retrieves it."""
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=200))
    mgr = MemoryManager(config=MemoryManagerConfig(home=tmp_path), builtin=builtin)
    mgr.initialize("trial-1")
    mgr.on_session_end(
        [
            {"role": "user", "content": "use ruff for linting"},
            {"role": "assistant", "content": "running ruff check --fix"},
        ]
    )
    res = mgr.handle_tool_call("session_search", {"query": "ruff", "limit": 5})
    assert res["success"]
    assert res["session_count"] == 1
    assert {s["session_id"] for s in res["sessions"]} == {"trial-1"}
    assert any("ruff" in h["snippet"].lower() for s in res["sessions"] for h in s["matches"])
    assert "use ruff for linting" in res["sessions"][0]["summary"]


def test_session_search_rejects_empty_and_nonint_limit(tmp_path):
    """Tool dispatch must not crash on bad model arguments."""
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=200))
    mgr = MemoryManager(config=MemoryManagerConfig(home=tmp_path), builtin=builtin)
    assert mgr.handle_tool_call("session_search", {"query": "  "})["success"] is False
    # `None` limit should fall back to default, not crash
    assert mgr.handle_tool_call("session_search", {"query": "anything", "limit": None})["success"]
    assert mgr.handle_tool_call("session_search", {"query": "anything", "limit": int})["success"]


class _RecordingModel:
    def __init__(self, response: dict | None = None):
        self.response = response or {"extra": {"actions": []}}
        self.seen: list[list[dict]] = []

    def query(self, messages):
        self.seen.append(messages)
        return self.response


def test_consolidation_at_session_end_writes_to_builtin(tmp_path):
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=400))
    mgr = MemoryManager(
        config=MemoryManagerConfig(
            home=tmp_path,
            consolidation=ConsolidationConfig(on_session_end=True, max_actions=2),
        ),
        builtin=builtin,
    )
    model = _RecordingModel(
        {
            "extra": {
                "actions": [
                    {"tool_name": "memory", "args": {"action": "add", "content": "consolidated fact"}},
                    {"tool_name": "bash", "args": {"command": "ls"}},
                ]
            }
        }
    )
    mgr.initialize("s1")
    mgr.on_session_end([{"role": "user", "content": "task X"}], model=model)
    assert builtin.load() == ["consolidated fact"]
    assert len(model.seen) == 1


def test_consolidation_off_by_default(tmp_path):
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=400))
    mgr = MemoryManager(config=MemoryManagerConfig(home=tmp_path), builtin=builtin)
    model = _RecordingModel()
    mgr.initialize("s1")
    mgr.on_session_end([{"role": "user", "content": "x"}], model=model)
    assert model.seen == []
    assert builtin.load() == []


def test_maybe_consolidate_after_idle_steps_without_memory_writes(tmp_path):
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=400))
    mgr = MemoryManager(
        config=MemoryManagerConfig(
            home=tmp_path,
            consolidation=ConsolidationConfig(every_n_steps=3),
        ),
        builtin=builtin,
    )
    model = _RecordingModel()
    assert mgr.maybe_consolidate(model, [{"role": "user", "content": "ok"}], n_calls=1) is None
    assert mgr.handle_tool_call("memory", {"action": "add", "content": "manual fact"})["success"]
    assert mgr.maybe_consolidate(model, [{"role": "user", "content": "ok"}], n_calls=2) is None

    triggered = [
        mgr.maybe_consolidate(model, [{"role": "user", "content": "ok"}], n_calls=i) is not None
        for i in range(3, 9)
    ]
    assert triggered == [False, False, True, False, False, True]
    assert len(model.seen) == 2


def test_builtin_disabled_exposes_provider_tools_only(tmp_path):
    mgr = MemoryManager.from_config(
        {"home": str(tmp_path), "builtin_enabled": False, "sessions_enabled": False, "provider": "hindsight"}
    )
    assert mgr.builtin is None
    assert "memory" not in mgr.tool_names
    assert "session_search" not in mgr.tool_names
    assert any(s["function"]["name"].startswith("hindsight_") for s in mgr.get_tool_schemas())


def test_from_config_threads_consolidation_and_sessions_paths(tmp_path):
    cfg = {
        "home": str(tmp_path),
        "char_limit": 300,
        "sessions_enabled": True,
        "sessions_path": str(tmp_path / "custom.db"),
        "consolidation": {"on_session_end": True, "every_n_steps": 5, "max_actions": 7},
    }
    mgr = MemoryManager.from_config(cfg)
    assert mgr.config.consolidation.on_session_end is True
    assert mgr.config.consolidation.every_n_steps == 5
    assert mgr.config.consolidation.max_actions == 7
    assert mgr.session_store is not None
    assert (tmp_path / "custom.db").exists()


def test_from_config_enables_filesystem_memory_prompt_and_session_end_write(tmp_path):
    cfg = {
        "home": str(tmp_path),
        "sessions_enabled": False,
        "filesystem": {"enabled": True, "chain_id": "chain-a"},
    }
    mgr = MemoryManager.from_config(cfg)
    mgr.initialize("repo__issue-1", step_index=2)

    block = mgr.system_prompt_block()
    assert "filesystem memory" in block.lower()
    assert str(tmp_path / "fs" / "chains" / "chain-a") in block

    model = _RecordingModel(
        {
            "content": '{"summary_md":"# repo__issue-1\\n\\n## Task\\nFix x.\\n","index_row":{"summary":"Fix x","files_symbols":"x.py","tests_errors":"pytest"},"repo_updates":""}'
        }
    )
    mgr.on_session_end(
        [
            {"role": "user", "content": "Fix x.py"},
            {"role": "assistant", "content": "run pytest"},
            {"role": "exit", "content": "Submitted", "extra": {"submission": "diff --git a/x.py b/x.py\n+fix\n"}},
        ],
        model=model,
    )

    case_dir = tmp_path / "fs" / "chains" / "chain-a" / "cases" / "2-repo__issue-1"
    assert (case_dir / "trajectory.md").exists()
    assert (case_dir / "summary.md").read_text().startswith("# repo__issue-1")
    assert "Fix x" in (tmp_path / "fs" / "chains" / "chain-a" / "INDEX.md").read_text()


def test_filesystem_memory_inherits_manager_home_for_manual_config(tmp_path):
    mgr = MemoryManager(
        config=MemoryManagerConfig(
            home=tmp_path,
            sessions_enabled=False,
            filesystem=FileSystemMemoryConfig(enabled=True, chain_id="chain-a"),
        )
    )

    mgr.initialize("repo__issue-1")

    assert mgr.filesystem_memory is not None
    assert mgr.filesystem_memory.chain_dir == tmp_path / "fs" / "chains" / "chain-a"


def test_filesystem_memory_disabled_by_default(tmp_path):
    mgr = MemoryManager(config=MemoryManagerConfig(home=tmp_path, filesystem=FileSystemMemoryConfig(enabled=False)))
    mgr.initialize("repo__issue-1")
    mgr.on_session_end([{"role": "user", "content": "x"}], model=_RecordingModel({"content": "{}"}))
    assert mgr.filesystem_memory is None
    assert not (tmp_path / "fs").exists()
