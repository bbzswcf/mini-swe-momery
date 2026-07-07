"""End-to-end integration tests for the memory subsystem.

Covers the wiring between the SWE-bench memory config -> `get_agent` ->
`MemoryAgent` -> `MemoryManager.from_config` -> registered provider, including
the parts that unit tests stub out:

- `MemoryManager.from_config(dict)` builds the manager + (optionally) a provider
- `MemoryAgent` either *owns* an internally-built manager (calls `shutdown` on
  exit) or *borrows* an external one (leaves it alone — needed so SWE-bench
  batches can reuse a manager across instances)
- The frozen-snapshot semantics survive multiple back-to-back `agent.run` calls
- The shipped yaml in `src/minisweagent/config/benchmarks/swebench_pro.yaml`
  actually loads through `get_agent` end-to-end with no special handling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from minisweagent import package_dir
from minisweagent.agents import get_agent
from minisweagent.agents.memory import MemoryAgent
from minisweagent.config import get_config_from_spec
from minisweagent.memory import (
    BuiltinMemory,
    BuiltinMemoryConfig,
    MemoryManager,
    MemoryManagerConfig,
    MemoryProvider,
)
from minisweagent.utils.serialize import recursive_merge

SYSTEM_TEMPLATE = "SYSTEM\n{{ memory_block }}"
INSTANCE_TEMPLATE = "TASK: {{ task }}"


class _FakeConfig:
    def model_dump(self, **_):
        return {}


class FakeModel:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.config = _FakeConfig()
        self.seen: list[list[dict]] = []
        self.extra_tools: list[dict] = []

    def query(self, messages, **_):
        self.seen.append([dict(m) for m in messages])
        return self.responses.pop(0)

    def format_message(self, **kw):
        return dict(kw)

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [{"role": "user", "content": json.dumps(o), "extra": {}} for o in outputs]

    def get_template_vars(self, **_):
        return {}

    def serialize(self):
        return {}


class FakeEnv:
    def __init__(self) -> None:
        self.config = _FakeConfig()

    def execute(self, action, cwd=""):
        return {"output": "ok", "returncode": 0, "exception_info": ""}

    def get_template_vars(self, **_):
        return {}

    def serialize(self):
        return {}


class TrackingProvider(MemoryProvider):
    """Records every lifecycle hook + tool call so tests can assert ordering."""

    name = "tracking"

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def is_available(self):
        return True

    def initialize(self, session_id, *, home, **_):
        self.events.append(("initialize", session_id, Path(home)))

    def get_tool_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "track_note",
                    "description": "test",
                    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                },
            }
        ]

    def system_prompt_block(self):
        return "PROVIDER_BLOCK"

    def handle_tool_call(self, name, args):
        self.events.append(("tool", name, dict(args)))
        return {"success": True, "stored": args.get("text", "")}

    def on_memory_write(self, action, content):
        self.events.append(("on_memory_write", action, content))

    def on_session_end(self, messages):
        self.events.append(("on_session_end", len(messages)))

    def shutdown(self):
        self.events.append(("shutdown",))


class FailingSessionEndProvider(TrackingProvider):
    def on_session_end(self, messages):
        raise RuntimeError("provider flush failed")


def _exit_msg() -> dict:
    return {
        "role": "exit",
        "content": "done",
        "extra": {"exit_status": "Submitted", "submission": "ok", "actions": [], "cost": 0.0},
    }


def _assistant(actions: list[dict]) -> dict:
    return {"role": "assistant", "content": "...", "extra": {"actions": actions, "cost": 0.0}}


# ---------------------------------------------------------------------------
# MemoryManager.from_config
# ---------------------------------------------------------------------------


def test_from_config_empty_returns_manager_with_no_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    mgr = MemoryManager.from_config({})
    assert mgr.provider is None
    assert mgr.config.char_limit == 48_000
    assert mgr.config.home == tmp_path / ".mini-memory"
    # Sessions are enabled by default — `session_search` is wired alongside `memory`.
    assert mgr.tool_names == {"memory", "session_search"}


def test_from_config_expands_tilde_home_and_overrides_char_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    mgr = MemoryManager.from_config({"home": "~/.custom-mem", "char_limit": 5000})
    assert mgr.config.home == tmp_path / ".custom-mem"
    assert mgr.config.char_limit == 5000


def test_from_config_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown memory provider"):
        MemoryManager.from_config({"provider": "redis"})


def test_from_config_provider_blocks_appear_only_when_configured(tmp_path):
    """Sanity: with no provider, no provider tool names leak in."""
    mgr = MemoryManager.from_config({"home": str(tmp_path)})
    assert mgr.tool_names == {"memory", "session_search"}
    assert mgr.system_prompt_block() == ""


# ---------------------------------------------------------------------------
# MemoryAgent ownership semantics
# ---------------------------------------------------------------------------


def _agent_with_memory_dict(tmp_path, responses, **mem_overrides):
    memory = {"home": str(tmp_path / "owned"), "char_limit": 200, **mem_overrides}
    return MemoryAgent(
        FakeModel(responses),
        FakeEnv(),
        memory=memory,
        system_template=SYSTEM_TEMPLATE,
        instance_template=INSTANCE_TEMPLATE,
        cost_limit=0,
        step_limit=0,
    )


def _agent_with_external_manager(manager, responses):
    return MemoryAgent(
        FakeModel(responses),
        FakeEnv(),
        manager=manager,
        system_template=SYSTEM_TEMPLATE,
        instance_template=INSTANCE_TEMPLATE,
        cost_limit=0,
        step_limit=0,
    )


def test_owned_manager_has_shutdown_called_on_exit(tmp_path):
    agent = _agent_with_memory_dict(tmp_path, [_exit_msg()])
    shutdown_calls: list[int] = []
    agent.manager.shutdown = lambda: shutdown_calls.append(1)
    agent.run(task="t", session_id="s")
    assert shutdown_calls == [1]


def test_external_manager_is_not_shut_down_by_agent(tmp_path):
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=200))
    manager = MemoryManager(config=MemoryManagerConfig(home=tmp_path), builtin=builtin)
    shutdown_calls: list[int] = []
    manager.shutdown = lambda: shutdown_calls.append(1)
    _agent_with_external_manager(manager, [_exit_msg()]).run(task="t", session_id="s")
    assert shutdown_calls == []


def test_owned_manager_shutdown_runs_even_when_session_raises(tmp_path):
    agent = _agent_with_memory_dict(tmp_path, [])  # no responses -> IndexError on first query
    shutdown_calls: list[int] = []
    agent.manager.shutdown = lambda: shutdown_calls.append(1)
    with pytest.raises(IndexError):
        agent.run(task="t", session_id="boom")
    assert shutdown_calls == [1]


def test_passing_both_manager_and_memory_dict_raises(tmp_path):
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=100))
    manager = MemoryManager(config=MemoryManagerConfig(home=tmp_path), builtin=builtin)
    with pytest.raises(ValueError, match="Pass either"):
        MemoryAgent(
            FakeModel([]),
            FakeEnv(),
            manager=manager,
            memory={"char_limit": 100},
            system_template=SYSTEM_TEMPLATE,
            instance_template=INSTANCE_TEMPLATE,
            cost_limit=0,
            step_limit=0,
        )


def test_manager_rejects_mismatched_builtin_path_and_home(tmp_path):
    """Provider state lives under `home`; the built-in store must too, or providers and the
    `MEMORY.md` file end up in different directories and per-instance isolation breaks."""
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "elsewhere" / "MEMORY.md", char_limit=100))
    with pytest.raises(ValueError, match="must equal"):
        MemoryManager(config=MemoryManagerConfig(home=tmp_path), builtin=builtin)


# ---------------------------------------------------------------------------
# Cross-session flows on a reused manager
# ---------------------------------------------------------------------------


def test_frozen_snapshot_refreshes_between_back_to_back_sessions(tmp_path):
    """Session 1 writes 'fact A' to disk; session 2's prompt reflects it."""
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=200))
    manager = MemoryManager(config=MemoryManagerConfig(home=tmp_path), builtin=builtin)

    s1_responses = [
        _assistant([{"tool_name": "memory", "args": {"action": "add", "content": "fact A"}, "tool_call_id": "1"}]),
        _exit_msg(),
    ]
    s1 = _agent_with_external_manager(manager, s1_responses)
    s1.run(task="first", session_id="s1")

    s2 = _agent_with_external_manager(manager, [_exit_msg()])
    s2.run(task="second", session_id="s2")

    s1_system = s1.model.seen[0][0]["content"]
    s2_system = s2.model.seen[0][0]["content"]
    assert "fact A" not in s1_system  # frozen snapshot taken before the write
    assert "fact A" in s2_system  # next session's snapshot picked it up


def test_provider_lifecycle_dispatched_in_order_with_memory_write_mirroring(tmp_path):
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=200))
    manager = MemoryManager(config=MemoryManagerConfig(home=tmp_path), builtin=builtin)
    provider = TrackingProvider()
    manager.register(provider)

    actions = [
        {"tool_name": "memory", "args": {"action": "add", "content": "stable fact"}, "tool_call_id": "1"},
        {"tool_name": "track_note", "args": {"text": "narrative"}, "tool_call_id": "2"},
    ]
    _agent_with_external_manager(manager, [_assistant(actions), _exit_msg()]).run(task="t", session_id="abc")

    kinds = [e[0] for e in provider.events]
    assert kinds[0] == "initialize"
    assert ("on_memory_write", "add", "stable fact") in provider.events
    assert ("tool", "track_note", {"text": "narrative"}) in provider.events
    assert kinds[-1] == "on_session_end"
    assert ("shutdown",) not in provider.events  # external manager → not shut down


def test_provider_session_end_failure_does_not_mask_submission(tmp_path):
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=200))
    manager = MemoryManager(config=MemoryManagerConfig(home=tmp_path), builtin=builtin)
    manager.register(FailingSessionEndProvider())
    assert _agent_with_external_manager(manager, [_exit_msg()]).run(task="t", session_id="s")["exit_status"] == "Submitted"


def test_provider_tool_schemas_reach_the_model_via_extra_tools(tmp_path):
    """Without this, the LLM never sees `memory` or any provider tool — silently broken (P0-1)."""
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=200))
    manager = MemoryManager(config=MemoryManagerConfig(home=tmp_path), builtin=builtin)
    manager.register(TrackingProvider())
    agent = _agent_with_external_manager(manager, [_exit_msg()])
    names = {t["function"]["name"] for t in agent.model.extra_tools}
    assert names == {"memory", "session_search", "track_note"}


def test_provider_block_appears_in_system_prompt_alongside_builtin(tmp_path):
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=200))
    builtin.add("baseline fact")
    manager = MemoryManager(config=MemoryManagerConfig(home=tmp_path), builtin=builtin)
    manager.register(TrackingProvider())
    agent = _agent_with_external_manager(manager, [_exit_msg()])
    agent.run(task="t", session_id="s")
    system_prompt = agent.model.seen[0][0]["content"]
    assert "baseline fact" in system_prompt
    assert "PROVIDER_BLOCK" in system_prompt


# ---------------------------------------------------------------------------
# Shipped yaml + get_agent integration
# ---------------------------------------------------------------------------


@pytest.fixture
def swebench_pro_cfg() -> dict:
    return yaml.safe_load((package_dir / "config" / "benchmarks" / "swebench_pro.yaml").read_text())


def test_swebench_pro_yaml_loads_with_expected_shape(swebench_pro_cfg):
    agent_cfg = swebench_pro_cfg["agent"]
    assert agent_cfg["agent_class"] == "memory"
    assert "{{ memory_block }}" in agent_cfg["system_template"]
    assert "SWE-bench" in agent_cfg["system_template"]
    mem = agent_cfg["memory"]
    assert mem["provider"] is None  # default off; opt-in via uncomment
    assert mem["char_limit"] == 48_000
    assert mem["sessions_enabled"] is True
    assert mem["consolidation"]["on_session_end"] is False
    assert mem["consolidation"]["every_n_steps"] == 0


def test_swebench_pro_filesystem_overlay_is_pluggable_and_prompt_aligned(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = recursive_merge(
        get_config_from_spec("swebench_pro.yaml"),
        get_config_from_spec("swebench_pro_filesystem.yaml"),
    )

    mem = cfg["agent"]["memory"]
    manager = MemoryManager.from_config(mem)
    manager.initialize("repo__issue-1", chain_id="chain-a", step_index=1)

    assert manager.tool_names == set()
    assert manager.filesystem_memory is not None
    assert manager.filesystem_memory.config.enabled is True
    assert "{{ memory_block }}" in cfg["agent"]["system_template"]
    assert "filesystem_memory" in manager.system_prompt_block()
    assert "`session_search`" not in cfg["agent"]["instance_template"]
    assert "`memory`" not in cfg["agent"]["instance_template"]
    assert "`session_search`" not in cfg["model"]["format_error_template"]
    assert "`memory`" not in cfg["model"]["format_error_template"]


def test_get_agent_builds_memory_agent_from_yaml(swebench_pro_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    agent_cfg = dict(swebench_pro_cfg["agent"])
    agent = get_agent(FakeModel([]), FakeEnv(), agent_cfg)
    assert isinstance(agent, MemoryAgent)
    assert agent._owns_manager is True
    assert agent.manager.provider is None
    assert agent.manager.config.home == tmp_path / ".mini-memory"
    assert agent.manager.session_store is not None
    assert agent.manager.config.consolidation.on_session_end is False


# ---------------------------------------------------------------------------
# Session store + consolidation wired through the agent
# ---------------------------------------------------------------------------


def test_session_transcript_indexed_and_searchable_after_run(tmp_path):
    """Running a session writes its messages to the FTS store and `session_search` finds them.

    Uses an external manager so the store survives `agent.run()` (owned managers
    close the store on shutdown — that's the whole point of `_owns_manager`).
    """
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=200))
    manager = MemoryManager(config=MemoryManagerConfig(home=tmp_path), builtin=builtin)
    actions = [
        {"tool_name": "memory", "args": {"action": "add", "content": "use ruff for lint"}, "tool_call_id": "1"}
    ]
    _agent_with_external_manager(manager, [_assistant(actions), _exit_msg()]).run(task="how?", session_id="trial-77")

    hits = manager.session_store.search("ruff")
    assert hits, "session_search should find the trace we just ran"
    assert {h["session_id"] for h in hits} == {"trial-77"}


def test_consolidation_at_session_end_fires_extra_model_query(tmp_path):
    """When `consolidation.on_session_end` is set, the agent issues a final memory-only LLM turn."""
    add_action = {"tool_name": "memory", "args": {"action": "add", "content": "consolidated"}}
    # Three responses: an assistant turn, the exit message, and the consolidation turn.
    responses = [_assistant([]), _exit_msg(), {"extra": {"actions": [add_action]}}]
    agent = _agent_with_memory_dict(
        tmp_path,
        responses,
        consolidation={"on_session_end": True, "max_actions": 1},
    )
    agent.run(task="t", session_id="cons-1")

    assert agent.manager.builtin.load() == ["consolidated"]
    # 3 model.query calls: 1 to produce the first assistant turn, 1 for the
    # exit message, and 1 final consolidation turn from on_session_end.
    assert len(agent.model.seen) == 3
