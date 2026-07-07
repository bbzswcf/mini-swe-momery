"""End-to-end smoke test for the new memory features.

Drives `MemoryAgent` with deterministic FakeModel responses and asserts:

1. The shipped `swebench_pro.yaml` defaults to a minimal MEMORY.md + session_search setup.
2. `session_search` can find the first session's
   trace through the manager's normal tool dispatch path.
3. `every_n_steps` consolidation triggers mid-run without breaking the loop.

Uses the actual yaml file (not a hand-rolled config) so a regression in either
the yaml shape or the manager wiring shows up here.
"""

from __future__ import annotations

import json

import pytest
import yaml

from minisweagent import package_dir
from minisweagent.agents import get_agent
from minisweagent.agents.memory import MemoryAgent
from minisweagent.memory import MemoryManager, MemoryManagerConfig


class _FakeConfig:
    def model_dump(self, **_):
        return {}


class FakeModel:
    """Pops responses in order; records each `query`'s message list."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.config = _FakeConfig()
        self.seen = []
        self.extra_tools = []

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
    def __init__(self):
        self.config = _FakeConfig()

    def execute(self, action, cwd=""):
        return {"output": "ok", "returncode": 0, "exception_info": ""}

    def get_template_vars(self, **_):
        return {}

    def serialize(self):
        return {}


def _exit():
    return {
        "role": "exit",
        "content": "done",
        "extra": {"exit_status": "Submitted", "submission": "ok", "actions": [], "cost": 0.0},
    }


def _assistant(actions):
    return {"role": "assistant", "content": "...", "extra": {"actions": actions, "cost": 0.0}}


def _shared_manager(tmp_path) -> MemoryManager:
    """A MemoryManager rooted at tmp_path; reused across sessions to avoid
    `_owns_manager` closing the SessionStore after the first run."""
    cfg = MemoryManagerConfig(home=tmp_path)
    return MemoryManager(config=cfg)


def test_e2e_swebench_pro_yaml_defaults_to_memory_and_session_search(tmp_path, monkeypatch):
    """Drive the **shipped** swebench_pro.yaml through `get_agent` end-to-end."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = yaml.safe_load((package_dir / "config" / "benchmarks" / "swebench_pro.yaml").read_text())
    agent_cfg = dict(cfg["agent"])
    # The yaml's templates reference {{ task }}; provide a minimal set so jinja
    # rendering succeeds without spinning up a real env / model harness.
    agent_cfg["system_template"] = "SYS\n{{ memory_block }}"
    agent_cfg["instance_template"] = "TASK: {{ task }}"

    add_action = {
        "tool_name": "memory",
        "args": {"action": "add", "content": "ruff check --fix is the project's lint command"},
        "tool_call_id": "1",
    }
    # 1 assistant turn (writes ruff fact) + exit; shipped config keeps session_search
    # on and consolidation off so this is a clean minimal memory baseline.
    model = FakeModel([_assistant([add_action]), _exit()])
    agent = get_agent(model, FakeEnv(), agent_cfg)
    assert isinstance(agent, MemoryAgent)
    assert agent.manager.session_store is not None
    assert agent.manager.config.consolidation.on_session_end is False

    agent.run(task="lint please", session_id="trial-A")

    assert len(model.seen) == 2
    entries = (tmp_path / ".mini-memory" / "MEMORY.md").read_text()
    assert "ruff check --fix" in entries
    assert (tmp_path / ".mini-memory" / "sessions.db").exists()


def test_e2e_session_search_tool_finds_prior_trace_within_next_session(tmp_path):
    """Run two sessions on a *shared* manager so session-2 can `session_search` session-1."""
    manager = _shared_manager(tmp_path)

    # --- Session 1: agent writes a memory, exits.
    s1_actions = [
        {
            "tool_name": "memory",
            "args": {"action": "add", "content": "use ruff for lint"},
            "tool_call_id": "a1",
        }
    ]
    s1_model = FakeModel([_assistant(s1_actions), _exit()])
    s1 = MemoryAgent(
        s1_model,
        FakeEnv(),
        manager=manager,
        system_template="SYS\n{{ memory_block }}",
        instance_template="T: {{ task }}",
        cost_limit=0,
        step_limit=0,
    )
    s1.run(task="initial", session_id="trial-1")

    # --- Session 2: agent issues a `session_search` tool call for "ruff".
    search_action = {
        "tool_name": "session_search",
        "args": {"query": "ruff"},
        "tool_call_id": "s1",
    }
    s2_model = FakeModel([_assistant([search_action]), _exit()])
    s2 = MemoryAgent(
        s2_model,
        FakeEnv(),
        manager=manager,
        system_template="SYS\n{{ memory_block }}",
        instance_template="T: {{ task }}",
        cost_limit=0,
        step_limit=0,
    )
    s2.run(task="follow-up", session_id="trial-2")

    # Session 2's prompt sees the persisted MEMORY.md entry from session 1
    # (frozen-snapshot semantics — written in s1, visible in s2's system prompt).
    s2_system = s2_model.seen[0][0]["content"]
    assert "use ruff for lint" in s2_system

    # On the 2nd model.query the conversation is [system, instance, assistant, observation];
    # the observation envelope is `{"output": <json_str>, ...}` — see `_result_to_output`.
    obs = s2_model.seen[1][3]
    payload = json.loads(json.loads(obs["content"])["output"])
    assert payload["success"] is True
    assert payload["session_count"] >= 1
    assert any(s["session_id"] == "trial-1" for s in payload["sessions"])
    assert any("ruff" in h["snippet"].lower() for s in payload["sessions"] for h in s["matches"])

    manager.shutdown()


def test_e2e_every_n_steps_consolidation_triggers_mid_run(tmp_path):
    """Long-trace checkpoint flush (`every_n_steps`) doesn't disturb the main loop."""
    from minisweagent.memory import ConsolidationConfig

    cfg = MemoryManagerConfig(
        home=tmp_path,
        consolidation=ConsolidationConfig(every_n_steps=2, max_actions=1),
    )
    manager = MemoryManager(config=cfg)

    checkpoint = {
        "tool_name": "memory",
        "args": {"action": "add", "content": "checkpoint fact"},
        "tool_call_id": "c1",
    }
    # 4 step responses so n_calls hits 2 and 4 (each triggers consolidation)
    # interleaved with consolidation responses (2 of them).
    responses = [
        _assistant([]),
        _assistant([]),
        {"extra": {"actions": [checkpoint]}},  # consolidation #1 (n_calls=2)
        _assistant([]),
        _exit(),
        {"extra": {"actions": []}},  # consolidation #2 (n_calls=4)
    ]
    model = FakeModel(responses)
    agent = MemoryAgent(
        model,
        FakeEnv(),
        manager=manager,
        system_template="SYS\n{{ memory_block }}",
        instance_template="T: {{ task }}",
        cost_limit=0,
        step_limit=0,
    )
    agent.run(task="long", session_id="checkpoint-trial")

    # Step responses (4) + consolidation turns (2) = 6 total model.query calls.
    assert len(model.seen) == 6
    # The first consolidation's memory.add did land on disk.
    assert (tmp_path / "MEMORY.md").read_text().strip() == "checkpoint fact"
    manager.shutdown()


@pytest.mark.parametrize(
    ("query", "expected_count"),
    [
        ("ruff", 1),
        ("nonexistent_token_xyz", 0),
        ("\"ruff check\"", 1),
    ],
)
def test_session_search_supports_fts5_query_syntax(tmp_path, query, expected_count):
    """Round-trip a session through the manager and verify FTS5 features work."""
    manager = _shared_manager(tmp_path)
    manager.initialize("trial-fts")
    manager.on_session_end(
        [
            {"role": "user", "content": "should I use ruff or flake8?"},
            {"role": "assistant", "content": "this repo uses ruff check --fix"},
        ]
    )
    res = manager.handle_tool_call("session_search", {"query": query, "limit": 10})
    assert res["success"]
    assert res["session_count"] == expected_count
    manager.shutdown()
