"""Integration tests for `MemoryAgent` using fake model + env."""

from __future__ import annotations

import json

import pytest

from minisweagent.agents.memory import MemoryAgent
from minisweagent.memory import (
    BuiltinMemory,
    BuiltinMemoryConfig,
    MemoryManager,
    MemoryManagerConfig,
    MemoryProvider,
)

SYSTEM_TEMPLATE = "SYSTEM\n{{ memory_block }}"
INSTANCE_TEMPLATE = "TASK: {{ task }}"


class _FakeConfig:
    def model_dump(self, **_):
        return {}


class FakeModel:
    """Pops scripted responses from a queue; raises if called when empty."""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.config = _FakeConfig()
        self.seen_messages: list[list[dict]] = []
        # MemoryAgent populates this at construction with manager.get_tool_schemas().
        self.extra_tools: list[dict] = []

    def query(self, messages, **_):
        self.seen_messages.append([dict(m) for m in messages])
        return self.responses.pop(0)

    def format_message(self, **kw):
        return dict(kw)

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [{"role": "user", "content": json.dumps(o), "extra": {}} for o in outputs]

    def get_template_vars(self, **_):
        return {}

    def serialize(self):
        return {}


class ResponseStyleFakeModel(FakeModel):
    def format_observation_messages(self, message, outputs, template_vars=None):
        actions = message.get("extra", {}).get("actions", [])
        return [
            {
                "type": "function_call_output",
                "call_id": action.get("tool_call_id"),
                "output": output["output"],
                "extra": {"raw_output": output["output"], "returncode": output["returncode"]},
            }
            for action, output in zip(actions, outputs)
        ]


class FakeEnv:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.config = _FakeConfig()

    def execute(self, action, cwd=""):
        self.calls.append(action)
        return {"output": f"ran:{action.get('command', '')}", "returncode": 0, "exception_info": ""}

    def get_template_vars(self, **_):
        return {}

    def serialize(self):
        return {}


class RecordingProvider(MemoryProvider):
    name = "rec"

    def __init__(self, *, schemas: list[dict] | None = None, initial_context: str = "") -> None:
        self.calls: list = []
        self._schemas = schemas or []
        self._initial_context = initial_context

    def is_available(self):
        return True

    def initialize(self, session_id, *, home, **_):
        self.calls.append(("initialize", session_id))

    def get_tool_schemas(self):
        return list(self._schemas)

    def handle_tool_call(self, name, args):
        self.calls.append(("handle_tool_call", name, args))
        return {"success": True, "result": f"rec-{name}"}

    def initial_context(self, query):
        self.calls.append(("initial_context", query))
        return self._initial_context

    def sync_turn(self, user, assistant):
        self.calls.append(("sync_turn", user, assistant))

    def on_session_end(self, messages):
        self.calls.append(("on_session_end", len(messages)))


def _exit_msg() -> dict:
    return {
        "role": "exit",
        "content": "done",
        "extra": {"exit_status": "Submitted", "submission": "ok", "actions": [], "cost": 0.0},
    }


def _assistant_msg(actions: list[dict]) -> dict:
    return {"role": "assistant", "content": "...", "extra": {"actions": actions, "cost": 0.0}}


@pytest.fixture
def manager(tmp_path):
    builtin = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=200))
    return MemoryManager(config=MemoryManagerConfig(home=tmp_path), builtin=builtin)


def _build(manager, model_responses: list[dict]) -> tuple[MemoryAgent, FakeModel, FakeEnv]:
    model, env = FakeModel(model_responses), FakeEnv()
    agent = MemoryAgent(
        model,
        env,
        manager=manager,
        system_template=SYSTEM_TEMPLATE,
        instance_template=INSTANCE_TEMPLATE,
        cost_limit=0,
        step_limit=0,
    )
    return agent, model, env


def test_memory_block_renders_into_system_prompt(manager):
    manager.builtin.add("project uses pytest")
    agent, model, _ = _build(manager, [_exit_msg()])
    agent.run(task="solve issue")
    system_prompt = model.seen_messages[0][0]["content"]
    assert "SYSTEM" in system_prompt
    assert "project uses pytest" in system_prompt
    assert "MEMORY (your persistent notes)" in system_prompt


def test_bash_action_routed_to_env_memory_action_routed_to_manager(manager):
    actions = [
        {"tool_name": "memory", "args": {"action": "add", "content": "fact A"}, "tool_call_id": "t1"},
        {"command": "ls -la", "tool_call_id": "t2"},
    ]
    agent, _, env = _build(manager, [_assistant_msg(actions), _exit_msg()])
    agent.run(task="t")
    assert env.calls == [{"command": "ls -la", "tool_call_id": "t2"}]
    assert manager.builtin.load() == ["fact A"]


def test_unknown_tool_name_falls_through_to_env(manager):
    """An action whose `tool_name` is not in the manager's tool set goes to env."""
    actions = [{"tool_name": "not_a_memory_tool", "command": "echo hi"}]
    agent, _, env = _build(manager, [_assistant_msg(actions), _exit_msg()])
    agent.run(task="t")
    assert env.calls == actions


def test_memory_action_observation_includes_serialized_result(manager):
    actions = [{"tool_name": "memory", "args": {"action": "add", "content": "x"}, "tool_call_id": "t"}]
    agent, _, _ = _build(manager, [_assistant_msg(actions), _exit_msg()])
    agent.run(task="t")
    obs = agent.messages[3]
    assert obs["role"] == "user"
    rendered = json.loads(obs["content"])
    assert rendered["returncode"] == 0 and rendered["exception_info"] == ""
    payload = json.loads(rendered["output"])
    assert payload["success"] and payload["entries"] == ["x"]


def test_memory_action_failure_still_yields_returncode_zero(manager):
    """Capacity / no-match failures stay returncode=0 — the JSON encodes the failure."""
    manager.builtin.add("only entry")
    actions = [{"tool_name": "memory", "args": {"action": "remove", "old_text": "missing"}, "tool_call_id": "t"}]
    agent, _, _ = _build(manager, [_assistant_msg(actions), _exit_msg()])
    agent.run(task="t")
    rendered = json.loads(agent.messages[3]["content"])
    assert rendered["returncode"] == 0 and rendered["exception_info"] == ""
    payload = json.loads(rendered["output"])
    assert payload["success"] is False and "no entry" in payload["error"].lower()


def test_session_lifecycle_initialize_and_on_session_end_dispatched(manager):
    rec = RecordingProvider()
    manager.register(rec)
    agent, _, _ = _build(manager, [_exit_msg()])
    agent.run(task="t", session_id="instance-42")
    assert rec.calls[0] == ("initialize", "instance-42")
    assert rec.calls[-1][0] == "on_session_end"


def test_on_session_end_fires_even_when_run_raises(manager):
    rec = RecordingProvider()
    manager.register(rec)
    agent, _, _ = _build(manager, [])  # empty responses → IndexError on first query
    with pytest.raises(IndexError):
        agent.run(task="t", session_id="boom")
    assert any(c[0] == "on_session_end" for c in rec.calls)


def test_provider_tool_schemas_pushed_to_model_extra_tools(manager):
    """Without this wiring the model never sees memory tools (P0-1)."""
    schema = {
        "type": "function",
        "function": {"name": "rec_search", "description": "x", "parameters": {"type": "object", "properties": {}}},
    }
    manager.register(RecordingProvider(schemas=[schema]))
    agent, model, _ = _build(manager, [_exit_msg()])
    # Built-in `memory` + `session_search` (default-enabled) + the registered provider tool reach the model.
    names = {t["function"]["name"] for t in model.extra_tools}
    assert names == {"memory", "session_search", "rec_search"}


def test_sync_turn_called_with_previous_user_and_assistant_text(manager):
    """sync_turn pairs the *previous* user/observation with the just-emitted assistant text."""
    rec = RecordingProvider()
    manager.register(rec)
    actions = [{"tool_name": "memory", "args": {"action": "add", "content": "fact"}, "tool_call_id": "t"}]
    agent, _, _ = _build(manager, [_assistant_msg(actions), _exit_msg()])
    agent.run(task="my-task", session_id="s")
    sync_calls = [c for c in rec.calls if c[0] == "sync_turn"]
    # First sync: instance template (the only prior user message) → first assistant.
    assert len(sync_calls) >= 1
    user_text, assistant_text = sync_calls[0][1], sync_calls[0][2]
    assert "my-task" in user_text
    assert "memory" in assistant_text  # tool-call summary present


def test_sync_turn_uses_response_api_function_call_output_as_previous_user(manager):
    rec = RecordingProvider()
    manager.register(rec)
    responses = [
        _assistant_msg([{"tool_name": "bash", "args": {"command": "pytest"}, "command": "pytest", "tool_call_id": "b1"}]),
        _assistant_msg([]),
        _exit_msg(),
    ]
    model, env = ResponseStyleFakeModel(responses), FakeEnv()
    MemoryAgent(
        model,
        env,
        manager=manager,
        system_template=SYSTEM_TEMPLATE,
        instance_template=INSTANCE_TEMPLATE,
        cost_limit=0,
        step_limit=0,
    ).run(task="run tests", session_id="s")

    sync_calls = [c for c in rec.calls if c[0] == "sync_turn"]
    assert "ran:pytest" in sync_calls[1][1]


def test_provider_tool_call_routed_via_manager(manager):
    """An assistant emitting a provider tool call lands on `provider.handle_tool_call`, not env."""
    rec = RecordingProvider(schemas=[{
        "type": "function",
        "function": {"name": "rec_search", "description": "x", "parameters": {"type": "object", "properties": {}}},
    }])
    manager.register(rec)
    actions = [{"tool_name": "rec_search", "args": {"query": "q"}, "tool_call_id": "c1"}]
    agent, _, env = _build(manager, [_assistant_msg(actions), _exit_msg()])
    agent.run(task="t", session_id="s")
    assert env.calls == []
    assert ("handle_tool_call", "rec_search", {"query": "q"}) in rec.calls


def test_initial_context_is_prefixed_to_first_user_message(manager):
    rec = RecordingProvider(initial_context="<memory-context>\nremember pytest flags\n</memory-context>")
    manager.register(rec)
    agent, model, _ = _build(manager, [_exit_msg()])
    agent.run(task="fix the failing tests", session_id="s")
    user_message = next(m for m in model.seen_messages[0] if m.get("role") == "user")
    assert "remember pytest flags" in user_message["content"]
    assert "fix the failing tests" in user_message["content"]
    assert ("initial_context", "fix the failing tests") in rec.calls
