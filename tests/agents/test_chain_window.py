"""Tests for the chain-window agent + threshold-triggered compression baseline."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml

from minisweagent.agents.chain_window import ChainWindowAgent
from minisweagent.agents.window_compress import (
    CompressionConfig,
    compress_history,
    extract_response_text,
    render_trace,
)
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import (
    DeterministicResponseAPIToolcallModel,
    make_response_api_output,
)


@pytest.fixture
def toolcall_config() -> dict:
    return yaml.safe_load(Path("src/minisweagent/config/mini.yaml").read_text())["agent"]


def _step_output(content: str, command: str, *, usage_tokens: int) -> dict:
    out = make_response_api_output(
        content,
        [{"command": command, "tool_call_id": f"tc_{hash(command) & 0xffff:04x}"}],
    )
    out["usage"] = {"input_tokens": usage_tokens}
    return out


class _ScriptedModel(DeterministicResponseAPIToolcallModel):
    """Like the deterministic response-API model but tags every reply with
    ``usage.input_tokens`` (read by the chain-window agent for the threshold
    check) and answers ``query_no_tools`` (used by the compression helper)
    with a canned summary so the main step queue stays untouched."""

    def __init__(
        self,
        *,
        step_outputs: list[dict],
        compression_text: str = "COMPRESSED_HISTORY_SUMMARY",
        compression_error: Exception | None = None,
    ):
        super().__init__(outputs=step_outputs)
        self._compression_text = compression_text
        self._compression_error = compression_error
        self.compression_calls: list[dict] = []

    def query_no_tools(self, messages: list[dict], **kwargs) -> dict:
        self.compression_calls.append({"messages": messages, "kwargs": kwargs})
        if self._compression_error is not None:
            raise self._compression_error
        return {
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": self._compression_text}],
                }
            ],
            "usage": {"input_tokens": 50},
        }


def _make_agent(model, env, toolcall_config, *, compression=None) -> ChainWindowAgent:
    return ChainWindowAgent(
        model, env, compression=compression, **{**toolcall_config, "cost_limit": 100.0, "step_limit": 50}
    )


# --- unit tests for window_compress helpers ---


def test_compression_config_rejects_bad_threshold():
    with pytest.raises(ValueError):
        CompressionConfig(threshold=0)
    with pytest.raises(ValueError):
        CompressionConfig(threshold=1.5)
    with pytest.raises(ValueError):
        CompressionConfig(model_window=0)


def test_compression_config_token_trigger_uses_threshold():
    assert CompressionConfig(model_window=100_000, threshold=0.6).token_trigger == 60_000


def test_extract_response_text_pulls_assistant_message():
    response = {
        "output": [
            {"type": "reasoning", "content": [{"type": "summary_text", "text": "ignored"}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello "}, {"type": "output_text", "text": "world"}],
            },
        ]
    }
    assert extract_response_text(response) == "hello \nworld"


def test_render_trace_includes_user_and_tool_messages():
    msgs = [
        {"role": "user", "content": [{"type": "input_text", "text": "fix foo"}]},
        {
            "object": "response",
            "output": [
                {"type": "function_call", "name": "bash", "arguments": '{"command": "ls"}'},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "running ls"}]},
            ],
        },
        {"type": "function_call_output", "call_id": "x", "output": "<returncode>0</returncode>"},
    ]
    trace = render_trace(msgs)
    assert "fix foo" in trace
    assert "[assistant tool_call:bash]" in trace
    assert "running ls" in trace
    assert "<returncode>0</returncode>" in trace


def test_compress_history_routes_through_query_no_tools(toolcall_config):
    model = _ScriptedModel(step_outputs=[], compression_text="MY_SUMMARY")
    result = compress_history(
        model,
        [{"role": "user", "content": [{"type": "input_text", "text": "old task"}]}],
        config=CompressionConfig(max_output_tokens=2048),
    )
    assert result == "MY_SUMMARY"
    assert len(model.compression_calls) == 1
    assert model.compression_calls[0]["kwargs"]["max_output_tokens"] == 2048


def test_compress_history_returns_none_when_model_query_no_tools_raises():
    model = _ScriptedModel(step_outputs=[], compression_error=RuntimeError("boom"))
    out = compress_history(
        model,
        [{"role": "user", "content": [{"type": "input_text", "text": "anything"}]}],
        config=CompressionConfig(),
    )
    assert out is None
    assert len(model.compression_calls) == 1


def test_compress_history_returns_none_when_model_has_no_query_no_tools():
    class _NoSuchAPI:
        pass

    out = compress_history(
        _NoSuchAPI(),
        [{"role": "user", "content": [{"type": "input_text", "text": "anything"}]}],
        config=CompressionConfig(),
    )
    assert out is None


# --- ChainWindowAgent integration ---


def _submit_action() -> dict:
    return {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo done"}


def test_chain_window_runs_two_tasks_in_one_window_without_compression(toolcall_config):
    model = _ScriptedModel(
        step_outputs=[
            _step_output("solving t1", "echo t1", usage_tokens=10),
            _step_output("submit t1", _submit_action()["command"], usage_tokens=20),
            _step_output("solving t2", "echo t2", usage_tokens=30),
            _step_output("submit t2", _submit_action()["command"], usage_tokens=40),
        ]
    )
    agent = _make_agent(
        model,
        LocalEnvironment(),
        toolcall_config,
        compression={"enabled": True, "model_window": 100_000, "threshold": 0.9},
    )
    results = agent.process_chain(
        [{"instance_id": "t1"}, {"instance_id": "t2"}],
        render_task=lambda inst: f"work on {inst['instance_id']}",
    )
    assert [r["instance_id"] for r in results] == ["t1", "t2"]
    assert [r["info"]["exit_status"] for r in results] == ["Submitted", "Submitted"]
    assert model.compression_calls == []
    # one shared system prompt
    assert sum(1 for m in agent.messages if m.get("role") == "system") == 1
    # two distinct user task messages
    assert sum(
        1
        for m in agent.messages
        if m.get("role") == "user" and "work on t" in (m.get("content", [{}])[0].get("text", "") if isinstance(m.get("content"), list) else "")
    ) == 2


def test_chain_window_compresses_between_tasks_when_threshold_exceeded(toolcall_config):
    model = _ScriptedModel(
        step_outputs=[
            _step_output("solving t1", "echo t1", usage_tokens=10),
            _step_output("submit t1", _submit_action()["command"], usage_tokens=80_000),
            _step_output("solving t2 after compression", "echo t2", usage_tokens=5_000),
            _step_output("submit t2", _submit_action()["command"], usage_tokens=10_000),
        ]
    )
    agent = _make_agent(
        model,
        LocalEnvironment(),
        toolcall_config,
        compression={"enabled": True, "model_window": 100_000, "threshold": 0.5},
    )
    results = agent.process_chain(
        [{"instance_id": "task_one"}, {"instance_id": "task_two"}],
        render_task=lambda inst: f"work on {inst['instance_id']}",
    )
    assert all(r["info"]["exit_status"] == "Submitted" for r in results)
    assert len(model.compression_calls) == 1
    # After compression, task 1's content is collapsed into a single user message
    # with a <compressed_history> wrapper.
    compressed = [
        m
        for m in agent.messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and m["content"]
        and "<compressed_history>" in m["content"][0].get("text", "")
    ]
    assert len(compressed) == 1
    assert agent._summary_present is True
    assert agent._compression_log[0]["reason"] == "pre_task"
    assert agent._compression_log[0]["status"] == "ok"


def test_chain_window_does_not_compress_first_task(toolcall_config):
    """No compression should fire on the very first task even if threshold is tripped,
    because there is nothing to compress between system prompt and current task."""
    model = _ScriptedModel(
        step_outputs=[
            _step_output("solving t1", "echo t1", usage_tokens=99_000),
            _step_output("submit t1", _submit_action()["command"], usage_tokens=99_000),
        ]
    )
    agent = _make_agent(
        model,
        LocalEnvironment(),
        toolcall_config,
        compression={"enabled": True, "model_window": 100_000, "threshold": 0.5},
    )
    agent.process_chain(
        [{"instance_id": "solo"}],
        render_task=lambda inst: f"work on {inst['instance_id']}",
    )
    assert model.compression_calls == []
    assert agent._summary_present is False


def test_chain_window_compresses_mid_task_when_step_trips_threshold(toolcall_config):
    """Threshold trip during task 2 (not at the boundary) still fires compression of
    task 1, leaving task 2 in progress intact."""
    model = _ScriptedModel(
        step_outputs=[
            _step_output("solving t1", "echo t1", usage_tokens=5_000),
            _step_output("submit t1", _submit_action()["command"], usage_tokens=10_000),
            _step_output("solving t2 (still under threshold)", "echo t2-a", usage_tokens=20_000),
            _step_output("solving t2 (trip threshold)", "echo t2-b", usage_tokens=90_000),
            _step_output("submit t2", _submit_action()["command"], usage_tokens=15_000),
        ]
    )
    agent = _make_agent(
        model,
        LocalEnvironment(),
        toolcall_config,
        compression={"enabled": True, "model_window": 100_000, "threshold": 0.8},
    )
    agent.process_chain(
        [{"instance_id": "t1"}, {"instance_id": "t2"}],
        render_task=lambda inst: f"work on {inst['instance_id']}",
    )
    assert len(model.compression_calls) == 1
    # Compression should be tagged post_step (mid-task) not pre_task
    assert agent._compression_log[0]["reason"] == "post_step"
    assert agent._compression_log[0]["status"] == "ok"
    # Current task (t2) prompt is preserved verbatim
    user_msgs = [m for m in agent.messages if m.get("role") == "user"]
    texts = [m["content"][0].get("text", "") if isinstance(m.get("content"), list) and m["content"] else "" for m in user_msgs]
    assert any("work on t2" in t for t in texts)


def test_chain_window_serialize_includes_chain_window_block(toolcall_config):
    model = _ScriptedModel(
        step_outputs=[
            _step_output("submit t1", _submit_action()["command"], usage_tokens=10),
        ]
    )
    agent = _make_agent(
        model,
        LocalEnvironment(),
        toolcall_config,
        compression={"enabled": True, "model_window": 100_000, "threshold": 0.9},
    )
    agent.process_chain(
        [{"instance_id": "solo"}],
        render_task=lambda inst: f"work on {inst['instance_id']}",
    )
    serialized = agent.serialize()
    cw = serialized["info"]["chain_window"]
    assert cw["completed_instance_ids"] == ["solo"]
    assert cw["compression"]["model_window"] == 100_000
    assert cw["last_input_tokens"] == 10


def test_agent_class_registry_resolves_chain_window():
    from minisweagent.agents import get_agent_class

    assert get_agent_class("chain_window") is ChainWindowAgent


def test_compression_failure_does_not_disable_future_attempts(toolcall_config):
    """Cooldown was removed: ``query_no_tools`` now has its own retry loop for
    transient errors (RateLimit / network), so a failure that surfaces here is
    either deterministic (re-firing logs the same failure — fine) or a
    retry-exhausted transient that may succeed next step. Either way we must
    NOT permanently disable compression — that previously killed long chains
    on a single TPM hit (real-world Modelhub -2004 storm) and is the original
    cooldown's biggest downside."""
    model = _ScriptedModel(
        step_outputs=[
            _step_output("solving t1", "echo t1", usage_tokens=90_000),
            _step_output("submit t1", _submit_action()["command"], usage_tokens=95_000),
            _step_output("solving t2 step a", "echo t2-a", usage_tokens=99_000),
            _step_output("solving t2 step b", "echo t2-b", usage_tokens=99_500),
            _step_output("submit t2", _submit_action()["command"], usage_tokens=99_700),
        ],
        compression_error=RuntimeError("modelhub refused"),
    )
    agent = _make_agent(
        model, LocalEnvironment(), toolcall_config,
        compression={"enabled": True, "model_window": 100_000, "threshold": 0.5},
    )
    agent.process_chain(
        [{"instance_id": "t1"}, {"instance_id": "t2"}],
        render_task=lambda inst: f"work on {inst['instance_id']}",
    )
    # Each threshold trip re-fires; compression stays enabled across failures.
    assert len(model.compression_calls) >= 2
    assert all(entry["status"] == "failed" for entry in agent._compression_log)
    assert agent.compression.enabled is True


def test_step_limit_is_per_task_not_per_chain(toolcall_config):
    """``step_limit`` must reset for every task in the chain. Otherwise long
    chains' later tasks all trip ``LimitsExceeded`` immediately because the
    agent instance (and ``self.n_calls``) is reused across tasks."""
    model = _ScriptedModel(
        step_outputs=[
            _step_output("solving t1 a", "echo t1a", usage_tokens=10),
            _step_output("submit t1", _submit_action()["command"], usage_tokens=10),
            _step_output("solving t2 a", "echo t2a", usage_tokens=10),
            _step_output("submit t2", _submit_action()["command"], usage_tokens=10),
            _step_output("solving t3 a", "echo t3a", usage_tokens=10),
            _step_output("submit t3", _submit_action()["command"], usage_tokens=10),
        ]
    )
    agent = ChainWindowAgent(
        model, LocalEnvironment(),
        compression={"enabled": False, "model_window": 100_000, "threshold": 0.9},
        **{**toolcall_config, "cost_limit": 100.0, "step_limit": 2},
    )
    results = agent.process_chain(
        [{"instance_id": f"t{i}"} for i in (1, 2, 3)],
        render_task=lambda inst: f"work on {inst['instance_id']}",
    )
    assert [r["info"]["exit_status"] for r in results] == ["Submitted", "Submitted", "Submitted"]
    assert agent.n_calls == 6


def test_seal_completed_task_pads_unanswered_function_call(toolcall_config):
    """After a `Submitted` step the last `function_call` has no matching
    `function_call_output`; without padding, the next task's first model.query
    would hit modelhub's -4003 "No tool output found for function call"."""
    model = _ScriptedModel(
        step_outputs=[_step_output("submit t1", _submit_action()["command"], usage_tokens=10)]
    )
    agent = _make_agent(
        model, LocalEnvironment(), toolcall_config,
        compression={"enabled": True, "model_window": 100_000, "threshold": 0.9},
    )
    agent.process_chain(
        [{"instance_id": "solo"}],
        render_task=lambda inst: f"work on {inst['instance_id']}",
    )
    last = agent.messages[-1]
    assert last.get("type") == "function_call_output"
    assert last.get("call_id")
    # Every function_call in the trailing response is paired with an output
    pair_msg = next(m for m in reversed(agent.messages) if isinstance(m, dict) and m.get("object") == "response")
    call_ids = {it["call_id"] for it in pair_msg["output"] if isinstance(it, dict) and it.get("type") == "function_call"}
    answered = {m["call_id"] for m in agent.messages if isinstance(m, dict) and m.get("type") == "function_call_output"}
    assert call_ids.issubset(answered)
