"""Unit tests for the memory-only consolidation pass.

Verifies the contract `MemoryManager.maybe_consolidate` /
`MemoryManager.on_session_end` rely on: only `memory` tool calls are honored,
extras are dropped, model errors are swallowed, and `max_actions` is enforced.
"""

from __future__ import annotations

from minisweagent.memory import BuiltinMemory, BuiltinMemoryConfig, consolidate_memory


class _FakeModel:
    def __init__(self, response: dict):
        self.response = response
        self.seen: list[list[dict]] = []

    def query(self, messages):
        self.seen.append(messages)
        return self.response


def _builtin(tmp_path, **kw) -> BuiltinMemory:
    return BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=kw.get("char_limit", 400)))


def test_only_memory_tool_calls_are_applied_others_skipped(tmp_path):
    builtin = _builtin(tmp_path)
    model = _FakeModel(
        {
            "extra": {
                "actions": [
                    {"tool_name": "memory", "args": {"action": "add", "content": "tests use pytest -xvs"}},
                    {"tool_name": "bash", "args": {"command": "ls"}},
                    {"tool_name": "session_search", "args": {"query": "x"}},
                ]
            }
        }
    )
    res = consolidate_memory(model, builtin, [{"role": "user", "content": "context"}])
    assert res == {"applied": 1, "skipped": 2}
    assert builtin.load() == ["tests use pytest -xvs"]


def test_max_actions_caps_applied_writes(tmp_path):
    """Even if the model returns 10 valid memory calls, only `max_actions` are honored."""
    builtin = _builtin(tmp_path, char_limit=2000)
    actions = [{"tool_name": "memory", "args": {"action": "add", "content": f"fact {i}"}} for i in range(10)]
    model = _FakeModel({"extra": {"actions": actions}})
    res = consolidate_memory(model, builtin, [{"role": "user", "content": "hi"}], max_actions=2)
    assert res["applied"] == 2
    assert builtin.load() == ["fact 0", "fact 1"]


def test_consolidation_tries_full_trace_before_truncating(tmp_path):
    builtin = _builtin(tmp_path)
    model = _FakeModel({"extra": {"actions": []}})
    consolidate_memory(
        model,
        builtin,
        [
            {"role": "user", "content": "early important task context"},
            {"role": "assistant", "content": "late conclusion"},
        ],
        summary_max_chars=20,
    )
    prompt = model.seen[0][0]["content"]
    assert "early important task context" in prompt
    assert "late conclusion" in prompt
    assert "earlier trace truncated" not in prompt


def test_consolidation_falls_back_to_tail_trace_when_full_trace_query_fails(tmp_path):
    class _FallbackModel:
        def __init__(self):
            self.seen: list[list[dict]] = []

        def query(self, messages):
            self.seen.append(messages)
            if len(self.seen) == 1:
                raise RuntimeError("context length exceeded")
            return {
                "extra": {
                    "actions": [
                        {"tool_name": "memory", "args": {"action": "add", "content": "fallback fact"}},
                    ]
                }
            }

    builtin = _builtin(tmp_path)
    model = _FallbackModel()
    res = consolidate_memory(
        model,
        builtin,
        [
            {"role": "user", "content": "early important task context"},
            {"role": "assistant", "content": "late conclusion"},
        ],
        summary_max_chars=40,
    )
    assert res == {"applied": 1, "skipped": 0}
    assert builtin.load() == ["fallback fact"]
    assert "early important task context" in model.seen[0][0]["content"]
    assert "…(earlier trace truncated)" in model.seen[1][0]["content"]
    assert "late conclusion" in model.seen[1][0]["content"]


def test_consolidation_truncates_before_query_when_trace_is_too_large(tmp_path):
    builtin = _builtin(tmp_path)
    model = _FakeModel({"extra": {"actions": []}})
    consolidate_memory(
        model,
        builtin,
        [
            {"role": "user", "content": "early important task context"},
            {"role": "assistant", "content": "x" * 80},
            {"role": "assistant", "content": "late conclusion"},
        ],
        summary_max_chars=40,
        full_trace_max_chars=60,
    )
    prompt = model.seen[0][0]["content"]
    assert len(model.seen) == 1
    assert "early important task context" not in prompt
    assert "…(earlier trace truncated)" in prompt
    assert "late conclusion" in prompt


def test_replace_and_remove_actions_route_correctly(tmp_path):
    builtin = _builtin(tmp_path)
    builtin.add("stale fact about ruff")
    builtin.add("fact to drop")
    model = _FakeModel(
        {
            "extra": {
                "actions": [
                    {"tool_name": "memory", "args": {"action": "replace", "old_text": "stale", "content": "fresh fact"}},
                    {"tool_name": "memory", "args": {"action": "remove", "old_text": "to drop"}},
                ]
            }
        }
    )
    res = consolidate_memory(model, builtin, [{"role": "user", "content": "x"}])
    assert res["applied"] == 2
    assert builtin.load() == ["fresh fact"]


def test_swallows_model_errors_and_reports_them(tmp_path):
    builtin = _builtin(tmp_path)

    class _BadModel:
        def query(self, messages):
            raise RuntimeError("api down")

    res = consolidate_memory(_BadModel(), builtin, [{"role": "user", "content": "x"}])
    assert res["applied"] == 0
    assert "api down" in res["error"]
    assert builtin.load() == []


def test_empty_trace_short_circuits_without_calling_model(tmp_path):
    builtin = _builtin(tmp_path)
    model = _FakeModel({"extra": {"actions": []}})
    assert consolidate_memory(model, builtin, []) == {"applied": 0, "skipped": 0, "error": "empty trace"}
    assert model.seen == []


def test_unknown_action_kind_is_skipped_not_raised(tmp_path):
    builtin = _builtin(tmp_path)
    model = _FakeModel(
        {
            "extra": {
                "actions": [
                    {"tool_name": "memory", "args": {"action": "rewrite", "content": "?"}},
                    {"tool_name": "memory", "args": {"action": "add", "content": "valid one"}},
                ]
            }
        }
    )
    res = consolidate_memory(model, builtin, [{"role": "user", "content": "x"}])
    assert res == {"applied": 1, "skipped": 1}
    assert builtin.load() == ["valid one"]
