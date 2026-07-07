from unittest.mock import MagicMock

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)


def _toolcall(name: str, arguments: str, call_id: str = "call_1") -> MagicMock:
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    tc.id = call_id
    return tc


class TestParseToolcallActions:
    def test_empty_tool_calls_raises_format_error(self):
        with pytest.raises(FormatError) as exc_info:
            parse_toolcall_actions([], format_error_template="{{ error }}")
        assert "No tool calls found" in exc_info.value.messages[0]["content"]

    def test_none_tool_calls_raises_format_error(self):
        with pytest.raises(FormatError) as exc_info:
            parse_toolcall_actions(None, format_error_template="{{ error }}")
        assert "No tool calls found" in exc_info.value.messages[0]["content"]

    def test_valid_bash_tool_call_emits_command_args_and_tool_name(self):
        result = parse_toolcall_actions(
            [_toolcall("bash", '{"command": "echo hello"}', "call_123")],
            format_error_template="{{ error }}",
        )
        assert result == [
            {
                "tool_name": "bash",
                "args": {"command": "echo hello"},
                "tool_call_id": "call_123",
                "command": "echo hello",
            }
        ]

    def test_multiple_valid_bash_tool_calls(self):
        calls = [_toolcall("bash", f'{{"command": "cmd{i}"}}', f"call_{i}") for i in range(3)]
        result = parse_toolcall_actions(calls, format_error_template="{{ error }}")
        assert [a["command"] for a in result] == ["cmd0", "cmd1", "cmd2"]
        assert [a["tool_call_id"] for a in result] == ["call_0", "call_1", "call_2"]
        assert all(a["tool_name"] == "bash" for a in result)

    def test_unknown_tool_raises_format_error_when_not_in_allowlist(self):
        with pytest.raises(FormatError) as exc_info:
            parse_toolcall_actions(
                [_toolcall("unknown_tool", '{"command": "test"}')],
                format_error_template="{{ error }}",
            )
        assert "Unknown tool 'unknown_tool'" in exc_info.value.messages[0]["content"]

    def test_invalid_json_raises_format_error(self):
        with pytest.raises(FormatError) as exc_info:
            parse_toolcall_actions(
                [_toolcall("bash", "not valid json")],
                format_error_template="{{ error }}",
            )
        assert "Error parsing tool call arguments" in exc_info.value.messages[0]["content"]

    def test_missing_command_raises_format_error_for_bash(self):
        with pytest.raises(FormatError) as exc_info:
            parse_toolcall_actions(
                [_toolcall("bash", '{"other_arg": "value"}')],
                format_error_template="{{ error }}",
            )
        assert "Missing 'command'" in exc_info.value.messages[0]["content"]

    def test_allowlist_lets_memory_tool_through_without_command_field(self):
        """Memory tools take action/content, not command — must pass parser when allowed."""
        result = parse_toolcall_actions(
            [_toolcall("memory", '{"action": "add", "content": "fact"}', "call_M")],
            format_error_template="{{ error }}",
            allowed_tools={"bash", "memory"},
        )
        assert result == [
            {
                "tool_name": "memory",
                "args": {"action": "add", "content": "fact"},
                "tool_call_id": "call_M",
            }
        ]
        # Bash key absent for non-bash tools.
        assert "command" not in result[0]

    def test_allowlist_routes_provider_tool_alongside_bash(self):
        calls = [
            _toolcall("bash", '{"command": "ls"}', "b1"),
            _toolcall("hindsight_recall", '{"query": "x"}', "h1"),
        ]
        result = parse_toolcall_actions(
            calls,
            format_error_template="{{ error }}",
            allowed_tools={"bash", "hindsight_recall"},
        )
        assert [a["tool_name"] for a in result] == ["bash", "hindsight_recall"]
        assert result[0]["command"] == "ls"
        assert result[1]["args"] == {"query": "x"}

    def test_args_must_be_object(self):
        with pytest.raises(FormatError) as exc_info:
            parse_toolcall_actions(
                [_toolcall("bash", "[1,2,3]")],
                format_error_template="{{ error }}",
            )
        assert "JSON object" in exc_info.value.messages[0]["content"]


class TestFormatToolcallObservationMessages:
    def test_basic_formatting(self):
        actions = [{"tool_name": "bash", "command": "echo test", "tool_call_id": "call_1"}]
        outputs = [{"output": "test output", "returncode": 0}]
        result = format_toolcall_observation_messages(
            actions=actions, outputs=outputs, observation_template="{{ output.output }}"
        )
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "test output"
        assert result[0]["extra"]["returncode"] == 0

    def test_multiple_outputs_keep_tool_call_ids_aligned(self):
        actions = [
            {"tool_name": "bash", "command": "cmd1", "tool_call_id": "call_1"},
            {"tool_name": "bash", "command": "cmd2", "tool_call_id": "call_2"},
        ]
        outputs = [{"output": "out1", "returncode": 0}, {"output": "out2", "returncode": 1}]
        result = format_toolcall_observation_messages(
            actions=actions, outputs=outputs, observation_template="{{ output.output }}"
        )
        assert [r["tool_call_id"] for r in result] == ["call_1", "call_2"]
        assert [r["content"] for r in result] == ["out1", "out2"]

    def test_with_template_vars(self):
        actions = [{"tool_name": "bash", "command": "test", "tool_call_id": "call_1"}]
        outputs = [{"output": "result", "returncode": 0}]
        result = format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template="{{ output.output }} - {{ custom_var }}",
            template_vars={"custom_var": "extra_info"},
        )
        assert result[0]["content"] == "result - extra_info"

    def test_exception_info_in_extra(self):
        actions = [{"tool_name": "bash", "command": "test", "tool_call_id": "call_1"}]
        outputs = [{"output": "", "returncode": 1, "exception_info": "Error occurred", "extra": {"detail": "more"}}]
        result = format_toolcall_observation_messages(
            actions=actions, outputs=outputs, observation_template="{{ output.output }}"
        )
        assert result[0]["extra"]["exception_info"] == "Error occurred"
        assert result[0]["extra"]["detail"] == "more"


class TestBashTool:
    def test_bash_tool_structure(self):
        assert BASH_TOOL["type"] == "function"
        assert BASH_TOOL["function"]["name"] == "bash"
        assert "command" in BASH_TOOL["function"]["parameters"]["properties"]
        assert "command" in BASH_TOOL["function"]["parameters"]["required"]
