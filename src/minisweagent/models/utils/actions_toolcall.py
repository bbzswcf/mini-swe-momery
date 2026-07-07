"""Parse actions & format observations with toolcalls"""

import json
import time
from collections.abc import Iterable

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


def _format_error(format_error_template: str, error: str) -> FormatError:
    return FormatError(
        {
            "role": "user",
            "content": Template(format_error_template, undefined=StrictUndefined).render(
                error=error, actions=[]
            ),
            "extra": {"interrupt_type": "FormatError"},
        }
    )


def parse_toolcall_actions(
    tool_calls: list,
    *,
    format_error_template: str,
    allowed_tools: Iterable[str] | None = None,
) -> list[dict]:
    """Parse tool calls from the response. Raises FormatError on unknown tool / bad args.

    Each returned action carries ``tool_name`` and parsed ``args`` so callers (e.g.
    ``MemoryAgent``) can route by tool name. ``bash`` actions additionally expose
    ``command`` for backward compatibility with ``env.execute(action)``.

    ``allowed_tools`` defaults to ``{"bash"}`` so legacy callers see no behavior
    change. Pass an extended set (``{"bash", "memory", "mem0_search", ...}``) to
    let the model emit memory-side tool calls without tripping FormatError.
    """
    if not tool_calls:
        raise _format_error(
            format_error_template,
            "No tool calls found in the response. Every response MUST include at least one tool call.",
        )
    allowed = set(allowed_tools) if allowed_tools is not None else {"bash"}
    actions = []
    for tool_call in tool_calls:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            raise _format_error(format_error_template, f"Error parsing tool call arguments: {e}.") from e
        if name not in allowed:
            raise _format_error(format_error_template, f"Unknown tool '{name}'.")
        if not isinstance(args, dict):
            raise _format_error(format_error_template, f"Tool '{name}' arguments must be a JSON object.")
        if name == "bash" and "command" not in args:
            raise _format_error(format_error_template, "Missing 'command' argument in bash tool call.")
        action = {"tool_name": name, "args": args, "tool_call_id": tool_call.id}
        if name == "bash":
            action["command"] = args["command"]
        actions.append(action)
    return actions


def format_toolcall_observation_messages(
    *,
    actions: list[dict],
    outputs: list[dict],
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Format execution outputs into tool result messages."""
    not_executed = {"output": "", "returncode": -1, "exception_info": "action was not executed"}
    padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
    results = []
    for action, output in zip(actions, padded_outputs):
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg = {
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if "tool_call_id" in action:
            msg["tool_call_id"] = action["tool_call_id"]
            msg["role"] = "tool"
        else:
            msg["role"] = "user"  # human issued commands
        if multimodal_regex:
            msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results
