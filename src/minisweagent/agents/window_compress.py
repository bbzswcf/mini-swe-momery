"""Threshold-triggered context compression for chain-window agents.

Drives a single ``model.query`` call against the *same* main model used by the
agent, asking it to summarise everything between the system prompt and the
currently-active task into one compact replacement user message. The agent
itself decides *when* to compress (see ``chain_window.ChainWindowAgent``);
this module just owns the prompt + extraction logic.

Compression unit: ``messages[1:current_task_anchor]`` — i.e. all completed
tasks within the current chain. The system prompt and the in-progress task
are never touched.

Inspired by recent agent /compact prompts (Claude Code, Cursor, OpenHands)
but specialised for SWE-bench-Pro style repo-bound chain tasks: every entry
in the chain edits the same repository, so the summary's job is to carry
forward repo-level engineering knowledge — build/test commands that work,
files / subsystems already touched, gotchas discovered, fix patterns reused
across tasks — rather than the verbatim conversation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("agent.window_compress")

_DEFAULT_SUMMARY_TEMPLATE = (
    "You are compressing the earlier portion of a coding-agent conversation "
    "because the shared context window is approaching its limit. The agent is "
    "solving a *chain* of SWE-bench-Pro tasks against the SAME git repository, "
    "back-to-back in one persistent context. You will be replacing the "
    "messages BETWEEN the system prompt and the currently-active task with a "
    "single compact summary. The system prompt and the in-progress task are "
    "preserved verbatim and stay outside your summary.\n\n"
    "Write a structured, information-dense summary that lets the agent keep "
    "solving more tasks in this repo without re-discovering what it already "
    "learned. Skip anything that is specific to the *current* task — that is "
    "still visible in the conversation. Prefer concrete file paths, function "
    "names, command lines and code idioms over prose; do NOT quote whole "
    "files or full command outputs.\n\n"
    "Hard rules:\n"
    "- Do NOT call any tools; reply with the summary text only.\n"
    "- Stay under roughly {char_budget} characters.\n"
    "- Use the section layout below exactly. Omit a section's bullets only "
    "  if you genuinely have nothing concrete to record there.\n\n"
    "# Compressed chain history\n\n"
    "## Completed tasks (in order)\n"
    "For each finished task, one entry like:\n"
    "- `<instance_id>` — problem in <=20 words. Fix: <core idea + key files>. "
    "Result: <Submitted / failed / skipped>.\n\n"
    "## Repository facts learned\n"
    "Durable engineering knowledge about THIS repo. Examples: working "
    "build/test/lint commands; how to run a single test; entrypoints; "
    "subsystem layout; coding/style conventions enforced by tests; flaky "
    "tests; container quirks (paths, env vars, pre-installed tooling).\n\n"
    "## Fix patterns that worked\n"
    "Concrete code idioms reused across past tasks (with file paths or "
    "function names) that future tasks may want to reuse.\n\n"
    "## Failed approaches to avoid\n"
    "Things the agent already tried and that did NOT work — one bullet each, "
    "including the symptom so the agent recognises the dead end.\n\n"
    "## Files/modules already modified in this chain\n"
    "Path -> one-line description of what changed and why. Future tasks may "
    "need to extend these edits or stay consistent with them.\n\n"
    "## Container/runtime state\n"
    "Side-effects the agent created and that persist across tasks (created "
    "files, installed deps, env vars exported, git index state). Note: each "
    "new task may run in a *fresh* container; only mention state that the "
    "next task can actually still observe.\n\n"
    "<earlier_messages>\n{trace}\n</earlier_messages>"
)


@dataclass
class CompressionConfig:
    enabled: bool = True
    model_window: int = 200000
    threshold: float = 0.6
    char_budget: int = 6000
    summary_template: str = _DEFAULT_SUMMARY_TEMPLATE
    max_output_tokens: int = 4096
    trace_max_chars: int = 400000
    """Hard cap on the raw trace we feed the compressor so we don't ourselves
    blow past the model window while asking for a summary."""

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold!r}")
        if self.model_window <= 0:
            raise ValueError("model_window must be positive")
        if self.char_budget <= 0:
            raise ValueError("char_budget must be positive")

    @property
    def token_trigger(self) -> int:
        return int(self.model_window * self.threshold)


def extract_response_text(message: dict) -> str:
    """Pull plain text out of a litellm-response-API message dict."""
    parts: list[str] = []
    for item in message.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for chunk in item.get("content", []) or []:
            if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                parts.append(chunk["text"])
    return "\n".join(p for p in parts if p).strip()


def render_trace(messages: list[dict]) -> str:
    """Flatten messages into a tagged transcript suitable for the prompt."""
    from minisweagent.memory.session_store import extract_message_text

    lines: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("type") or "message"
        if msg.get("object") == "response":
            for item in msg.get("output", []) or []:
                if not isinstance(item, dict):
                    continue
                kind = item.get("type")
                if kind == "function_call":
                    args = item.get("arguments") or ""
                    lines.append(f"[assistant tool_call:{item.get('name', '')}] {args}")
                elif kind == "message":
                    text = "".join(
                        c.get("text", "") for c in (item.get("content") or []) if isinstance(c, dict)
                    )
                    if text.strip():
                        lines.append(f"[assistant] {text.strip()}")
                elif kind == "reasoning":
                    continue  # encrypted, opaque
            continue
        text = extract_message_text(msg).strip()
        if text:
            lines.append(f"[{role}] {text}")
    return "\n".join(lines)


def compress_history(
    model: Any,
    middle_messages: list[dict],
    *,
    config: CompressionConfig,
) -> str | None:
    """Ask the main model to summarise ``middle_messages``. Returns ``None`` on
    failure so the caller can fall back to keeping the raw messages.

    Goes through ``model.query_no_tools`` rather than ``model.query`` so the
    request ships *without* a ``tools`` array or ``tool_choice`` — modelhub's
    Responses gateway treats ``tool_choice="none"`` alongside a tools array as
    ``UnsupportedParamsError``, which is in the model's ``abort_exceptions``
    list and silently re-raises out of the retry loop with no log line.
    Compression is pure text→text so dropping tools also saves tokens."""
    trace = render_trace(middle_messages)
    if not trace.strip():
        return None
    if len(trace) > config.trace_max_chars:
        head = trace[: config.trace_max_chars // 2]
        tail = trace[-config.trace_max_chars // 2 :]
        trace = f"{head}\n…(middle of trace truncated, {len(trace) - config.trace_max_chars} chars elided)…\n{tail}"
    prompt = config.summary_template.format(char_budget=config.char_budget, trace=trace)
    query = getattr(model, "query_no_tools", None)
    if query is None:
        logger.warning(
            "chain-window compression skipped: %s has no `query_no_tools` method", type(model).__name__
        )
        return None
    try:
        response = query(
            [{"role": "user", "content": prompt}], max_output_tokens=config.max_output_tokens
        )
    except Exception as exc:
        logger.warning(
            "chain-window compression LLM call failed: %s: %s", type(exc).__name__, exc, exc_info=True
        )
        return None
    text = extract_response_text(response)
    if not text:
        logger.warning("chain-window compression returned empty text; keeping raw history")
        return None
    return text
