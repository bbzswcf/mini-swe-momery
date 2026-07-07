"""One-shot memory consolidation turn (digest §5 / "session 结束跑一轮 LLM-only").

Asks the model to update ``MEMORY.md`` based on a finished (or in-progress)
session trace, using *only* the built-in ``memory`` tool. Bypasses the agent
loop — first tries the full trace in one ``model.query`` call, then falls back
to the old tail-truncated trace if that call fails.

Hermes-style flush triggers covered (digest §5, "Memory Flush"):

- ``on_session_end``: invoked from ``MemoryAgent.run()``'s ``finally`` so the
  trial's lessons are persisted before the agent shuts down.
- ``every_n_steps``: invoked from ``MemoryAgent.step()`` only after N model
  calls have elapsed without a successful MEMORY.md write/update. This is the
  mini-swe-agent equivalent of hermes' "background flush before session
  expiration / compression" — we have no compression, but a long trace still
  benefits from a mid-run checkpoint that pushes hard-won facts to disk so the
  next trial sees them even if the current one OOMs / times out.

Failures are swallowed: consolidation is best-effort and must never break the
main agent flow. Non-``memory`` tool calls in the response are ignored (model
sometimes still emits ``bash``/``session_search`` despite the prompt — we only
honor what we asked for).
"""

from __future__ import annotations

import logging

from minisweagent.memory.builtin import BuiltinMemory
from minisweagent.memory.session_store import extract_message_text

logger = logging.getLogger("memory.consolidation")

_PROMPT_TEMPLATE = (
    "You just finished (or paused) an engineering session. Review the trace "
    "below and update MEMORY.md so that *future similar sessions* benefit.\n\n"
    "Hard rules:\n"
    "- ONLY call the `memory` tool (action add | replace | remove). Do NOT "
    "call bash, session_search, or any provider tool — they will be ignored.\n"
    "- Make AT MOST {max_actions} memory tool calls in total.\n"
    "- Add only durable engineering knowledge (build/test commands, repo "
    "conventions, verified gotchas, fix patterns, failed approaches). Skip "
    "the issue text, raw logs/diffs, and trivially re-discoverable facts.\n"
    "- Prefer `replace` over `add` when an existing entry can be tightened "
    "or extended; prefer `remove` over leaving stale entries around.\n"
    "- Stay within the {char_limit}-char MEMORY.md budget; current usage "
    "is shown below.\n\n"
    "<current_memory>\n{snapshot}\n</current_memory>\n\n"
    "<session_trace>\n{trace}\n</session_trace>"
)


def consolidate_memory(
    model,
    builtin: BuiltinMemory,
    messages: list[dict],
    *,
    max_actions: int = 3,
    summary_max_chars: int = 4000,
    full_trace_max_chars: int = 1_000_000,
) -> dict:
    """Run one memory-only LLM turn and apply its `memory` tool calls.

    Returns ``{"applied": int, "skipped": int, "error": str?}`` for caller
    introspection. Never raises.
    """
    trace = _format_messages(messages)
    if not trace.strip():
        return {"applied": 0, "skipped": 0, "error": "empty trace"}
    if len(trace) > full_trace_max_chars:
        logger.warning("Consolidation trace is too large (%s chars); using truncated trace.", len(trace))
        trace = _summarize_messages(messages, max_chars=summary_max_chars)
    snapshot = builtin._render(builtin.load()) or "(MEMORY.md is empty)"
    try:
        response = model.query(
            [
                {
                    "role": "user",
                    "content": _build_prompt(
                        trace,
                        snapshot=snapshot,
                        max_actions=max_actions,
                        char_limit=builtin.config.char_limit,
                    ),
                }
            ]
        )
    except Exception as e:
        logger.warning("Full-trace consolidation LLM call failed; retrying with truncated trace: %s", e)
        fallback_trace = _summarize_messages(messages, max_chars=summary_max_chars)
        try:
            response = model.query(
                [
                    {
                        "role": "user",
                        "content": _build_prompt(
                            fallback_trace,
                            snapshot=snapshot,
                            max_actions=max_actions,
                            char_limit=builtin.config.char_limit,
                        ),
                    }
                ]
            )
        except Exception as fallback_e:
            logger.warning("Consolidation LLM fallback call failed: %s", fallback_e)
            return {"applied": 0, "skipped": 0, "error": f"{e}; fallback failed: {fallback_e}"}

    actions = ((response.get("extra") or {}).get("actions") or [])[:max_actions]
    applied, skipped = 0, 0
    for action in actions:
        if action.get("tool_name") != "memory":
            skipped += 1
            continue
        args = action.get("args") or {}
        op = args.get("action", "")
        if op == "add":
            builtin.add(args.get("content", ""))
        elif op == "replace":
            builtin.replace(args.get("old_text", ""), args.get("content", ""))
        elif op == "remove":
            builtin.remove(args.get("old_text", ""))
        else:
            skipped += 1
            continue
        applied += 1
    return {"applied": applied, "skipped": skipped}


def _build_prompt(trace: str, *, snapshot: str, max_actions: int, char_limit: int) -> str:
    return _PROMPT_TEMPLATE.format(
        max_actions=max_actions,
        char_limit=char_limit,
        snapshot=snapshot,
        trace=trace,
    )


def _format_messages(messages: list[dict]) -> str:
    return "\n".join(_message_chunks(messages))


def _message_chunks(messages: list[dict]) -> list[str]:
    chunks: list[str] = []
    for msg in messages:
        text = extract_message_text(msg).strip()
        if text:
            chunks.append(f"[{msg.get('role', '')}] {text}")
    return chunks


def _summarize_messages(messages: list[dict], *, max_chars: int) -> str:
    """Flatten messages into a tagged transcript, tail-truncated to ``max_chars``.

    Tail-biased: when the budget is exceeded we keep the most recent turns and
    drop earliest ones, on the assumption that the lessons learned at the end
    of a session are usually the most consolidation-worthy.
    """
    chunks = _message_chunks(messages)

    if not chunks:
        return ""

    total = sum(len(c) + 1 for c in chunks)
    if total <= max_chars:
        return "\n".join(chunks)

    kept: list[str] = []
    used = 0
    for chunk in reversed(chunks):
        if used + len(chunk) + 1 > max_chars:
            break
        kept.append(chunk)
        used += len(chunk) + 1
    kept.reverse()
    return "…(earlier trace truncated)\n" + "\n".join(kept)
