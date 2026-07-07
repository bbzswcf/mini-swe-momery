"""`MemoryAgent` extends `DefaultAgent` with persistent memory + provider tools.

- Injects the frozen MEMORY snapshot into the system prompt via the
  `memory_block` template variable (see digest §1.3 / §3).
- Pushes ``manager.get_tool_schemas()`` onto the model's ``extra_tools`` at
  construction time so the model actually sees the ``memory`` /
  ``hindsight_*`` / ``mem0_*`` tools alongside ``bash``. Appends rather than
  replaces so other consumers' tools survive.
- Routes built-in `memory` tool calls and any registered provider tool calls
  to `MemoryManager`; all other actions flow to the environment unchanged.
- After each step calls ``manager.sync_turn(user, assistant)`` so providers
  can buffer trial transcripts; Hindsight writes the buffered task once at
  session end.
- At run start, prepends any provider ``initial_context(task)`` to the first
  user message, giving Hindsight a cache-friendly one-shot recall path.
- After each step also calls ``manager.maybe_consolidate(model, messages,
  n_calls=...)`` — a no-op unless ``consolidation.every_n_steps > 0`` and that
  many model calls have elapsed without a successful MEMORY.md write/update.
  When triggered it runs a memory-only LLM turn that may write to MEMORY.md
  (digest §5 / "background memory flush" equivalent).
- `manager.initialize` runs at session start; `manager.on_session_end(messages,
  model=...)` always runs in `finally` — it both records the session into the
  FTS-backed `SessionStore` (powering the `session_search` tool) and, if
  ``consolidation.on_session_end`` is set, runs one final consolidation turn
  so the trial's lessons are persisted before shutdown. If the agent built its
  own manager from a config dict, it also calls `manager.shutdown()` on exit;
  an externally-injected manager is left alone so the caller can reuse it
  across sessions (e.g. SWE-bench batches).
"""

from __future__ import annotations

import json

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.memory import MemoryManager
from minisweagent.memory.session_store import extract_message_text


class MemoryAgent(DefaultAgent):
    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        manager: MemoryManager | None = None,
        memory: dict | None = None,
        config_class: type = AgentConfig,
        **kwargs,
    ) -> None:
        if manager is not None and memory is not None:
            raise ValueError("Pass either `manager` or `memory`, not both.")
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.manager = manager if manager is not None else MemoryManager.from_config(memory or {})
        self._owns_manager = manager is None
        # Expose memory/provider tool schemas to the model. Each underlying model
        # class (LitellmModel, PortkeyModel, …) reads `extra_tools` in its `_query`
        # and `_parse_actions` paths. Append rather than replace so we don't trample
        # tools registered by other consumers; test doubles without the attribute
        # simply skip this — they wire actions manually.
        if hasattr(self.model, "extra_tools"):
            self.model.extra_tools = [
                *list(getattr(self.model, "extra_tools", []) or []),
                *self.manager.get_tool_schemas(),
            ]

    def get_template_vars(self, **kwargs) -> dict:
        return super().get_template_vars(memory_block=self.manager.system_prompt_block(), **kwargs)

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        tool_names = self.manager.tool_names
        outputs = [
            _result_to_output(self.manager.handle_tool_call(a["tool_name"], a.get("args", {})))
            if a.get("tool_name") in tool_names
            else self.env.execute(a)
            for a in actions
        ]
        formatted = self.model.format_observation_messages(message, outputs, self.get_template_vars())
        # Sync the just-completed (previous user → assistant) pair before observations
        # are appended, so `_previous_user_text` doesn't pick up the freshly-formatted
        # observations as the "user" side. Providers should keep this hook cheap.
        self._sync_turn(message)
        return self.add_messages(*formatted)

    def step(self) -> list[dict]:
        result = super().step()
        try:
            self.manager.maybe_consolidate(self.model, self.messages, n_calls=self.n_calls)
        except Exception:  # consolidation must never break the main loop
            pass
        return result

    def run(self, task: str = "", session_id: str = "default", **kwargs) -> dict:
        self.manager.initialize(session_id, **kwargs)
        if context := self.manager.initial_context(task):
            task = f"{context}\n\n{task}"
        try:
            return super().run(task, **kwargs)
        finally:
            try:
                self.manager.on_session_end(self.messages, model=self.model)
            finally:
                if self._owns_manager:
                    self.manager.shutdown()

    def _sync_turn(self, assistant_message: dict) -> None:
        user_text = self._previous_user_text(assistant_message)
        assistant_text = _serialize_assistant(assistant_message)
        if user_text or assistant_text:
            self.manager.sync_turn(user_text, assistant_text)

    def _previous_user_text(self, assistant_message: dict) -> str:
        """The most recent user/tool message before this assistant turn."""
        for msg in reversed(self.messages):
            if msg is assistant_message:
                continue
            if msg.get("role") in ("user", "tool") or msg.get("type") == "function_call_output":
                return extract_message_text(msg)
        return ""


def _serialize_assistant(message: dict) -> str:
    """Render the assistant turn as text for sync_turn — text + tool-call summary."""
    content = message.get("content") or ""
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and "text" in p)
    actions = message.get("extra", {}).get("actions", [])
    if not actions:
        return content
    summary = json.dumps(
        [
            {"tool": a.get("tool_name", "bash"), "args": a.get("args") or {"command": a.get("command", "")}}
            for a in actions
        ],
        ensure_ascii=False,
    )
    return f"{content}\n{summary}".strip()


def _result_to_output(result: dict) -> dict:
    """Convert a manager tool-call result into an env-style observation output.

    Memory tool calls always succeed at the *transport* level — failure (capacity,
    no match, etc.) is encoded in the JSON itself via `success`/`error` keys, which
    the model reads. We therefore always emit ``returncode=0`` and skip
    ``exception_info`` so the model doesn't conflate this with a shell failure.
    """
    return {"output": json.dumps(result, ensure_ascii=False), "returncode": 0, "exception_info": ""}
