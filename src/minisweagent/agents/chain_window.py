"""``ChainWindowAgent`` — run a whole SWE-bench-Pro chain in one LLM context window.

Drop-in agent that extends ``DefaultAgent``:

- Adds a ``process_chain(instances, ...)`` entry point that solves every
  instance in a chain back-to-back, reusing the same ``self.messages``.
- Tracks the index of the latest task message (``_task_anchor``) so the
  "compressible region" is unambiguous: ``messages[1:_task_anchor]``.
- After every step (and just before starting a new task) checks the latest
  ``usage.input_tokens`` against a configurable fraction of the model's
  context window; when exceeded, replaces the compressible region with one
  user message produced by ``window_compress.compress_history`` (a single
  ``model.query`` call against the same main model).

Plug-and-play: not registered as the default agent, not used by the existing
``run/benchmarks/swebench.py`` chain dispatcher. Use the bundled
``run/benchmarks/swebench_chain_window.py`` runner or instantiate directly.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Callable

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.agents.window_compress import (
    CompressionConfig,
    compress_history,
)
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded


class ChainWindowAgent(DefaultAgent):
    def __init__(
        self,
        model: Any,
        env: Any,
        *,
        compression: dict | None = None,
        config_class: type = AgentConfig,
        **kwargs,
    ) -> None:
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.compression = CompressionConfig(**(compression or {}))
        self._task_anchor: int = 0
        self._completed_instance_ids: list[str] = []
        self._last_input_tokens: int = 0
        self._compression_log: list[dict] = []
        self._summary_present: bool = False
        # Per-task counters. ``self.n_calls`` and ``self.cost`` keep accumulating
        # across the whole chain (so the chain's total LLM spend is visible in
        # trajectories/logs), but ``step_limit`` / ``cost_limit`` are interpreted
        # *per task* — otherwise a long chain's later tasks would all hit
        # ``LimitsExceeded`` on their very first query because the agent instance
        # is reused across tasks.
        self._task_n_calls: int = 0
        self._task_cost: float = 0.0

    # ------------------------------------------------------------------ chain

    def process_chain(
        self,
        instances: list[dict],
        *,
        render_task: Callable[[dict], str],
        on_instance_end: Callable[[dict, dict, list[dict]], None] | None = None,
    ) -> list[dict]:
        """Run every instance in ``instances`` back-to-back in one context window.

        Returns a list of per-instance ``{instance_id, info}`` dicts. Per-instance
        trajectories are emitted via the optional ``on_instance_end`` callback so
        callers (the runner) decide how to persist them.
        """
        results: list[dict] = []
        for i, instance in enumerate(instances):
            instance_id = instance["instance_id"]
            task = render_task(instance)
            try:
                info = self._run_one(task, first=(i == 0))
                err: Exception | None = None
            except Exception as exc:
                info = {"exit_status": type(exc).__name__, "submission": "", "exception_str": str(exc)}
                err = exc
            messages_for_instance = self._extract_instance_messages()
            self._completed_instance_ids.append(instance_id)
            results.append({"instance_id": instance_id, "info": info, "error": err})
            if on_instance_end is not None:
                on_instance_end(instance, info, messages_for_instance)
            self._seal_completed_task()
        return results

    # ----------------------------------------------------------------- internal

    def _run_one(self, task: str, *, first: bool) -> dict:
        self.extra_template_vars["task"] = task
        self._task_n_calls = 0
        self._task_cost = 0.0
        if first:
            self.messages = []
            self.add_messages(
                self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
                self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
            )
            self._task_anchor = len(self.messages) - 1
        else:
            self.add_messages(
                self.model.format_message(role="user", content=self._render_template(self.config.instance_template))
            )
            self._task_anchor = len(self.messages) - 1
            self._maybe_compress(reason="pre_task")
        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {}) or {}

    def query(self) -> dict:
        # Override DefaultAgent.query: limit checks are per-task here, not
        # per-agent. Otherwise the second task in a chain would inherit the
        # first task's ``n_calls`` and trip ``step_limit`` immediately.
        if 0 < self.config.step_limit <= self._task_n_calls or 0 < self.config.cost_limit <= self._task_cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        self._task_n_calls += 1
        message = self.model.query(self.messages)
        cost_delta = message.get("extra", {}).get("cost", 0.0)
        self.cost += cost_delta
        self._task_cost += cost_delta
        self.add_messages(message)
        self._record_token_usage()
        return message

    def step(self) -> list[dict]:
        result = super().step()
        self._maybe_compress(reason="post_step")
        return result

    def _record_token_usage(self) -> None:
        for msg in reversed(self.messages):
            usage = msg.get("usage") if isinstance(msg, dict) else None
            if isinstance(usage, dict) and isinstance(usage.get("input_tokens"), int):
                self._last_input_tokens = usage["input_tokens"]
                return

    def _has_compressible_region(self) -> bool:
        """True iff at least one full completed task sits between system and anchor.

        After a compression the region looks like ``[summary]`` — exactly one
        message — and there's no benefit to re-compressing it on its own. We only
        re-fire if a *new* completed task has been appended on top of the summary.
        """
        region_len = self._task_anchor - 1
        if region_len <= 0:
            return False
        if self._summary_present and region_len <= 1:
            return False
        return True

    def _exceeds_threshold(self) -> bool:
        return self._last_input_tokens >= self.compression.token_trigger

    def _maybe_compress(self, *, reason: str) -> None:
        if not self.compression.enabled:
            return
        if not self._has_compressible_region():
            return
        if not self._exceeds_threshold():
            return
        self._compress_now(reason=reason)

    def _compress_now(self, *, reason: str) -> None:
        middle = self.messages[1 : self._task_anchor]
        summary = compress_history(self.model, middle, config=self.compression)
        if not summary:
            # No cooldown: ``query_no_tools`` already wraps transient errors
            # (RateLimit / network) in the same retry loop ``query`` uses, so
            # the failures that bubble up here are either ``abort_exceptions``
            # (UnsupportedParams / Auth / NotFound — re-fire is harmless, the
            # next compress call deterministically fails the same way and gets
            # logged again) or retry-exhausted transients (worth re-trying on
            # the next step). Either way, recording the failure and letting
            # the threshold re-trigger normally beats permanently giving up.
            self._compression_log.append(
                {
                    "reason": reason,
                    "status": "failed",
                    "input_tokens_before": self._last_input_tokens,
                    "after_anchor": self._task_anchor,
                    "completed_instance_ids": list(self._completed_instance_ids),
                }
            )
            return
        replacement = self.model.format_message(
            role="user",
            content=f"<compressed_history>\n{summary}\n</compressed_history>",
        )
        before = len(self.messages)
        self.messages[1 : self._task_anchor] = [replacement]
        delta = before - len(self.messages)
        self._task_anchor -= delta
        self._summary_present = True
        self._compression_log.append(
            {
                "reason": reason,
                "status": "ok",
                "input_tokens_before": self._last_input_tokens,
                "messages_collapsed": delta + 1,
                "after_anchor": self._task_anchor,
                "completed_instance_ids": list(self._completed_instance_ids),
            }
        )

    def _strip_trailing_exit(self) -> None:
        if self.messages and self.messages[-1].get("role") == "exit":
            self.messages.pop()

    def _pad_unanswered_tool_calls(self) -> None:
        """Ensure every ``function_call`` in the last assistant response has a
        matching ``function_call_output``.

        When the agent submits, ``env.execute`` raises ``Submitted`` mid-iteration
        and ``execute_actions`` never appends observation messages. In single-task
        mode that's fine because the agent shuts down; in chain-window mode the
        next task's first model.query() would send a Responses-API conversation
        with an unmatched function_call and modelhub rejects it with -4003
        ("No tool output found for function call ..."). Synthesise a placeholder
        output so the conversation stays well-formed across the task boundary.
        """
        if not self.messages:
            return
        last = self.messages[-1]
        if not isinstance(last, dict) or last.get("object") != "response":
            return
        outputs = last.get("output") or []
        for item in outputs:
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            call_id = item.get("call_id")
            if not call_id:
                continue
            self.messages.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": "<task submitted; no further output>",
                }
            )

    def _seal_completed_task(self) -> None:
        """Make the message history safe to continue a Responses-API conversation."""
        self._strip_trailing_exit()
        self._pad_unanswered_tool_calls()

    def _extract_instance_messages(self) -> list[dict]:
        """Return the messages that belong to the most recent task (incl. system + summary)."""
        head: list[dict] = [self.messages[0]] if self.messages else []
        if self._summary_present and self._task_anchor >= 2:
            head.append(self.messages[1])
        tail = self.messages[self._task_anchor :]
        return deepcopy(head + tail)

    # ----------------------------------------------------------------- serialize

    def serialize(self, *extra_dicts: dict) -> dict:
        base = super().serialize(*extra_dicts)
        base.setdefault("info", {})["chain_window"] = {
            "compression": asdict(self.compression),
            "compression_log": list(self._compression_log),
            "completed_instance_ids": list(self._completed_instance_ids),
            "last_input_tokens": self._last_input_tokens,
        }
        return base
