import logging
import time
from collections.abc import Callable

import litellm

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall_response import (
    BASH_TOOL_RESPONSE_API,
    format_toolcall_observation_messages,
    parse_toolcall_actions_response,
    to_response_api_tool,
)
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_response_model")


class LitellmResponseModelConfig(LitellmModelConfig):
    pass


class LitellmResponseModel(LitellmModel):
    def __init__(self, *, config_class: Callable = LitellmResponseModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        """Flatten response objects into their output items for stateless API calls."""
        result = []
        for msg in messages:
            if msg.get("object") == "response":
                for item in msg.get("output", []):
                    result.append({k: v for k, v in item.items() if k != "extra"})
            else:
                result.append({k: v for k, v in msg.items() if k != "extra"})
        return result

    def _tools(self) -> list[dict]:
        return [BASH_TOOL_RESPONSE_API, *(to_response_api_tool(t) for t in self.extra_tools)]

    def _allowed_tool_names(self) -> set[str]:
        return {t["name"] for t in self._tools()}

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.responses(
                model=self.config.model_name,
                input=messages,
                tools=self._tools(),
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def query_no_tools(self, messages: list[dict], **kwargs) -> dict:
        """Plain text→text call without a ``tools`` array or ``tool_choice``.

        Used by features like chain-window compression where attaching the bash
        tool + ``tool_choice="none"`` makes modelhub's Responses gateway raise
        ``UnsupportedParamsError`` (which sits in ``abort_exceptions`` and
        silently re-raises out of the retry loop). Tool-related kwargs are
        stripped from ``model_kwargs`` for the same reason."""
        request_kwargs = dict(self.config.model_kwargs or {})
        for key in ("reasoning", "include", "parallel_tool_calls", "tool_choice", "tools"):
            request_kwargs.pop(key, None)
        request_kwargs.update(kwargs)
        prepared = self._prepare_messages_for_api(messages)
        # Share the same retry policy as ``query()`` — modelhub Responses TPM
        # limits (RateLimitError) are frequent and transient; without a retry
        # loop a single TPM hit during compression would warn-and-bail, then
        # ``ChainWindowAgent._compress_now`` disables compression for the rest
        # of the chain. ``UnsupportedParamsError`` / ``AuthenticationError``
        # stay in ``abort_exceptions`` so genuinely-fatal calls still surface.
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = litellm.responses(
                    model=self.config.model_name,
                    input=prepared,
                    **request_kwargs,
                )
        return response.model_dump() if hasattr(response, "model_dump") else dict(response)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        message["extra"] = {
            "actions": self._parse_actions(response),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _parse_actions(self, response) -> list[dict]:
        return parse_toolcall_actions_response(
            getattr(response, "output", []),
            format_error_template=self.config.format_error_template,
            allowed_tools=self._allowed_tool_names(),
        )

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )
