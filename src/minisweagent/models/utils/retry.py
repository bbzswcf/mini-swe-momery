"""Retry utility for model queries."""

import logging
import os
import re

from tenacity import RetryCallState, Retrying, retry_if_not_exception_type, stop_after_attempt, wait_exponential

_DEFAULT_STOP_AFTER_ATTEMPT = 20
_DEFAULT_WAIT_MIN_SECONDS = 10
_DEFAULT_WAIT_MAX_SECONDS = 60

_LITELLM_HELP_LINE_PATTERNS = (
    re.compile(r"^\s*Provider List:\s+https://docs\.litellm\.ai/docs/providers\s*$"),
    re.compile(r"^\s*Give Feedback / Get Help:\s+https://github\.com/BerriAI/litellm/issues/new\s*$"),
    re.compile(r"^\s*LiteLLM\.Info: If you need to debug this error, use `litellm\._turn_on_debug\(\)'\.\s*$"),
)


def clean_retry_error_message(message: str) -> str:
    """Drop LiteLLM help links that otherwise flood concurrent batch logs."""
    return "\n".join(
        line for line in message.splitlines() if not any(pattern.match(line) for pattern in _LITELLM_HELP_LINE_PATTERNS)
    ).strip()


def before_sleep_clean_log(logger: logging.Logger, log_level: int):
    def log_it(retry_state: RetryCallState) -> None:
        if retry_state.outcome is None or retry_state.next_action is None:
            return
        exc = retry_state.outcome.exception()
        if exc is None:
            return
        fn_name = getattr(retry_state.fn, "__qualname__", None) if retry_state.fn else "<unknown>"
        logger.log(
            log_level,
            "Retrying %s in %.0f seconds as it raised %s: %s.",
            fn_name,
            retry_state.next_action.sleep,
            type(exc).__name__,
            clean_retry_error_message(str(exc)),
        )

    return log_it


def retry(*, logger: logging.Logger, abort_exceptions: list[type[Exception]]) -> Retrying:
    """Thin wrapper around tenacity.Retrying to make use of global config etc.

    Args:
        logger: Logger to use for reporting retries
        abort_exceptions: Exceptions to abort on.

    Returns:
        A tenacity.Retrying object.
    """
    return Retrying(
        reraise=True,
        stop=stop_after_attempt(
            int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", str(_DEFAULT_STOP_AFTER_ATTEMPT)))
        ),
        wait=wait_exponential(multiplier=1, min=_DEFAULT_WAIT_MIN_SECONDS, max=_DEFAULT_WAIT_MAX_SECONDS),
        before_sleep=before_sleep_clean_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(tuple(abort_exceptions)),
    )
