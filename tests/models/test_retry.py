import logging

from minisweagent.models.utils.retry import clean_retry_error_message, retry


def test_clean_retry_error_message_removes_litellm_help_lines():
    assert clean_retry_error_message(
        "\n".join(
            [
                "RateLimitError: useful error",
                "Provider List: https://docs.litellm.ai/docs/providers",
                "Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new",
                "LiteLLM.Info: If you need to debug this error, use `litellm._turn_on_debug()'.",
            ]
        )
    ) == "RateLimitError: useful error"


def test_litellm_debug_help_is_suppressed():
    import litellm

    import minisweagent.models.litellm_model  # noqa: F401

    assert litellm.suppress_debug_info is True
    assert litellm.set_verbose is False


def test_retry_defaults_use_longer_backoff():
    retrying = retry(logger=logging.getLogger(__name__), abort_exceptions=[])
    assert retrying.stop.max_attempt_number == 20
    assert retrying.wait.min == 10
    assert retrying.wait.max == 60
