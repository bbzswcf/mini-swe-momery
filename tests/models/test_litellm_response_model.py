"""Verify that every entry in `model_kwargs` actually lands in the outgoing
HTTP request when `LitellmResponseModel` is wired through `litellm.responses`.

Anchors the swebench_pro.yaml config so that any future regression (e.g. litellm
silently dropping `extra_headers`, `include`, or `reasoning`) fails loudly.
"""

import json

import httpx
import pytest

from minisweagent.config import get_config_from_spec
from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_response_model import LitellmResponseModel


_FAKE_RESPONSE = {
    "id": "resp_test",
    "object": "response",
    "created_at": 1234567890,
    "model": "gpt-test",
    "status": "completed",
    "output": [],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "metadata": {},
    "temperature": 1.0,
    "top_p": 1.0,
    "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


@pytest.fixture
def captured_request(monkeypatch):
    """Intercept the openai SDK's outbound httpx call and capture it."""
    captured: dict = {}

    def fake_send(self, request, **_):
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        body = request.read()
        captured["body"] = json.loads(body) if body else None
        return httpx.Response(200, json=_FAKE_RESPONSE, request=request)

    monkeypatch.setattr(httpx.Client, "send", fake_send)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "1")
    return captured


def test_swebench_pro_model_kwargs_reach_http_request(captured_request):
    cfg = get_config_from_spec("swebench_pro")["model"]
    mk = cfg["model_kwargs"]
    model = LitellmResponseModel(
        model_name=cfg["model_name"], model_kwargs=mk, cost_tracking="ignore_errors"
    )

    with pytest.raises(FormatError):  # empty `output` => no tool calls
        model.query([{"role": "user", "content": "hi", "type": "message"}])

    assert captured_request["url"].startswith(mk["api_base"])
    headers = {k.lower(): v for k, v in captured_request["headers"].items()}
    for k, v in mk["extra_headers"].items():
        assert headers[k.lower()] == v
    body = captured_request["body"]
    assert body["reasoning"] == mk["reasoning"]
    assert body["max_output_tokens"] == mk["max_output_tokens"]
    assert body["store"] == mk["store"]
    assert body["include"] == mk["include"]
    assert body["parallel_tool_calls"] == mk["parallel_tool_calls"]
    # `drop_params` is a litellm-side flag and intentionally does not appear in
    # the HTTP payload; assert that it was filtered out rather than leaking.
    assert "drop_params" not in body
