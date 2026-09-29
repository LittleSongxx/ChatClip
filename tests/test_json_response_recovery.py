import copy
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from app.llm import AnthropicJsonClient, JsonResponseError, LangChainJsonClient, parse_json_object


class ScriptedChatModel:
    """Stands in for the LangChain chat model created by ``create_chat_model``."""

    def __init__(self, responses: list[BaseMessage]) -> None:
        self.responses = list(responses)
        self.bindings: list[dict] = []
        self.invocations: list[list[BaseMessage]] = []

    def bind(self, **kwargs):
        self.bindings.append(copy.deepcopy(kwargs))
        return self

    def invoke(self, messages, *_args, **_kwargs):
        self.invocations.append(copy.deepcopy(list(messages)))
        if not self.responses:
            raise AssertionError("scripted model exhausted")
        return self.responses.pop(0)


def test_only_trailing_commas_outside_strings_are_removed():
    assert parse_json_object('{"text":"literal ,} and ,]", "ranges":[1,2,],}') == {
        "text": "literal ,} and ,]", "ranges": [1, 2],
    }


@pytest.mark.parametrize("text", ['{"start": 10, "end":', '{"reason":"unescaped "quote""}', '{"start":1 "end":2}'])
def test_ambiguous_or_truncated_evidence_is_rejected(text):
    with pytest.raises(JsonResponseError):
        parse_json_object(text)


def test_invalid_json_retries_with_correction_and_larger_budget():
    model = ScriptedChatModel([
        AIMessage(content='{"start":1 "end":2}'),
        AIMessage(content='{"start":1,"end":2}'),
    ])
    with patch("app.llm.client.create_chat_model", return_value=model):
        client = LangChainJsonClient(api_key="test", model="test", base_url="http://invalid")
        client._cancelled.wait = lambda seconds: False
        result = client.complete_json("original evidence", maximum_tokens=1000)
    assert result["end"] == 2
    assert len(model.invocations) == 2
    assert model.invocations[1][0] == model.invocations[0][0]
    assert "JSON" in model.invocations[1][-1].content
    assert model.bindings[-1]["max_tokens"] == 2000


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("budget", [1000, 20000])
def test_length_stop_retries_without_reducing_original_budget(provider, budget):
    first = provider == "openai"
    stop_key = "finish_reason" if first else "stop_reason"
    stopped_early = "length" if first else "max_tokens"
    completed = "stop" if first else "end_turn"
    model = ScriptedChatModel([
        AIMessage(content='{"ok":true}', response_metadata={stop_key: stopped_early}),
        AIMessage(content='{"ok":true}', response_metadata={stop_key: completed}),
    ])
    with patch("app.llm.client.create_chat_model", return_value=model):
        if first:
            client = LangChainJsonClient(api_key="test", model="test", base_url="http://invalid")
        else:
            client = AnthropicJsonClient(auth_token="test", model="test", base_url="http://invalid")
        client._cancelled.wait = lambda seconds: False
        assert client.complete_json("original evidence", maximum_tokens=budget)["ok"] is True
    assert len(model.invocations) == 2
    if first:
        assert model.bindings[-1]["max_tokens"] == max(budget, min(16384, budget * 2))
    else:
        assert model.bindings[-1]["max_tokens"] == max(16384, budget * 2)
    assert model.invocations[1][0] == model.invocations[0][0]
    assert "JSON" in model.invocations[1][-1].content
