"""CodeBuddy rejects non-stream chat; the agent must stream and assemble."""
from unittest.mock import MagicMock

from openai.types.chat import ChatCompletionChunk

from agent import ContextAwareAgent, ContextMode

# Same 400 the live CodeBuddy endpoint returns for stream=false.
_NON_STREAM_ERROR = (
    "Error code: 400 - {'code': 11101, "
    "'msg': 'Non-stream chat request is currently not supported'}"
)


def _chunk(delta, *, finish_reason=None):
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-test",
            "created": 0,
            "model": "auto",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


def _text_stream(text):
    return [
        _chunk({"role": "assistant", "content": text}),
        _chunk({}, finish_reason="stop"),
    ]


def _tool_call_stream():
    """CodeBuddy repeats ``role`` on every chunk; the SDK concatenates it."""
    return [
        _chunk(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "convert_currency",
                            "arguments": "",
                        },
                    }
                ],
            }
        ),
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {
                            "arguments": (
                                '{"amount": 1000, "from_currency": "USD",'
                                ' "to_currency": "EUR"}'
                            ),
                        },
                    }
                ],
            }
        ),
        _chunk({"role": "assistant"}, finish_reason="tool_calls"),
    ]


def test_codebuddy_streams_when_non_stream_is_rejected():
    agent = ContextAwareAgent(
        "test-codebuddy-key",
        ContextMode.NO_TOOL_CALLS,
        provider="codebuddy",
        verbose=False,
    )
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        if not kwargs.get("stream"):
            raise Exception(_NON_STREAM_ERROR)
        return iter(_text_stream("FINAL ANSWER: 42"))

    agent.client = MagicMock()
    agent.client.chat.completions.create = fake_create

    result = agent.execute_task("what is 6*7", max_iterations=2)

    assert captured.get("stream") is True
    assert result.get("error") is None
    assert result["completed"] is True
    assert "42" in (result.get("final_answer") or "")


def test_codebuddy_round_trip_keeps_canonical_tool_history():
    """Streamed tool calls must round-trip without duplicated roles or extras.

    Live CodeBuddy returns 11148 ('tool calls and tool results do not
    match') if the follow-up messages keep the assembled junk: a
    concatenated ``assistantassistant`` role, empty ``function_call``,
    and extra tool-call fields.
    """
    agent = ContextAwareAgent(
        "test-codebuddy-key",
        ContextMode.FULL,
        provider="codebuddy",
        verbose=False,
    )
    calls = []

    def fake_create(**kwargs):
        if not kwargs.get("stream"):
            raise Exception(_NON_STREAM_ERROR)
        calls.append(kwargs)
        if len(calls) == 1:
            return iter(_tool_call_stream())
        assistant = [m for m in kwargs["messages"] if "assistant" in str(m.get("role"))]
        assert assistant, kwargs["messages"]
        msg = assistant[-1]
        assert msg["role"] == "assistant"
        assert "function_call" not in msg
        tool_calls = msg["tool_calls"]
        assert len(tool_calls) == 1
        assert set(tool_calls[0]) == {"id", "type", "function"}
        assert tool_calls[0]["id"] == "call_1"
        tools = [m for m in kwargs["messages"] if m.get("role") == "tool"]
        assert len(tools) == 1
        assert tools[0]["tool_call_id"] == "call_1"
        return iter(_text_stream("FINAL ANSWER: 920 EUR"))

    agent.client = MagicMock()
    agent.client.chat.completions.create = fake_create

    result = agent.execute_task("Convert 1000 USD to EUR", max_iterations=3)

    assert result.get("error") is None
    assert result["completed"] is True
    assert len(calls) == 2
    assert "920" in (result.get("final_answer") or "")
