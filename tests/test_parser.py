import json

import pytest

from burnstop.parser import SSEParser, canonical_args, parse_json, tool_fingerprint
from burnstop.pricing import Pricing
from conftest import sse


def feed_in_pieces(parser: SSEParser, data: bytes, size: int = 7):
    for i in range(0, len(data), size):  # lines deliberately split across chunks
        parser.feed(data[i:i + size])
    return parser.finish()


def test_anthropic_json_with_cache_and_tool_use():
    body = json.dumps({
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": "Running it."},
                    {"type": "tool_use", "id": "t1", "name": "bash", "input": {"b": 2, "a": 1}}],
        "usage": {"input_tokens": 120, "output_tokens": 45, "cache_read_input_tokens": 5000,
                  "cache_creation_input_tokens": 300,
                  "cache_creation": {"ephemeral_5m_input_tokens": 200, "ephemeral_1h_input_tokens": 100}},
    }).encode()
    u = parse_json("anthropic", "/v1/messages", body)
    assert (u.model, u.input_tokens, u.output_tokens) == ("claude-opus-5-5", 120, 45)
    assert (u.cache_read_tokens, u.cache_write_tokens, u.cache_write_1h_tokens) == (5000, 300, 100)
    assert u.tool_calls == [("bash", '{"a":1,"b":2}')] and not u.estimated


def test_anthropic_sse_incremental_tool_args_and_usage():
    data = sse([
        {"type": "message_start", "message": {"model": "claude-sonnet-5-5", "usage": {
            "input_tokens": 900, "output_tokens": 1, "cache_read_input_tokens": 4000,
            "cache_creation_input_tokens": 250}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "t", "name": "bash", "input": {}}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '{"command": "pyt'}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": 'est tests/"}'}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 77}},
        {"type": "message_stop"},
    ])
    u = feed_in_pieces(SSEParser("anthropic", "/v1/messages"), data)
    assert (u.model, u.input_tokens, u.output_tokens) == ("claude-sonnet-5-5", 900, 77)
    assert (u.cache_read_tokens, u.cache_write_tokens) == (4000, 250)
    assert u.tool_calls == [("bash", '{"command":"pytest tests/"}')] and not u.estimated


def test_openai_chat_json_tool_calls_and_cached_tokens():
    body = json.dumps({"model": "gpt-4.1-2025-04-14", "choices": [{"message": {
        "content": None, "tool_calls": [{"index": 0, "function": {
            "name": "run", "arguments": '{"cmd": "ls", "dir": "/"}'}}]}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 50,
                  "prompt_tokens_details": {"cached_tokens": 800}}}).encode()
    u = parse_json("openai", "/v1/chat/completions", body)
    assert (u.input_tokens, u.cache_read_tokens, u.output_tokens) == (200, 800, 50)
    assert u.tool_calls == [("run", '{"cmd":"ls","dir":"/"}')]


def test_openai_chat_sse_tool_calls_merged_by_index_and_usage_chunk():
    chunks = [
        {"model": "gpt-4o", "choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "search", "arguments": '{"q": '}},
            {"index": 1, "function": {"name": "fetch", "arguments": '{"url": "a"}'}}]}}]},
        {"model": "gpt-4o", "choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '"cats"}'}}]}}]},
        {"model": "gpt-4o", "choices": [], "usage": {"prompt_tokens": 30, "completion_tokens": 9}},
    ]
    u = feed_in_pieces(SSEParser("openai", "/v1/chat/completions"), sse(chunks, anthropic=False), 5)
    assert u.tool_calls == [("search", '{"q":"cats"}'), ("fetch", '{"url":"a"}')]
    assert (u.input_tokens, u.output_tokens, u.estimated) == (30, 9, False)


def test_openai_sse_without_usage_is_estimated():
    chunks = [{"model": "gpt-4o", "choices": [{"delta": {"content": "x" * 400}}]}]
    u = feed_in_pieces(SSEParser("openai", "/v1/chat/completions", request_chars=800),
                       sse(chunks, anthropic=False))
    assert u.estimated and u.input_tokens == 200 and u.output_tokens == 100


def test_openai_responses_json_and_sse():
    final = {"model": "gpt-5", "usage": {"input_tokens": 500, "output_tokens": 20,
                                         "input_tokens_details": {"cached_tokens": 100}},
             "output": [{"type": "function_call", "name": "shell", "arguments": '{"c":1}'}]}
    u = parse_json("openai", "/v1/responses", json.dumps(final).encode())
    assert (u.input_tokens, u.cache_read_tokens, u.tool_calls) == (400, 100, [("shell", '{"c":1}')])
    s = feed_in_pieces(SSEParser("openai", "/v1/responses"), sse([
        {"type": "response.output_text.delta", "delta": "abc"},
        {"type": "response.completed", "response": final}], anthropic=False))
    assert (s.input_tokens, s.output_tokens, s.tool_calls) == (400, 20, [("shell", '{"c":1}')])


def test_canonical_args_ignore_key_order_and_whitespace():
    assert canonical_args('{"b": 1, "a": [1, 2]}') == canonical_args({"a": [1, 2], "b": 1})
    assert tool_fingerprint("bash", canonical_args('{"x":1}')) != tool_fingerprint("sh", canonical_args('{"x":1}'))


def test_pricing_prefix_fallback_and_cache_math():
    p = Pricing()
    assert p.lookup("claude-sonnet-4-5-20250929")[0].input == 3
    assert p.lookup("claude-opus-4-1-20250805")[0].input == 15
    assert p.lookup("gpt-4o-mini-2024-07-18")[0].input == 0.15
    price, known = p.lookup("totally-new-model")
    assert not known and price.input == 15 and price.output == 75
    u = parse_json("anthropic", "/v1/messages", json.dumps({"model": "claude-opus-5-5", "usage": {
        "input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
        "cache_creation": {"ephemeral_1h_input_tokens": 1_000_000}}}).encode())
    # 4 input + 20 output + 0.20 cache read + 1h write at 2x input (8)
    assert p.cost(u) == pytest.approx(32.20)
    # newer OpenAI models must not inherit cheaper base-model prices through the prefix
    assert p.lookup("gpt-5.5")[0].output == 30 and p.lookup("gpt-5-pro")[0].output == 120
    assert p.lookup("o3-pro-2025-06-10")[0].input == 20 and p.lookup("gpt-5-2025-08-07")[0].input == 1.25
    assert p.lookup("gpt-5.9-preview") == (p.fallback, False)      # unknown dotted version: fallback
    custom = Pricing({"my-model": {"input": 1, "output": 2}})
    assert custom.lookup("my-model-v2")[0].output == 2
    assert custom.lookup("my-model2")[1] is False                   # no match across a name boundary
