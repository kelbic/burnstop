"""Real local HTTP/SSE round trips: client -> burnstop -> fake upstream."""
import asyncio
import json

import aiohttp

from conftest import ANTHROPIC_KEY, OPENAI_KEY, anthropic_stream, sse

AH = {"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
OH = {"authorization": f"Bearer {OPENAI_KEY}", "content-type": "application/json"}
MSG = {"model": "claude-sonnet-5-5", "max_tokens": 64, "stream": True,
       "messages": [{"role": "user", "content": "hi"}]}


async def post(url, path, body, headers):
    async with aiohttp.ClientSession() as s:
        async with s.post(url + path, json=body, headers=headers) as r:
            return r.status, await r.read(), dict(r.headers)


async def test_sse_chunks_reach_client_before_upstream_finishes(proxy, upstream):
    url, _ = proxy
    upstream.gate = asyncio.Event()
    upstream.queue(sse_body=anthropic_stream())
    async with aiohttp.ClientSession() as s:
        async with s.post(url + "/v1/messages", json=MSG, headers=AH) as r:
            first = await asyncio.wait_for(r.content.readuntil(b"\n\n"), timeout=3)
            assert b"message_start" in first           # arrived while upstream is still paused
            upstream.gate.set()
            rest = await r.read()
    assert b"message_stop" in rest


async def test_loop_trips_then_402_without_touching_upstream(proxy, upstream):
    url, p = proxy
    upstream.queue(sse_body=anthropic_stream("bash", {"command": "pytest -q"}), times=4)
    for _ in range(4):
        status, _, _ = await post(url, "/v1/messages", MSG, AH)
        assert status == 200
    calls = len(upstream.requests)
    status, body, headers = await post(url, "/v1/messages", MSG, AH)
    assert status == 402 and len(upstream.requests) == calls
    err = json.loads(body)
    assert err["type"] == "error" and err["error"]["type"] == "burnstop_tripped"   # Anthropic shape
    assert "repeated 4x" in headers["X-Burnstop-Tripped"]


async def test_openai_error_shape_after_trip(proxy, upstream):
    url, p = proxy
    p.breaker.trip("manual test trip")
    status, body, headers = await post(url, "/v1/chat/completions", {"model": "gpt-4o"}, OH)
    err = json.loads(body)["error"]
    assert status == 402 and err["type"] == "burnstop_tripped" and "manual test trip" in err["message"]
    assert not upstream.requests


async def test_non_metered_paths_pass_through_until_tripped(proxy, upstream):
    url, p = proxy
    upstream.queue(body=b'{"data": []}')
    async with aiohttp.ClientSession() as s:
        async with s.get(url + "/v1/models", headers=AH) as r:
            assert r.status == 200 and await r.json() == {"data": []}
        p.breaker.trip("manual")
        async with s.get(url + "/v1/models", headers=AH) as r:
            assert r.status == 402
    assert len(upstream.requests) == 1


async def test_count_tokens_is_not_billed(proxy, upstream, cfg):
    url, p = proxy
    upstream.queue(body=b'{"input_tokens": 12}')
    status, _, _ = await post(url, "/v1/messages/count_tokens", {"messages": []}, AH)
    assert status == 200 and p.breaker.spend()["session"] == 0


async def test_auth_headers_forwarded_to_upstream(proxy, upstream):
    url, _ = proxy
    upstream.queue(sse_body=anthropic_stream())
    await post(url, "/v1/messages", MSG, AH)
    await post(url, "/openai/v1/chat/completions", {"model": "gpt-4o"}, OH)
    assert upstream.requests[0]["headers"]["x-api-key"] == ANTHROPIC_KEY
    assert upstream.requests[1]["headers"]["Authorization"] == f"Bearer {OPENAI_KEY}"
    assert upstream.requests[1]["path"] == "/v1/chat/completions"


async def test_openai_stream_usage_injection_and_400_retry(proxy, upstream):
    url, p = proxy
    chunks = [{"model": "gpt-4o", "choices": [{"delta": {"content": "ok"}}]},
              {"model": "gpt-4o", "choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 10}}]
    upstream.queue(sse_body=sse(chunks, anthropic=False))
    body = {"model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "x"}]}
    status, _, _ = await post(url, "/v1/chat/completions", body, OH)
    assert status == 200
    sent = json.loads(upstream.requests[0]["body"])
    assert sent["stream_options"] == {"include_usage": True}
    assert p.breaker.spend()["session"] > 0
    # an upstream that rejects stream_options gets the original request instead
    upstream.reject_stream_options = True
    upstream.queue(sse_body=sse(chunks, anthropic=False))
    status, _, _ = await post(url, "/v1/chat/completions", body, OH)
    assert status == 200 and b"stream_options" not in upstream.requests[-1]["body"]


async def test_upstream_errors_trip_consecutive_error_rule(proxy, upstream):
    url, p = proxy
    upstream.queue(status=529, body=b'{"type":"error","error":{"type":"overloaded_error"}}', times=6)
    for _ in range(6):
        status, _, _ = await post(url, "/v1/messages", {**MSG, "stream": False}, AH)
        assert status == 529                              # errors are passed through untouched
    status, _, _ = await post(url, "/v1/messages", {**MSG, "stream": False}, AH)
    assert status == 402 and "consecutive upstream errors" in p.breaker.reason
