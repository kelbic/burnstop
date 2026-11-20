"""Shared fixtures: a scriptable fake upstream (Anthropic + OpenAI + webhook) and a live
burnstop proxy wired to it. Everything is local; no test can reach a real API."""
from __future__ import annotations

import asyncio
import copy
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from burnstop.alerts import Alerter
from burnstop.breaker import Breaker
from burnstop.config import DEFAULTS, validate
from burnstop.pricing import Pricing
from burnstop.proxy import Proxy

ANTHROPIC_KEY = "sk-ant-api03-TESTSECRET0123456789abcdefghij"
OPENAI_KEY = "sk-proj-TESTSECRET0123456789abcdefghijklmn"


def sse(events: list[dict], anthropic: bool = True) -> bytes:
    out = b""
    for e in events:
        head = f"event: {e['type']}\n" if anthropic else ""
        out += f"{head}data: {json.dumps(e)}\n\n".encode()
    if not anthropic:
        out += b"data: [DONE]\n\n"
    return out


def anthropic_stream(tool: str = "bash", args: dict | None = None, inp: int = 1000,
                     out: int = 200, model: str = "claude-sonnet-5-5") -> bytes:
    raw = json.dumps(args if args is not None else {"command": "pytest"})
    return sse([
        {"type": "message_start", "message": {"model": model, "usage": {
            "input_tokens": inp, "output_tokens": 1, "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": "t", "name": tool, "input": {}}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": raw[:5]}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": raw[5:]}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": out}},
        {"type": "message_stop"},
    ])


class Upstream:
    """Scriptable fake provider. Push responses with .queue(); records every request."""

    def __init__(self):
        self.responses: list = []
        self.requests: list[dict] = []
        self.webhooks: list[dict] = []
        self.reject_stream_options = False
        self.gate: asyncio.Event | None = None   # SSE pauses after the first chunk until set

    def queue(self, status=200, body=b"", sse_body: bytes | None = None, times=1):
        self.responses += [(status, body, sse_body)] * times

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if request.path == "/webhook" or "/sendMessage" in request.path:
            self.webhooks.append({"path": request.path, "body": await request.text()})
            return web.json_response({"ok": True})
        raw = await request.read()
        self.requests.append({"path": request.path, "headers": dict(request.headers), "body": raw})
        if self.reject_stream_options and b"stream_options" in raw:
            return web.json_response({"error": {"message": "unknown field"}}, status=400)
        status, body, sse_body = self.responses.pop(0) if self.responses else (200, b"{}", None)
        if sse_body is None:
            return web.Response(status=status, body=body, content_type="application/json")
        resp = web.StreamResponse(status=status, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        first, rest = sse_body.split(b"\n\n", 1)
        await resp.write(first + b"\n\n")
        if self.gate is not None:
            await self.gate.wait()
        await resp.write(rest)
        await resp.write_eof()
        return resp


@pytest.fixture
async def upstream():
    up = Upstream()
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", up.handle)
    server = TestServer(app)
    await server.start_server()
    up.url = str(server.make_url("")).rstrip("/")
    yield up
    await server.close()


@pytest.fixture
def cfg(tmp_path, upstream):
    c = copy.deepcopy(DEFAULTS)
    c["upstreams"] = {"anthropic": upstream.url, "openai": upstream.url}
    c["state_file"] = str(tmp_path / "state.json")
    c["history_file"] = str(tmp_path / "history.jsonl")
    c["alerts"] = [{"type": "generic", "url": f"{upstream.url}/webhook"}]
    return validate(c)


@pytest.fixture
async def proxy(cfg, upstream):
    """-> (base url, Proxy). Emitted log lines are kept in proxy.lines."""
    lines: list[str] = []
    p = Proxy(cfg, Breaker(cfg), Pricing(), Alerter(cfg, telegram_api=upstream.url), emit=lines.append)
    p.lines = lines
    server = TestServer(p.app())
    await server.start_server()
    yield str(server.make_url("")).rstrip("/"), p
    await server.close()


async def settle(p: Proxy) -> None:
    """Wait for background alert tasks."""
    for _ in range(50):
        if not p._tasks:
            return
        await asyncio.sleep(0.02)
