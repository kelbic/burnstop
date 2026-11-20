"""Offline fake Anthropic Messages API for the burnstop demo (no network, no API key).

POST /v1/messages  -> realistic SSE (or JSON) responses. The "agent" makes two normal moves,
                      then keeps asking to re-run the same failing test, while its context
                      (and the bill) grows every turn.
POST /webhook      -> prints the burnstop alert it receives.

    python demo/mock_upstream.py --port 9100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from aiohttp import web

MODEL = "claude-sonnet-5-5"
SCRIPT = [
    ("Let me look at the auth module first.", "read_file", {"path": "src/auth.py"}),
    ("The token check is inverted. Fixing it.", "edit_file",
     {"path": "src/auth.py", "old": "if not expired", "new": "if expired"}),
]
LOOP = ("Running the tests again to confirm the fix.", "bash",
        {"command": "pytest tests/test_auth.py -q"})


def _turn(body: dict) -> int:
    return sum(1 for m in body.get("messages", []) if m.get("role") == "assistant")


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def messages(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    turn = _turn(body)
    text, tool, args = SCRIPT[turn] if turn < len(SCRIPT) else LOOP
    usage = {"input_tokens": 9_000 + 16_000 * turn, "output_tokens": 1,
             "cache_read_input_tokens": 2_000 * turn, "cache_creation_input_tokens": 1_500}
    tool_id = f"toolu_demo_{turn:02d}"
    if not body.get("stream"):
        return web.json_response({
            "id": f"msg_demo_{turn}", "type": "message", "role": "assistant", "model": MODEL,
            "content": [{"type": "text", "text": text},
                        {"type": "tool_use", "id": tool_id, "name": tool, "input": args}],
            "stop_reason": "tool_use", "usage": {**usage, "output_tokens": 380}})
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    raw_args = json.dumps(args)
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": f"msg_demo_{turn}", "type": "message", "role": "assistant", "model": MODEL,
            "content": [], "usage": usage}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {
            "type": "tool_use", "id": tool_id, "name": tool, "input": {}}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {
            "type": "input_json_delta", "partial_json": raw_args[:12]}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {
            "type": "input_json_delta", "partial_json": raw_args[12:]}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                           "usage": {"output_tokens": 380}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    for name, data in events:
        await resp.write(_sse(name, data))
        await asyncio.sleep(0.03)
    await resp.write_eof()
    return resp


async def webhook(request: web.Request) -> web.Response:
    payload = await request.json()
    first = (payload.get("text") or "").splitlines()[0] if payload.get("text") else payload
    print(f"\033[35m[webhook] alert delivered: {first}\033[0m", flush=True)
    return web.json_response({"ok": True})


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/messages", messages)
    app.router.add_post("/webhook", webhook)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9100)
    port = ap.parse_args().port
    print(f"mock upstream on http://127.0.0.1:{port}", file=sys.stderr, flush=True)
    web.run_app(make_app(), host="127.0.0.1", port=port, print=None, access_log=None)
