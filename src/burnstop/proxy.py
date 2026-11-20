"""Async reverse proxy: transparent streaming plus pre/post circuit-breaker checks."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable

import aiohttp
from aiohttp import web

from .alerts import Alerter, redact
from .breaker import TRIPPED, Breaker
from .parser import ANTHROPIC, OPENAI, SSEParser, parse_json, tool_fingerprint
from .pricing import Pricing

log = logging.getLogger("burnstop")

HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "keep-alive",
               "proxy-connection", "te", "trailer", "upgrade", "accept-encoding",
               "content-encoding"}
METERED = {"/v1/messages": ANTHROPIC, "/v1/chat/completions": OPENAI, "/v1/responses": OPENAI}


def route(path: str, headers) -> tuple[str, str]:
    """-> (provider, upstream path). Explicit /anthropic or /openai prefixes win."""
    for provider in (ANTHROPIC, OPENAI):
        prefix = f"/{provider}"
        if path == prefix or path.startswith(prefix + "/"):
            return provider, path[len(prefix):] or "/"
    if path.rstrip("/") in METERED:
        return METERED[path.rstrip("/")], path
    if "x-api-key" in headers or "anthropic-version" in headers:
        return ANTHROPIC, path
    return (OPENAI if "authorization" in headers else ANTHROPIC), path


def trip_response(provider: str, reason: str, status: int) -> web.Response:
    msg = f"burnstop circuit breaker is TRIPPED: {reason}. Run `burnstop reset` to re-arm."
    if provider == ANTHROPIC:
        body = {"type": "error", "error": {"type": "burnstop_tripped", "message": msg}}
    else:
        body = {"error": {"message": msg, "type": "burnstop_tripped", "param": None,
                          "code": "burnstop_tripped"}}
    return web.json_response(body, status=status, headers={"X-Burnstop-Tripped": reason[:200]})


class Proxy:
    def __init__(self, cfg: dict, breaker: Breaker, pricing: Pricing, alerter: Alerter,
                 emit: Callable[[str], None] | None = None):
        self.cfg, self.breaker, self.pricing, self.alerter = cfg, breaker, pricing, alerter
        self.emit = emit or (lambda line: None)
        self.session: aiohttp.ClientSession | None = None
        self._tasks: set[asyncio.Task] = set()

    def app(self) -> web.Application:
        app = web.Application(client_max_size=256 * 1024 * 1024)
        app.router.add_route("*", "/{tail:.*}", self.handle)
        app.on_startup.append(self._start)
        app.on_cleanup.append(self._stop)
        return app

    async def _start(self, _app) -> None:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=900)
        self.session = aiohttp.ClientSession(timeout=timeout, auto_decompress=True)

    async def _stop(self, _app) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.session:
            await self.session.close()

    def _on_trip(self, reason: str) -> None:
        self.emit(f"TRIPPED {reason}")
        task = asyncio.create_task(self.alerter.fire(
            reason, self.breaker.spend(), self.breaker.tool_names))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def handle(self, request: web.Request) -> web.StreamResponse:
        provider, path = route(request.path, request.headers)
        metered = request.method == "POST" and path.rstrip("/") in METERED
        if metered:
            reason, tripped_now = self.breaker.precheck()
        else:  # other paths pass through untouched unless the breaker is already TRIPPED
            self.breaker.sync()
            reason, tripped_now = (self.breaker.reason if self.breaker.status == TRIPPED
                                   else None), False
        if reason:
            if tripped_now:
                self._on_trip(reason)
            return trip_response(provider, reason, self.cfg["lockout"]["trip_status_code"])

        body = await request.read()
        out_body, injected = body, False
        if metered and provider == OPENAI and path.endswith("/chat/completions") \
                and self.cfg["inject_openai_stream_usage"]:
            out_body, injected = _inject_usage(body)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        headers["Accept-Encoding"] = "identity"
        url = self.cfg["upstreams"][provider].rstrip("/") + path
        if request.query_string:
            url += "?" + request.query_string

        try:
            upstream = await self.session.request(request.method, url, headers=headers,
                                                  data=out_body, allow_redirects=False)
            if injected and upstream.status == 400:  # upstream rejected stream_options: retry as-is
                upstream.release()
                upstream = await self.session.request(request.method, url, headers=headers,
                                                      data=body, allow_redirects=False)
        except aiohttp.ClientError as exc:
            log.warning("upstream %s unreachable: %s", provider, type(exc).__name__)
            if metered:
                self._account(provider, path, 502, None, len(body))
            return web.json_response({"error": {"message": "burnstop: upstream unreachable"}},
                                     status=502)

        resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS}
        is_sse = upstream.headers.get("Content-Type", "").startswith("text/event-stream")
        if not is_sse:
            data = await upstream.read()
            upstream.release()
            if metered:
                usage = parse_json(provider, path, data, len(body)) if upstream.status < 400 else None
                self._account(provider, path, upstream.status, usage, len(body))
            return web.Response(body=data, status=upstream.status, headers=resp_headers)

        parser = SSEParser(provider, path, len(body))
        out = web.StreamResponse(status=upstream.status, headers=resp_headers)
        try:
            await out.prepare(request)
            async for chunk in upstream.content.iter_any():
                parser.feed(chunk)
                await out.write(chunk)  # forward immediately: no buffering of the stream
            await out.write_eof()
        except (ConnectionResetError, aiohttp.ClientError) as exc:
            log.info("stream interrupted: %s", type(exc).__name__)
        finally:
            upstream.release()
            if metered:  # account even if the client disconnected mid-stream
                self._account(provider, path, upstream.status, parser.finish(), len(body))
        return out

    def _account(self, provider: str, path: str, status: int, usage, req_chars: int) -> None:
        cost = self.pricing.cost(usage) if usage else 0.0
        names = [n for n, _ in usage.tool_calls] if usage else []
        entry = {"ts": time.time(), "provider": provider, "path": path, "status": status,
                 "model": usage.model if usage else "", "cost_usd": round(cost, 6),
                 "input_tokens": usage.input_tokens if usage else 0,
                 "output_tokens": usage.output_tokens if usage else 0,
                 "cache_read_tokens": usage.cache_read_tokens if usage else 0,
                 "cache_write_tokens": usage.cache_write_tokens if usage else 0,
                 "estimated": bool(usage and usage.estimated), "tool_names": names,
                 "tool_fps": [tool_fingerprint(n, a) for n, a in usage.tool_calls] if usage else []}
        reason, tripped_now = self.breaker.record(entry)
        spend = self.breaker.spend()
        self.emit(redact(
            f"{time.strftime('%H:%M:%S')} {provider:<9} {entry['model'][:28]:<20} "
            f"in={entry['input_tokens']} out={entry['output_tokens']}"
            f"{'~' if entry['estimated'] else ''} ${cost:.4f} | 10m ${spend['10m']:.2f} "
            f"1h ${spend['1h']:.2f} day ${spend['day']:.2f} | "
            f"{'TRIPPED' if reason else 'OK'}{' HTTP %d' % status if status >= 400 else ''}"))
        if tripped_now:
            self._on_trip(reason)


def _inject_usage(body: bytes) -> tuple[bytes, bool]:
    try:
        data = json.loads(body)
    except ValueError:
        return body, False
    if not isinstance(data, dict) or not data.get("stream"):
        return body, False
    opts = data.get("stream_options") or {}
    if opts.get("include_usage"):
        return body, False
    data["stream_options"] = {**opts, "include_usage": True}
    return json.dumps(data).encode(), True


async def serve(proxy: Proxy, host: str, port: int) -> web.AppRunner:
    runner = web.AppRunner(proxy.app(), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()
    return runner
