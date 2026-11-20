"""Token usage and tool-call extraction from Anthropic / OpenAI responses (JSON and SSE).

The SSE parser is incremental: feed() takes raw chunks exactly as they arrive (lines may be
split across chunks) and never needs the full response, so the proxy can forward each chunk
to the client before parsing it.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

ANTHROPIC, OPENAI = "anthropic", "openai"


@dataclass
class Usage:
    provider: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0
    estimated: bool = False
    tool_calls: list[tuple[str, str]] = field(default_factory=list)  # (name, canonical args)
    text_chars: int = 0
    saw_usage: bool = False


def canonical_args(raw) -> str:
    """Stable JSON for tool arguments: key order and whitespace must not hide a repeat."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except ValueError:
            return raw.strip()
    return json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tool_fingerprint(name: str, args: str) -> str:
    return hashlib.sha256(f"{name}\0{args}".encode()).hexdigest()[:16]


def _i(d: dict | None, key: str) -> int:
    v = (d or {}).get(key)
    return int(v) if isinstance(v, (int, float)) else 0


def _anthropic_usage(u: Usage, usage: dict | None, overwrite: bool = True) -> None:
    if not usage:
        return
    u.saw_usage = True
    for attr, key in (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens"),
                      ("cache_read_tokens", "cache_read_input_tokens"),
                      ("cache_write_tokens", "cache_creation_input_tokens")):
        val = _i(usage, key)
        if overwrite or val:
            setattr(u, attr, val)
    if isinstance(usage.get("cache_creation"), dict):
        u.cache_write_1h_tokens = _i(usage["cache_creation"], "ephemeral_1h_input_tokens")


def _openai_usage(u: Usage, usage: dict | None) -> None:
    if not usage:
        return
    u.saw_usage = True
    total_in = _i(usage, "prompt_tokens") or _i(usage, "input_tokens")
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    u.cache_read_tokens = _i(details, "cached_tokens")
    u.input_tokens = max(total_in - u.cache_read_tokens, 0)  # cached tokens priced separately
    u.output_tokens = _i(usage, "completion_tokens") or _i(usage, "output_tokens")


def _responses_output(u: Usage, output: list | None) -> None:
    for item in output or []:
        if item.get("type") == "function_call":
            u.tool_calls.append((item.get("name", ""), canonical_args(item.get("arguments", ""))))
        for part in item.get("content") or []:
            u.text_chars += len(part.get("text") or "")


def _finish(u: Usage, request_chars: int) -> Usage:
    if not u.saw_usage:  # provider sent no usage: conservative chars/4 estimate
        u.estimated = True
        u.input_tokens = math.ceil(request_chars / 4)
        u.output_tokens = math.ceil((u.text_chars + sum(len(a) for _, a in u.tool_calls)) / 4)
    return u


def parse_json(provider: str, path: str, body: bytes, request_chars: int = 0) -> Usage:
    u = Usage(provider)
    try:
        data = json.loads(body or b"{}")
    except ValueError:
        return _finish(u, request_chars)
    if not isinstance(data, dict):
        return _finish(u, request_chars)
    u.model = str(data.get("model") or "")
    if provider == ANTHROPIC:
        _anthropic_usage(u, data.get("usage"))
        for block in data.get("content") or []:
            if block.get("type") == "tool_use":
                u.tool_calls.append((block.get("name", ""), canonical_args(block.get("input", {}))))
            u.text_chars += len(block.get("text") or "")
    elif path.endswith("/responses"):
        _openai_usage(u, data.get("usage"))
        _responses_output(u, data.get("output"))
    else:
        _openai_usage(u, data.get("usage"))
        for choice in data.get("choices") or []:
            msg = choice.get("message") or {}
            u.text_chars += len(msg.get("content") or "")
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                u.tool_calls.append((fn.get("name", ""), canonical_args(fn.get("arguments", ""))))
    return _finish(u, request_chars)


class SSEParser:
    """Incremental server-sent-events parser for one streamed response."""

    def __init__(self, provider: str, path: str, request_chars: int = 0):
        self.u = Usage(provider)
        self.responses_api = provider == OPENAI and path.endswith("/responses")
        self.request_chars = request_chars
        self._buf = b""
        self._data: list[str] = []
        self._blocks: dict[int, list[str]] = {}  # index -> [name, partial args json]

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._line(line.rstrip(b"\r").decode("utf-8", "replace"))

    def _line(self, line: str) -> None:
        if not line:
            if self._data:
                self._event("\n".join(self._data))
            self._data = []
        elif line.startswith("data:"):
            self._data.append(line[5:].lstrip(" "))

    def _event(self, payload: str) -> None:
        if payload.strip() == "[DONE]":
            return
        try:
            data = json.loads(payload)
        except ValueError:
            return
        if not isinstance(data, dict):
            return
        if self.u.provider == ANTHROPIC:
            self._anthropic(data)
        elif self.responses_api:
            self._responses(data)
        else:
            self._chat(data)

    def _anthropic(self, d: dict) -> None:
        kind = d.get("type")
        if kind == "message_start":
            msg = d.get("message") or {}
            self.u.model = str(msg.get("model") or self.u.model)
            _anthropic_usage(self.u, msg.get("usage"))
        elif kind == "content_block_start":
            cb = d.get("content_block") or {}
            if cb.get("type") == "tool_use":
                start = json.dumps(cb["input"]) if cb.get("input") else ""
                self._blocks[int(d.get("index", 0))] = [cb.get("name", ""), start]
        elif kind == "content_block_delta":
            delta = d.get("delta") or {}
            if delta.get("type") == "input_json_delta" and int(d.get("index", 0)) in self._blocks:
                self._blocks[int(d.get("index", 0))][1] += delta.get("partial_json", "")
            self.u.text_chars += len(delta.get("text") or "")
        elif kind == "message_delta":  # cumulative counts; zeros must not erase message_start
            _anthropic_usage(self.u, d.get("usage"), overwrite=False)

    def _chat(self, d: dict) -> None:
        self.u.model = str(d.get("model") or self.u.model)
        _openai_usage(self.u, d.get("usage"))
        for choice in d.get("choices") or []:
            delta = choice.get("delta") or {}
            self.u.text_chars += len(delta.get("content") or "")
            for tc in delta.get("tool_calls") or []:
                slot = self._blocks.setdefault(int(tc.get("index", 0)), ["", ""])
                fn = tc.get("function") or {}
                slot[0] += fn.get("name") or ""
                slot[1] += fn.get("arguments") or ""

    def _responses(self, d: dict) -> None:
        kind = d.get("type", "")
        if kind.endswith(".delta"):
            self.u.text_chars += len(d.get("delta") or "") if isinstance(d.get("delta"), str) else 0
        resp = d.get("response") or {}
        if kind in ("response.completed", "response.incomplete", "response.failed"):
            self.u.model = str(resp.get("model") or self.u.model)
            self.u.text_chars = 0  # the final output is authoritative
            _openai_usage(self.u, resp.get("usage"))
            _responses_output(self.u, resp.get("output"))

    def finish(self) -> Usage:
        if self._buf.strip():
            self._line(self._buf.decode("utf-8", "replace").rstrip("\r"))
        self._line("")
        for idx in sorted(self._blocks):
            name, args = self._blocks[idx]
            self.u.tool_calls.append((name, canonical_args(args)))
        return _finish(self.u, self.request_chars)
