"""Trip alerts (Telegram / Slack / Discord / generic JSON webhooks), kill switch, redaction.

Fires once per ARMED -> TRIPPED transition. Every outgoing text passes through redact(); the
Telegram URL embeds the bot token, so URLs are never logged.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import signal

import aiohttp

log = logging.getLogger("burnstop")

SECRET_HEADERS = {"authorization", "proxy-authorization", "x-api-key", "api-key",
                  "anthropic-api-key", "openai-api-key", "cookie", "set-cookie"}
_SECRETS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{6,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]{8,}=*"),
    re.compile(r"\b\d{6,}:[A-Za-z0-9_\-]{30,}\b"),  # Telegram bot token
]


def redact(text: str) -> str:
    for rx in _SECRETS:
        text = rx.sub("[REDACTED]", text)
    return text


def trip_message(reason: str, spend: dict, tool_names: list[str]) -> str:
    lines = [f"burnstop TRIPPED: {reason}",
             "Spend: " + " | ".join(f"{k} ${v:.2f}" for k, v in spend.items())]
    if tool_names:
        lines.append("Last tool call: " + ", ".join(tool_names))
    lines.append("Agent traffic is blocked. Run `burnstop reset` to re-arm.")
    return redact("\n".join(lines))


class Alerter:
    def __init__(self, cfg: dict, telegram_api: str = "https://api.telegram.org"):
        self.targets = cfg["alerts"]
        self.kill = cfg["kill_switch"]
        self.telegram_api = telegram_api
        self.sent = 0

    def _request(self, target: dict, text: str, event: dict) -> tuple[str, dict]:
        kind = target["type"]
        if kind == "telegram":
            url = f"{self.telegram_api}/bot{target['bot_token']}/sendMessage"
            return url, {"chat_id": target["chat_id"], "text": text}
        if kind == "slack":
            return target["url"], {"text": text}
        if kind == "discord":
            return target["url"], {"content": text}
        return target["url"], {"event": "burnstop.tripped", "text": text, **event}

    async def fire(self, reason: str, spend: dict, tool_names: list[str]) -> None:
        text = trip_message(reason, spend, tool_names)
        event = {"reason": redact(reason), "spend_usd": spend, "tool_names": tool_names}
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for target in self.targets:
                url, payload = self._request(target, text, event)
                try:
                    async with session.post(url, json=payload) as resp:
                        ok = resp.status < 300
                    log.info("alert %s: %s", target["type"], "sent" if ok else f"HTTP {resp.status}")
                    self.sent += ok
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    log.warning("alert %s failed: %s", target["type"], type(exc).__name__)
        await self.kill_switch()

    async def kill_switch(self) -> None:
        pid_file, cmd = self.kill.get("kill_pid_file"), self.kill.get("on_trip_cmd")
        if pid_file:
            try:
                pid = int(open(os.path.expanduser(pid_file)).read().strip())
                os.kill(pid, signal.SIGTERM)
                log.warning("kill switch: sent SIGTERM to pid %d", pid)
            except (OSError, ValueError) as exc:
                log.warning("kill switch: pid file %s unusable: %s", pid_file, type(exc).__name__)
        if cmd:
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            log.warning("kill switch: on_trip_cmd exited with %s", await proc.wait())
