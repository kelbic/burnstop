"""Exactly one alert per trip; API keys never reach logs, state, history or webhooks."""
import asyncio
import copy
import json
import logging
import subprocess
import sys

import aiohttp

from burnstop.alerts import Alerter, redact
from burnstop.config import DEFAULTS, validate
from conftest import ANTHROPIC_KEY, OPENAI_KEY, anthropic_stream, settle

TOOL_SECRET = "sk-ant-api03-SECRETINSIDETOOLARGS999"


async def _loop(url: str, n: int, headers: dict) -> list[int]:
    codes = []
    async with aiohttp.ClientSession() as s:
        for _ in range(n):
            async with s.post(url + "/v1/messages", headers=headers, json={
                    "model": "claude-sonnet-5-5", "stream": True, "messages": [
                        {"role": "user", "content": f"use key {ANTHROPIC_KEY}"}]}) as r:
                await r.read()
                codes.append(r.status)
    return codes


async def test_one_alert_per_trip_and_no_secret_anywhere(proxy, upstream, cfg, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    url, p = proxy
    upstream.queue(sse_body=anthropic_stream("bash", {"command": f"curl -H 'x-api-key: {TOOL_SECRET}'"}),
                   times=4)
    headers = {"x-api-key": ANTHROPIC_KEY, "authorization": f"Bearer {OPENAI_KEY}",
               "anthropic-version": "2023-06-01"}
    codes = await _loop(url, 7, headers)
    await settle(p)
    assert codes == [200, 200, 200, 200, 402, 402, 402]
    assert len(upstream.webhooks) == 1                       # ARMED -> TRIPPED once, not per block
    webhook = upstream.webhooks[0]["body"]
    assert "repeated 4x" in json.loads(webhook)["reason"]

    p.breaker.reset()
    upstream.queue(sse_body=anthropic_stream("edit", {"f": 1}), times=4)
    await _loop(url, 5, headers)
    await settle(p)
    assert len(upstream.webhooks) == 2                       # a new trip after reset alerts again

    everything = "\n".join([
        webhook, upstream.webhooks[1]["body"], open(cfg["state_file"]).read(),
        open(cfg["history_file"]).read(), caplog.text, "\n".join(p.lines),
        capsys.readouterr().out])
    for secret in (ANTHROPIC_KEY, OPENAI_KEY, TOOL_SECRET, "TESTSECRET", "SECRETINSIDE"):
        assert secret not in everything


def test_redact_patterns():
    text = (f"x-api-key: {ANTHROPIC_KEY} Authorization: Bearer {OPENAI_KEY} "
            "legacy sk-abcdefghijklmnopqrstuvwx bot 123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw")
    out = redact(text)
    for leaked in ("TESTSECRET", "sk-abcdef", "AAHdqTcv"):
        assert leaked not in out
    assert redact("model claude-sonnet-5-5 cost $0.01") == "model claude-sonnet-5-5 cost $0.01"


async def test_telegram_format_and_token_not_logged(upstream, caplog):
    caplog.set_level(logging.DEBUG)
    token = "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"
    c = copy.deepcopy(DEFAULTS)
    c["alerts"] = [{"type": "telegram", "bot_token": token, "chat_id": "42"},
                   {"type": "slack", "url": f"{upstream.url}/webhook"},
                   {"type": "discord", "url": f"{upstream.url}/webhook"}]
    a = Alerter(validate(c), telegram_api=upstream.url)
    await a.fire("identical tool call repeated 4x: bash", {"10m": 1.0}, ["bash"])
    tg, slack, discord = upstream.webhooks
    assert tg["path"] == f"/bot{token}/sendMessage"
    assert json.loads(tg["body"])["chat_id"] == "42" and "TRIPPED" in json.loads(tg["body"])["text"]
    assert "text" in json.loads(slack["body"]) and "content" in json.loads(discord["body"])
    # only burnstop's own records: the fake Telegram server's access log shows its own URL
    ours = "\n".join(r.getMessage() for r in caplog.records if r.name == "burnstop")
    assert a.sent == 3 and "alert telegram: sent" in ours
    assert token not in ours and "AAHdqTcv" not in ours


async def test_kill_switch_pid_and_command_run_once(tmp_path):
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pid_file, marker = tmp_path / "agent.pid", tmp_path / "ran"
    pid_file.write_text(str(victim.pid))
    c = copy.deepcopy(DEFAULTS)
    c["kill_switch"] = {"kill_pid_file": str(pid_file), "on_trip_cmd": f"echo x >> {marker}"}
    await Alerter(validate(c)).fire("budget", {"10m": 3.0}, [])
    assert victim.wait(timeout=5) != 0                     # terminated by SIGTERM
    assert marker.read_text() == "x\n"
    await asyncio.sleep(0)
