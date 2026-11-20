"""Offline demo: a coding agent gets stuck re-running a failing test; burnstop stops it.

One command, no API key, no internet:

    python demo/runaway_agent.py            # starts mock upstream + burnstop, runs the agent
    python demo/runaway_agent.py --fast     # no pauses (CI)

The agent talks to burnstop exactly like an Anthropic SDK client would (ANTHROPIC_BASE_URL).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from burnstop.parser import SSEParser  # noqa: E402

RED, YEL, CYAN, DIM, BOLD, RESET = "\033[31m", "\033[33m", "\033[36m", "\033[2m", "\033[1m", "\033[0m"
FAKE_KEY = "sk-ant-demo-0000-not-a-real-key"  # burnstop forwards it, never logs it


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise SystemExit(f"port {port} did not open")


def call(base: str, messages: list) -> tuple[int, bytes]:
    req = urllib.request.Request(
        f"{base}/v1/messages", method="POST",
        data=json.dumps({"model": "claude-sonnet-5-5", "max_tokens": 4096, "stream": True,
                         "messages": messages}).encode(),
        headers={"content-type": "application/json", "x-api-key": FAKE_KEY,
                 "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def run_agent(base: str, pause: float) -> int:
    messages = [{"role": "user", "content": "Fix the failing auth test in this repo."}]
    for step in range(1, 21):
        status, body = call(base, messages)
        if status != 200:
            err = json.loads(body).get("error", {})
            print(f"{RED}{BOLD}agent > step {step}: HTTP {status} {err.get('type')} - "
                  f"agent halted, API balance saved.{RESET}", flush=True)
            return 0
        parser = SSEParser("anthropic", "/v1/messages")
        parser.feed(body)
        usage = parser.finish()
        name, args = usage.tool_calls[0]
        a = json.loads(args)
        shown = str(a.get("command") or a.get("path") or args)[:60]
        outcome = "1 failed, 12 passed" if name == "bash" else "ok"
        print(f"{CYAN}agent > step {step}: {name}({shown}) -> {outcome}{RESET}", flush=True)
        messages += [{"role": "assistant", "content": [
                         {"type": "tool_use", "id": f"t{step}", "name": name, "input": json.loads(args)}]},
                     {"role": "user", "content": [
                         {"type": "tool_result", "tool_use_id": f"t{step}", "content": outcome}]}]
        time.sleep(pause)
    print(f"{YEL}agent ran 20 steps without being stopped{RESET}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="no pauses between steps")
    ap.add_argument("--proxy", help="use an already running burnstop (e.g. http://127.0.0.1:8787)")
    args = ap.parse_args()
    pause = 0 if args.fast else 0.9
    if args.proxy:
        return run_agent(args.proxy.rstrip("/"), pause)

    up_port, proxy_port = free_port(), free_port()
    state = Path(tempfile.mkdtemp(prefix="burnstop-demo-"))
    cfg = state / "burnstop.yaml"
    cfg.write_text(json.dumps({  # JSON is valid YAML
        "listen": {"host": "127.0.0.1", "port": proxy_port},
        "upstreams": {"anthropic": f"http://127.0.0.1:{up_port}"},
        "limits": {"max_usd_per_10m": 5.0, "max_usd_per_hour": 10.0, "max_usd_per_day": 20.0},
        "loop_guard": {"max_identical_tool_calls": 4},
        "state_file": str(state / "state.json"), "history_file": str(state / "history.jsonl"),
        "alerts": [{"type": "generic", "url": f"http://127.0.0.1:{up_port}/webhook"}]}))
    env = {**os.environ, "PYTHONPATH": str(HERE.parent / "src")}
    procs = [subprocess.Popen([sys.executable, str(HERE / "mock_upstream.py"), "--port", str(up_port)],
                              env=env, stderr=subprocess.DEVNULL)]
    try:
        wait_port(up_port)
        procs.append(subprocess.Popen([sys.executable, "-m", "burnstop.cli", "start", "-c", str(cfg)],
                                      env=env))
        wait_port(proxy_port)
        time.sleep(pause)
        print(f"{DIM}$ export ANTHROPIC_BASE_URL=http://127.0.0.1:{proxy_port}{RESET}", flush=True)
        rc = run_agent(f"http://127.0.0.1:{proxy_port}", pause)
        time.sleep(0.8)  # let the webhook line land
        return rc
    finally:
        for p in reversed(procs):
            p.terminate()
            p.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
