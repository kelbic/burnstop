"""Command line: burnstop init | start | status | reset | tail."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .alerts import Alerter
from .breaker import TRIPPED, Breaker
from .config import ConfigError, load
from .pricing import Pricing
from .proxy import Proxy, serve

TEMPLATE = """\
# burnstop.yaml - set any rule to null to disable it.
listen: {host: 127.0.0.1, port: 8787}

limits:                      # USD, sliding windows
  max_usd_per_10m: 2.00      # sudden burn
  max_usd_per_hour: 5.00
  max_usd_per_day: 20.00
  max_usd_per_session: null  # since start or `burnstop reset --clear-session`

loop_guard:
  max_identical_tool_calls: 4   # same tool + same arguments in N responses in a row
  max_consecutive_errors: 6     # 4xx/5xx from upstream in a row...
  error_window_minutes: 5       # ...within this window
  max_requests_per_minute: 30   # a `while True` without sleeps

lockout:
  trip_status_code: 402      # SDKs auto-retry 429/5xx; 402 stops them
  cooldown_minutes: null     # null = stay locked until `burnstop reset`

state_file: ~/.burnstop/state.json
history_file: ~/.burnstop/history.jsonl

alerts: []
#  - {type: telegram, bot_token: "123456:ABC...", chat_id: "123456789"}
#  - {type: slack, url: "https://hooks.slack.com/services/..."}
#  - {type: discord, url: "https://discord.com/api/webhooks/..."}
#  - {type: generic, url: "https://example.com/burnstop-webhook"}

kill_switch:
  kill_pid_file: null        # SIGTERM the PID in this file on trip
  on_trip_cmd: null          # e.g. "systemctl --user stop my-agent"

# pricing:
#   models: {my-finetune: {input: 3.0, output: 15.0, cache_read: 0.3, cache_write: 3.75}}
#   unknown_model_fallback_per_1m: {input: 15.0, output: 75.0}
"""

RED, GREEN, DIM, BOLD, RESET = "\033[31m", "\033[32m", "\033[2m", "\033[1m", "\033[0m"


def _c(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if sys.stdout.isatty() else text


def _config(path: str | None) -> dict:
    if path is None and Path("burnstop.yaml").exists():
        path = "burnstop.yaml"
    return load(path)


def cmd_init(args) -> int:
    target = Path(args.path)
    if target.exists() and not args.force:
        print(f"{target} already exists (use --force to overwrite)")
        return 1
    target.write_text(TEMPLATE)
    print(f"wrote {target}. Next: burnstop start, then export ANTHROPIC_BASE_URL=http://127.0.0.1:8787")
    return 0


def _emit(line: str) -> None:
    if line.startswith("TRIPPED"):
        bar = "!" * 64
        print(_c(RED + BOLD, f"\n{bar}\n  CIRCUIT BREAKER TRIPPED: {line[8:]} - saved your API balance\n"
                             f"  Further agent requests get HTTP 402. `burnstop reset` to re-arm.\n{bar}\n"),
              flush=True)
    else:
        print(_c(RED, line) if line.endswith("TRIPPED") else line, flush=True)


async def _run(cfg: dict) -> None:
    breaker = Breaker(cfg)
    pricing = Pricing(cfg["pricing"]["models"], cfg["pricing"]["unknown_model_fallback_per_1m"])
    proxy = Proxy(cfg, breaker, pricing, Alerter(cfg), emit=_emit)
    host, port = cfg["listen"]["host"], cfg["listen"]["port"]
    runner = await serve(proxy, host, port)
    state = _c(RED, "TRIPPED: " + str(breaker.reason)) if breaker.status == TRIPPED else _c(GREEN, "ARMED")
    lim, g = cfg["limits"], cfg["loop_guard"]
    usd = {k: "off" if v is None else f"${v:.2f}" for k, v in lim.items()}
    print(f"burnstop {__version__} listening on http://{host}:{port}  [{state}]\n"
          f"  anthropic -> {cfg['upstreams']['anthropic']}\n  openai    -> {cfg['upstreams']['openai']}\n"
          + _c(DIM, f"  caps: 10m {usd['max_usd_per_10m']} | 1h {usd['max_usd_per_hour']} | day "
                    f"{usd['max_usd_per_day']} | loop {g['max_identical_tool_calls']}x | errors "
                    f"{g['max_consecutive_errors']} | rpm {g['max_requests_per_minute']}"), flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


def cmd_start(args) -> int:
    cfg = _config(args.config)
    if args.port:
        cfg["listen"]["port"] = args.port
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        asyncio.run(_run(cfg))
    except KeyboardInterrupt:
        pass
    return 0


def cmd_status(args) -> int:
    b = Breaker(_config(args.config))
    if args.json:
        print(json.dumps({"status": b.status, "reason": b.reason, "spend_usd": b.spend(),
                          "recent": list(b.recent)}, indent=2))
        return 0
    head = _c(RED, f"TRIPPED  {b.reason}") if b.status == TRIPPED else _c(GREEN, "ARMED")
    print(f"status: {head}")
    if b.tripped_at:
        print(f"tripped at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(b.tripped_at))}")
    print("spend:  " + "  ".join(f"{k} ${v:.4f}" for k, v in b.spend().items()))
    for e in b.recent:
        stamp = time.strftime("%H:%M:%S", time.localtime(e.get("ts", 0)))
        tools = ",".join(e.get("tool_names") or []) or "-"
        print(f"  {stamp} {e.get('model', '')[:28]:<28} ${e.get('cost_usd', 0):.4f} "
              f"HTTP {e.get('status')} tools={tools}")
    return 0


def cmd_reset(args) -> int:
    b = Breaker(_config(args.config))
    was = b.status
    b.reset(clear_session=args.clear_session)
    print(f"{was} -> ARMED" + (" (session spend cleared)" if args.clear_session else ""))
    return 0


def cmd_tail(args) -> int:
    path = Path(_config(args.config)["history_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    with path.open() as f:
        for line in f.readlines()[-args.lines:]:
            print(line.rstrip())
        if not args.follow:
            return 0
        try:
            while True:
                line = f.readline()
                print(line.rstrip(), flush=True) if line else time.sleep(0.5)
        except KeyboardInterrupt:
            return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="burnstop", description="Circuit breaker for AI agent API spend.")
    ap.add_argument("--version", action="version", version=f"burnstop {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="write a starter burnstop.yaml")
    p.add_argument("path", nargs="?", default="burnstop.yaml")
    p.add_argument("--force", action="store_true")
    for name, helptext in (("start", "run the proxy"), ("status", "show state and spend"),
                           ("reset", "re-arm a tripped breaker"), ("tail", "print request history")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("-c", "--config", default=None)
        if name == "start":
            p.add_argument("--port", type=int, default=None)
        if name == "status":
            p.add_argument("--json", action="store_true")
        if name == "reset":
            p.add_argument("--clear-session", action="store_true")
        if name == "tail":
            p.add_argument("-n", "--lines", type=int, default=20)
            p.add_argument("-f", "--follow", action="store_true")
    args = ap.parse_args(argv)
    try:
        return {"init": cmd_init, "start": cmd_start, "status": cmd_status,
                "reset": cmd_reset, "tail": cmd_tail}[args.cmd](args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
