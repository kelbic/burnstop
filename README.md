# burnstop

**A local circuit breaker for autonomous AI agents.** burnstop sits between your agent and the
Anthropic / OpenAI API, tracks spend per request in real time, and cuts the agent off with an
HTTP 402 the moment it starts looping or burning money: before the overnight run empties your key.

![Demo: an agent loops on a failing test; burnstop trips on the 4th identical tool call](docs/demo.gif)

## Why

- **Console spend limits are monthly and org-wide.** They trigger after the money is gone, and
  they stop every app on the key, not the one runaway agent.
- **Agents fail in loops, not crashes.** A coding agent re-running the same failing test keeps
  resending a growing context. Each turn costs more than the last, and `systemd` happily
  restarts it when it does crash.
- **You are asleep when it happens.** You need a rule that fires in seconds and stays tripped
  until a human looks, not a dashboard you check in the morning.

## Quickstart (30 seconds)

```bash
pip install git+https://github.com/kelbic/burnstop
burnstop init        # writes burnstop.yaml with sane defaults
burnstop start       # listens on 127.0.0.1:8787
```

Point your agent at it. Your API key stays in the agent and is passed through unchanged.

```bash
# Claude Code, Anthropic SDKs
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787

# OpenAI SDKs and most OpenAI-compatible clients (note the /v1)
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
```

Try it without a key or network: `python demo/runaway_agent.py` starts a fake upstream,
burnstop, and an agent that gets stuck on a failing test.

## Rules

Every rule is configurable in `burnstop.yaml`; set any of them to `null` to disable it.

| Rule | Default | Trips when |
|---|---|---|
| `max_usd_per_10m` | `$2.00` | spend in the last 10 minutes reaches the cap |
| `max_usd_per_hour` | `$5.00` | spend in the last hour reaches the cap |
| `max_usd_per_day` | `$20.00` | spend in the last 24 hours reaches the cap |
| `max_usd_per_session` | off | spend since start / `reset --clear-session` reaches the cap |
| `max_identical_tool_calls` | `4` | N responses in a row request the same tool with the same arguments |
| `max_consecutive_errors` | `6` | upstream returns N 4xx/5xx in a row (within `error_window_minutes`) |
| `max_requests_per_minute` | `30` | a `while True` without sleeps |

Spend is computed from the `usage` block of each response (including Anthropic cache reads and
writes and OpenAI cached tokens), streamed or not. If a provider omits usage, burnstop estimates
`chars / 4` and marks the entry `estimated`. Prices for current Claude and OpenAI models are
built in; add or override models under `pricing.models`. Unknown models fall back to a
deliberately expensive default.

## What happens when it trips

1. The response that crossed the line is delivered; the **next** request from the agent gets an
   immediate `402` with a provider-shaped error body (`{"type": "error", ...}` for Anthropic,
   `{"error": {...}}` for OpenAI) and an `X-Burnstop-Tripped: <reason>` header. 402 is used
   instead of 429 so official SDKs do not auto-retry.
2. One alert goes to each configured webhook (Telegram, Slack, Discord or generic JSON), not
   one per blocked request.
3. Optional kill switch: SIGTERM a PID from a file and/or run `on_trip_cmd`
   (for example `systemctl --user stop my-agent`), once.
4. The lockout is written to `~/.burnstop/state.json` and survives restarts. It stays until you
   run `burnstop reset` (or after `cooldown_minutes`, if you opt in).

## Commands

```text
burnstop init                  write a starter burnstop.yaml
burnstop start [-c FILE] [--port N]
burnstop status [--json]       ARMED / TRIPPED, reason, spend per window, last 5 requests
burnstop reset [--clear-session]
burnstop tail [-f]             request history (JSONL)
```

## Security

- `x-api-key`, `Authorization` and every other header are forwarded upstream untouched and are
  never written to logs, `state.json`, the history file, stdout or webhooks.
- Tool-call arguments are never stored. The loop detector keeps only the tool name and a
  SHA-256 fingerprint of the canonicalized arguments.
- Everything is local: two files in `~/.burnstop/`, no telemetry, no accounts.

## Limitations

- One proxy process, one state file. Run one burnstop per agent host.
- Prices are a static table. Check `burnstop/pricing.py` against your provider's pricing page.
- The loop detector catches exact repeats. An agent that loops with slightly different
  arguments each turn is caught by the spend windows instead.
- burnstop cannot see work billed outside the API calls that pass through it.

## Local CLI vs Cloud Pro

| | Open source CLI | Cloud Pro (early access) |
|---|---|---|
| Spend caps, loop guard, 402 kill switch | yes | yes |
| Local webhooks (Telegram / Slack / Discord) | yes | yes |
| External dead-man switch (alerts if the host hangs or OOMs) | no | yes |
| Multi-host fleet dashboard | no | yes |
| Remote one-click kill switch from your phone | no | yes |
| Price | $0, MIT | $19/mo |

Cloud Pro is opening in batches. Join the waitlist on the
[project page](https://kelbic.github.io/burnstop/).

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite is fully offline: a fake upstream serves Anthropic and OpenAI responses (JSON
and SSE) to a real burnstop instance.

## License

MIT
