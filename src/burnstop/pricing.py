"""Per-1M-token prices (USD) and request cost calculation.

Anthropic rows: Anthropic API pricing as of 2026-11 (cache_write = 5-minute TTL write; 1-hour
TTL writes are billed at 2x input). OpenAI rows: developers.openai.com/api/docs/pricing, checked
2026-11-20. Long-context tiers (OpenAI >272K, Haiku 5.5 >100K input) cost more and are not modeled.
Prices change - verify them and override or add models under `pricing.models` in burnstop.yaml.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("burnstop")


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float
    cache_write: float


def _p(inp: float, out: float, cache_read: float, cache_write: float | None = None) -> Price:
    return Price(inp, out, cache_read, inp * 1.25 if cache_write is None else cache_write)


BUILTIN: dict[str, Price] = {
    # Anthropic
    "claude-fable-5-2": _p(8, 40, 0.20), "claude-mythos-5-2": _p(8, 40, 0.20),
    "claude-fable-5-1": _p(10, 50, 0.25), "claude-mythos-5-1": _p(10, 50, 0.25),
    "claude-fable-5": _p(10, 50, 1.00), "claude-mythos-5": _p(10, 50, 1.00),
    "claude-opus-5-5": _p(4, 20, 0.20),
    "claude-opus-5": _p(5, 25, 0.50),
    "claude-opus-4-8": _p(5, 25, 0.50), "claude-opus-4-7": _p(5, 25, 0.50),
    "claude-opus-4-6": _p(5, 25, 0.50), "claude-opus-4-5": _p(5, 25, 0.50),
    "claude-opus-4-1": _p(15, 75, 1.50), "claude-opus-4": _p(15, 75, 1.50),
    "claude-sonnet-5-5": _p(2, 10, 0.10), "claude-sonnet-5": _p(2, 10, 0.20),
    "claude-sonnet-4": _p(3, 15, 0.30), "claude-3-7-sonnet": _p(3, 15, 0.30),
    "claude-haiku-5-5": _p(0.10, 0.50, 0.01),
    "claude-haiku-4-5": _p(1, 5, 0.10), "claude-3-5-haiku": _p(0.80, 4, 0.08),
    # OpenAI (no separate cache-write charge: cached input is reported as cached_tokens)
    "gpt-6.2-astra": _p(8, 40, 0.40, 8), "gpt-6.2-sol": _p(2, 10, 0.10, 2),
    "gpt-6.1-sol": _p(2, 10, 0.10, 2), "gpt-6.1-terra": _p(1, 5, 0.05, 1),
    "gpt-6-astra": _p(10, 50, 1, 10), "gpt-6-sol": _p(2, 10, 0.20, 2), "gpt-6-luna": _p(0.10, 0.50, 0.01, 0.10),
    "gpt-5.6-sol": _p(4, 20, 0.40, 4), "gpt-5.6-terra": _p(2, 12, 0.20, 2), "gpt-5.6-luna": _p(0.20, 1.20, 0.02, 0.20),
    "gpt-5.5-pro": _p(30, 180, 30, 30), "gpt-5.5": _p(5, 30, 0.50, 5),
    "gpt-5.4-pro": _p(30, 180, 30, 30), "gpt-5.4-mini": _p(0.75, 4.50, 0.075, 0.75),
    "gpt-5.4-nano": _p(0.20, 1.25, 0.02, 0.20), "gpt-5.4": _p(2.50, 15, 0.25, 2.50),
    "gpt-5.3-codex": _p(1.75, 14, 0.175, 1.75), "gpt-5.2-pro": _p(21, 168, 21, 21),
    "gpt-5.2": _p(1.75, 14, 0.175, 1.75), "gpt-5.1": _p(1.25, 10, 0.125, 1.25),
    "gpt-5-pro": _p(15, 120, 15, 15), "o3-pro": _p(20, 80, 20, 20),
    "gpt-5-nano": _p(0.05, 0.40, 0.005, 0.05), "gpt-5-mini": _p(0.25, 2, 0.025, 0.25),
    "gpt-5": _p(1.25, 10, 0.125, 1.25),
    "gpt-4.1-nano": _p(0.10, 0.40, 0.025, 0.10), "gpt-4.1-mini": _p(0.40, 1.60, 0.10, 0.40),
    "gpt-4.1": _p(2, 8, 0.50, 2),
    "gpt-4o-mini": _p(0.15, 0.60, 0.075, 0.15), "gpt-4o": _p(2.50, 10, 1.25, 2.50),
    "o4-mini": _p(1.10, 4.40, 0.275, 1.10), "o3-mini": _p(1.10, 4.40, 0.55, 1.10),
    "o3": _p(2, 8, 0.50, 2),
}


def _price_from(d: dict) -> Price:
    inp = float(d["input"])
    return Price(inp, float(d["output"]), float(d.get("cache_read", inp)),
                 float(d.get("cache_write", inp * 1.25)))


class Pricing:
    """Model -> Price with longest-prefix matching and a conservative fallback."""

    def __init__(self, overrides: dict | None = None, fallback: dict | None = None):
        self.table = dict(BUILTIN)
        for name, d in (overrides or {}).items():
            self.table[name.lower()] = _price_from(d)
        self.fallback = _price_from(fallback or {"input": 15, "output": 75})
        self._warned: set[str] = set()

    def lookup(self, model: str) -> tuple[Price, bool]:
        name = (model or "").lower().rsplit("/", 1)[-1]  # "anthropic/claude-..." routers
        # prefix must end at a version boundary: "gpt-5" prices "gpt-5-2025-08-07", not "gpt-5.7"
        best = max((k for k in self.table if name.startswith(k)
                    and name[len(k):len(k) + 1] in ("", "-", "@", ":", "_")), key=len, default=None)
        if best is not None:
            return self.table[best], True
        if name not in self._warned:
            self._warned.add(name)
            log.warning("unknown model %r: using fallback price $%.2f/$%.2f per 1M tokens",
                        model, self.fallback.input, self.fallback.output)
        return self.fallback, False

    def cost(self, usage) -> float:
        """USD for one parsed response (see parser.Usage)."""
        p, _ = self.lookup(usage.model)
        cw_5m = usage.cache_write_tokens - usage.cache_write_1h_tokens
        total = (usage.input_tokens * p.input + usage.output_tokens * p.output
                 + usage.cache_read_tokens * p.cache_read + cw_5m * p.cache_write
                 + usage.cache_write_1h_tokens * p.input * 2)
        return total / 1_000_000
