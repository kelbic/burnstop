"""burnstop.yaml loading, defaults, env overrides and validation.

Unknown keys are an error, not a warning: a typo in a limit name must not silently disable
the breaker it was meant to configure.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "listen": {"host": "127.0.0.1", "port": 8787},
    "upstreams": {"anthropic": "https://api.anthropic.com", "openai": "https://api.openai.com"},
    "limits": {"max_usd_per_10m": 2.0, "max_usd_per_hour": 5.0, "max_usd_per_day": 20.0,
               "max_usd_per_session": None},
    "loop_guard": {"max_identical_tool_calls": 4, "max_consecutive_errors": 6,
                   "error_window_minutes": 5, "max_requests_per_minute": 30},
    "lockout": {"trip_status_code": 402, "cooldown_minutes": None},
    "state_file": "~/.burnstop/state.json",
    "history_file": "~/.burnstop/history.jsonl",
    "inject_openai_stream_usage": True,
    "pricing": {"models": {}, "unknown_model_fallback_per_1m": {"input": 15.0, "output": 75.0}},
    "alerts": [],
    "kill_switch": {"kill_pid_file": None, "on_trip_cmd": None},
}

ALERT_FIELDS = {"telegram": {"bot_token", "chat_id"}, "slack": {"url"},
                "discord": {"url"}, "generic": {"url"}}
ENV = {"BURNSTOP_HOST": ("listen", "host"), "BURNSTOP_PORT": ("listen", "port"),
       "BURNSTOP_ANTHROPIC_UPSTREAM": ("upstreams", "anthropic"),
       "BURNSTOP_OPENAI_UPSTREAM": ("upstreams", "openai")}


class ConfigError(ValueError):
    pass


def _merge(base: dict, over: dict, path: str = "") -> dict:
    for key, val in over.items():
        where = f"{path}{key}"
        if key not in base:
            raise ConfigError(f"unknown config key: {where}")
        if isinstance(base[key], dict) and key not in ("models",):
            if not isinstance(val, dict):
                raise ConfigError(f"{where} must be a mapping")
            _merge(base[key], val, where + ".")
        else:
            base[key] = val
    return base


def _num(val, where: str, integer: bool = False, minimum: float = 0) -> None:
    if val is None:
        return
    ok = isinstance(val, int) if integer else isinstance(val, (int, float))
    if isinstance(val, bool) or not ok or val < minimum:
        kind = "an integer" if integer else "a number"
        raise ConfigError(f"{where} must be {kind} >= {minimum} or null")


def validate(cfg: dict) -> dict:
    for key, val in cfg["limits"].items():
        _num(val, f"limits.{key}")
    for key, val in cfg["loop_guard"].items():
        _num(val, f"loop_guard.{key}", integer=True, minimum=1)
    code = cfg["lockout"]["trip_status_code"]
    if not isinstance(code, int) or not 400 <= code <= 599:
        raise ConfigError("lockout.trip_status_code must be an HTTP status between 400 and 599")
    _num(cfg["lockout"]["cooldown_minutes"], "lockout.cooldown_minutes", minimum=0)
    _num(cfg["listen"]["port"], "listen.port", integer=True, minimum=1)
    for name, price in cfg["pricing"]["models"].items():
        if not isinstance(price, dict) or not {"input", "output"} <= set(price):
            raise ConfigError(f"pricing.models.{name} needs at least input and output")
    if not isinstance(cfg["alerts"], list):
        raise ConfigError("alerts must be a list")
    for i, alert in enumerate(cfg["alerts"]):
        kind = alert.get("type") if isinstance(alert, dict) else None
        if kind not in ALERT_FIELDS:
            raise ConfigError(f"alerts[{i}].type must be one of {sorted(ALERT_FIELDS)}")
        missing = ALERT_FIELDS[kind] - set(alert)
        if missing:
            raise ConfigError(f"alerts[{i}] ({kind}) is missing {sorted(missing)}")
    return cfg


def load(path: str | None = None) -> dict:
    """Defaults <- YAML file (if any) <- BURNSTOP_* env vars, then validate."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        raw = yaml.safe_load(Path(path).expanduser().read_text()) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
        _merge(cfg, raw)
    for var, keys in ENV.items():
        if os.environ.get(var):
            node = cfg
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = int(os.environ[var]) if keys[-1] == "port" else os.environ[var]
    if os.environ.get("BURNSTOP_STATE_DIR"):
        state_dir = Path(os.environ["BURNSTOP_STATE_DIR"])
        cfg["state_file"] = str(state_dir / "state.json")
        cfg["history_file"] = str(state_dir / "history.jsonl")
    for key in ("state_file", "history_file"):
        cfg[key] = str(Path(cfg[key]).expanduser())
    return validate(cfg)
