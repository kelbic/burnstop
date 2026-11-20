"""Circuit-breaker rules: sliding spend windows, loop guard and a persistent lockout.

State lives in two local files: state.json (ARMED/TRIPPED, reason, session spend, streaks)
and history.jsonl (one line per metered response). Only tool names and argument hashes are
persisted - never arguments, headers or bodies.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

ARMED, TRIPPED = "ARMED", "TRIPPED"
WINDOWS = (("10m", 600, "max_usd_per_10m"), ("1h", 3600, "max_usd_per_hour"),
           ("day", 86400, "max_usd_per_day"))
HISTORY_COMPACT_BYTES = 5_000_000


class Breaker:
    def __init__(self, cfg: dict, clock=time.time):
        self.limits, self.guard = cfg["limits"], cfg["loop_guard"]
        self.cooldown = cfg["lockout"]["cooldown_minutes"]
        self.state_path, self.history_path = Path(cfg["state_file"]), Path(cfg["history_file"])
        self.clock = clock
        self.status, self.reason, self.tripped_at = ARMED, None, None
        self.session_started_at, self.session_spend = clock(), 0.0
        self.windows_since = 0.0  # reset() acknowledges the burn: windows restart from here
        self.tool_fp, self.tool_names, self.tool_count = None, [], 0
        self.error_count, self.last_error_ts = 0, 0.0
        self.spend_log: deque[tuple[float, float]] = deque()   # (ts, usd), last 24h
        self.req_times: deque[float] = deque()                  # pre-request timestamps
        self.recent: deque[dict] = deque(maxlen=5)
        self._mtime = None
        self._load()

    # ── persistence ────────────────────────────────────────────────────────────────────
    def _load(self) -> None:
        now = self.clock()
        if self.history_path.exists():
            lines = self.history_path.read_text().splitlines()
            for line in lines:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("ts", 0) > now - 86400:
                    self.spend_log.append((e["ts"], float(e.get("cost_usd", 0))))
                self.recent.append(e)
            if self.history_path.stat().st_size > HISTORY_COMPACT_BYTES:
                keep = [ln for ln in lines if '"ts"' in ln][-5000:]
                self.history_path.write_text("\n".join(keep) + "\n")
        self._read_state()

    def _read_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            st = json.loads(self.state_path.read_text())
        except ValueError:
            return
        self.status = st.get("status", ARMED)
        self.reason, self.tripped_at = st.get("reason"), st.get("tripped_at")
        self.session_started_at = st.get("session_started_at", self.session_started_at)
        self.session_spend = float(st.get("session_spend_usd", 0.0))
        self.windows_since = float(st.get("windows_since", 0.0))
        streak = st.get("streaks", {})
        self.tool_fp, self.tool_count = streak.get("tool_fp"), streak.get("tool_count", 0)
        self.tool_names = streak.get("tool_names", [])
        self.error_count = streak.get("error_count", 0)
        self.last_error_ts = streak.get("last_error_ts", 0.0)
        self._mtime = self._stamp()

    def _stamp(self):
        st = self.state_path.stat()  # inode changes on every os.replace: same-tick writes differ
        return st.st_mtime_ns, st.st_ino, st.st_size

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        st = {"status": self.status, "reason": self.reason, "tripped_at": self.tripped_at,
              "session_started_at": self.session_started_at,
              "session_spend_usd": round(self.session_spend, 6), "spend_usd": self.spend(),
              "windows_since": self.windows_since,
              "last_tool_calls": {"names": self.tool_names, "repeats": self.tool_count},
              "recent": list(self.recent),
              "streaks": {"tool_fp": self.tool_fp, "tool_names": self.tool_names,
                          "tool_count": self.tool_count, "error_count": self.error_count,
                          "last_error_ts": self.last_error_ts}}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=2))
        os.replace(tmp, self.state_path)
        self._mtime = self._stamp()

    def sync(self) -> None:
        """Pick up `burnstop reset` written by another process."""
        try:
            stamp = self._stamp()
        except FileNotFoundError:
            return
        if stamp != self._mtime:
            self._read_state()

    # ── accounting ─────────────────────────────────────────────────────────────────────
    def spend(self) -> dict:
        now = self.clock()
        while self.spend_log and self.spend_log[0][0] <= now - 86400:
            self.spend_log.popleft()
        out = {name: round(sum(c for ts, c in self.spend_log
                               if ts > max(now - secs, self.windows_since)), 6)
               for name, secs, _ in WINDOWS}
        out["session"] = round(self.session_spend, 6)
        return out

    def _budget_breach(self) -> str | None:
        spend = self.spend()
        checks = [(n, k) for n, _, k in WINDOWS] + [("session", "max_usd_per_session")]
        for name, key in checks:
            limit = self.limits.get(key)
            if limit is not None and spend[name] >= limit:
                return f"spend ${spend[name]:.2f} in {name} window reached {key} ${limit:.2f}"
        return None

    def trip(self, reason: str) -> bool:
        """-> True only on the ARMED -> TRIPPED transition (alerts fire exactly once)."""
        if self.status == TRIPPED:
            return False
        self.status, self.reason, self.tripped_at = TRIPPED, reason, self.clock()
        self.save()
        return True

    def precheck(self) -> tuple[str | None, bool]:
        """Before forwarding a metered request. -> (block reason or None, just tripped)."""
        self.sync()
        now = self.clock()
        if self.status == TRIPPED and self.cooldown is not None \
                and now - (self.tripped_at or now) >= self.cooldown * 60:
            self.reset()
        if self.status == TRIPPED:
            return self.reason, False
        while self.req_times and self.req_times[0] <= now - 60:
            self.req_times.popleft()
        rpm = self.guard.get("max_requests_per_minute")
        reason = self._budget_breach()
        if reason is None and rpm is not None and len(self.req_times) >= rpm:
            reason = f"{len(self.req_times) + 1} requests in 60s exceeds max_requests_per_minute {rpm}"
        if reason:
            return reason, self.trip(reason)
        self.req_times.append(now)
        return None, False

    def record(self, entry: dict) -> tuple[str | None, bool]:
        """After a metered response. entry: ts/cost_usd/status/tool_fps/tool_names/... ."""
        now = entry.setdefault("ts", self.clock())
        self.spend_log.append((now, entry["cost_usd"]))
        self.session_spend += entry["cost_usd"]
        self.recent.append(entry)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

        reason = self._budget_breach()
        fp = "|".join(sorted(entry.get("tool_fps") or [])) or None
        if fp is None:
            self.tool_fp, self.tool_names, self.tool_count = None, [], 0
        elif fp == self.tool_fp:
            self.tool_count += 1
        else:
            self.tool_fp, self.tool_names, self.tool_count = fp, entry.get("tool_names", []), 1
        n = self.guard.get("max_identical_tool_calls")
        if reason is None and n is not None and self.tool_count >= n:
            reason = f"identical tool call repeated {self.tool_count}x: {', '.join(self.tool_names)}"

        if entry.get("status", 200) >= 400:
            window = (self.guard.get("error_window_minutes") or 5) * 60
            self.error_count = self.error_count + 1 if now - self.last_error_ts <= window else 1
            self.last_error_ts = now
        else:
            self.error_count = 0
        m = self.guard.get("max_consecutive_errors")
        if reason is None and m is not None and self.error_count >= m:
            reason = f"{self.error_count} consecutive upstream errors (HTTP {entry['status']})"

        if reason:
            return reason, self.trip(reason)
        self.save()
        return None, False

    def reset(self, clear_session: bool = False) -> None:
        self.status, self.reason, self.tripped_at = ARMED, None, None
        self.tool_fp, self.tool_names, self.tool_count = None, [], 0
        self.error_count, self.last_error_ts = 0, 0.0
        self.req_times.clear()
        self.windows_since = self.clock()
        if clear_session:
            self.session_started_at, self.session_spend = self.clock(), 0.0
        self.save()
