import copy

import pytest

from burnstop.breaker import ARMED, TRIPPED, Breaker
from burnstop.config import DEFAULTS, validate


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def make(tmp_path):
    def _make(clock=None, **overrides):
        c = copy.deepcopy(DEFAULTS)
        c["state_file"], c["history_file"] = str(tmp_path / "s.json"), str(tmp_path / "h.jsonl")
        for section, values in overrides.items():
            c[section].update(values)
        return Breaker(validate(c), clock=clock or Clock())
    return _make


def entry(cost=0.0, status=200, tool=None):
    fps = [f"fp-{tool}"] if tool else []
    return {"cost_usd": cost, "status": status, "tool_fps": fps, "tool_names": [tool] if tool else []}


def test_10m_budget_trips(make):
    b = make()
    assert b.record(entry(0.9)) == (None, False)
    assert b.record(entry(0.9)) == (None, False)
    reason, tripped = b.record(entry(0.9))
    assert tripped and b.status == TRIPPED and "max_usd_per_10m" in reason
    assert b.precheck() == (reason, False)          # every later request is blocked, no 2nd alert


def test_hour_budget_trips_when_10m_window_does_not(make):
    clock = Clock()
    b = make(clock)
    for _ in range(3):
        assert b.record(entry(1.5))[0] is None
        clock.t += 700                                # outside the 10m window each time
    assert "max_usd_per_hour" in b.record(entry(1.5))[0]   # 4 x $1.50 = $6.00 within the hour


def test_identical_tool_calls_trip_on_nth(make):
    b = make()
    for _ in range(3):
        assert b.record(entry(tool="bash"))[0] is None
    reason, tripped = b.record(entry(tool="bash"))
    assert tripped and "repeated 4x" in reason and "bash" in reason


def test_different_or_missing_tool_call_breaks_the_streak(make):
    b = make()
    for tool in ("bash", "bash", "bash", "read", "bash", "bash", "bash", None, "bash", "bash", "bash"):
        assert b.record(entry(tool=tool))[0] is None


def test_consecutive_errors_trip_and_success_resets(make):
    clock = Clock()
    b = make(clock)
    for _ in range(5):
        assert b.record(entry(status=529))[0] is None
    assert b.record(entry(status=200))[0] is None   # success resets the streak
    for _ in range(5):
        assert b.record(entry(status=500))[0] is None
    assert "6 consecutive upstream errors" in b.record(entry(status=500))[0]


def test_errors_outside_window_do_not_accumulate(make):
    clock = Clock()
    b = make(clock)
    for _ in range(10):
        assert b.record(entry(status=500))[0] is None
        clock.t += 400                                # > 5 min apart


def test_requests_per_minute(make):
    clock = Clock()
    b = make(clock, loop_guard={"max_requests_per_minute": 30})
    for _ in range(30):
        assert b.precheck() == (None, False)
    reason, tripped = b.precheck()
    assert tripped and "max_requests_per_minute" in reason


def test_lockout_survives_restart_and_reset(make):
    b = make()
    for _ in range(4):
        b.record(entry(tool="bash"))
    assert b.status == TRIPPED
    again = make()
    assert again.status == TRIPPED and "repeated" in again.reason
    assert again.precheck()[0] is not None
    again.reset()
    assert make().status == ARMED and make().precheck() == (None, False)


def test_reset_restarts_spend_windows(make):
    b = make()
    b.record(entry(2.5))
    assert b.status == TRIPPED
    b.reset()
    assert b.precheck() == (None, False)              # burn acknowledged: windows restart
    assert b.spend()["session"] == 2.5                # session total kept...
    b.reset(clear_session=True)
    assert b.spend()["session"] == 0                  # ...unless cleared


def test_external_reset_is_picked_up_by_running_proxy(make):
    running = make()
    running.record(entry(5.0))
    assert running.precheck()[0] is not None
    make().reset()                                    # `burnstop reset` from another process
    assert running.precheck() == (None, False)


def test_cooldown_auto_rearms_only_when_configured(make):
    clock = Clock()
    b = make(clock, lockout={"cooldown_minutes": 30})
    b.record(entry(5.0))
    clock.t += 29 * 60
    assert b.precheck()[0] is not None
    clock.t += 2 * 60
    assert b.precheck() == (None, False)
    hard = make(clock)                                # default: cooldown null
    hard.trip("manual")
    clock.t += 10 * 86400
    assert hard.precheck()[0] == "manual"


def test_rules_can_be_disabled_with_null(make):
    b = make(limits={"max_usd_per_10m": None, "max_usd_per_hour": None, "max_usd_per_day": None},
             loop_guard={"max_identical_tool_calls": None})
    for _ in range(10):
        assert b.record(entry(10.0, tool="bash"))[0] is None
