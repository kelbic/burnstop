import json

import pytest

from burnstop.breaker import Breaker
from burnstop.cli import main
from burnstop.config import load


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BURNSTOP_STATE_DIR", str(tmp_path / "state"))
    return tmp_path


def test_init_writes_valid_config_and_refuses_overwrite(workdir, capsys):
    assert main(["init"]) == 0
    cfg = load("burnstop.yaml")
    assert cfg["loop_guard"]["max_identical_tool_calls"] == 4
    assert cfg["lockout"]["trip_status_code"] == 402
    assert main(["init"]) == 1
    assert main(["init", "--force"]) == 0


def test_status_armed_then_tripped(workdir, capsys):
    assert main(["status"]) == 0
    assert "ARMED" in capsys.readouterr().out
    b = Breaker(load(None))
    b.record({"cost_usd": 0.5, "status": 200, "tool_fps": [], "tool_names": [], "model": "claude-opus-5-5"})
    b.trip("identical tool call repeated 4x: bash")
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "TRIPPED" in out and "repeated 4x" in out and "claude-opus-5-5" in out
    assert main(["status", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "TRIPPED" and data["spend_usd"]["session"] == 0.5


def test_reset_and_clear_session(workdir, capsys):
    b = Breaker(load(None))
    b.record({"cost_usd": 9.0, "status": 200, "tool_fps": [], "tool_names": []})
    assert b.status == "TRIPPED"
    assert main(["reset"]) == 0 and "TRIPPED -> ARMED" in capsys.readouterr().out
    after = Breaker(load(None))
    assert after.status == "ARMED" and after.spend()["session"] == 9.0
    assert main(["reset", "--clear-session"]) == 0
    assert Breaker(load(None)).spend()["session"] == 0


def test_config_typo_is_an_error(workdir, capsys):
    (workdir / "burnstop.yaml").write_text("limits:\n  max_usd_per_hr: 5\n")
    assert main(["status"]) == 2
    assert "unknown config key: limits.max_usd_per_hr" in capsys.readouterr().err


def test_tail_prints_history(workdir, capsys):
    b = Breaker(load(None))
    b.record({"cost_usd": 0.1, "status": 200, "tool_fps": [], "tool_names": [], "model": "gpt-4o"})
    assert main(["tail", "-n", "5"]) == 0
    assert "gpt-4o" in capsys.readouterr().out
