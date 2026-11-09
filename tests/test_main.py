from types import SimpleNamespace

import main


def test_main_runs_every_stage(monkeypatch, capsys) -> None:
    commands = []

    def run(command, check):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(main.subprocess, "run", run)

    assert main.main() == 0
    assert [command[2:5] for command in commands[:5]] == [
        ["src", "demo", "part1"],
        ["src", "demo", "part2"],
        ["src", "demo", "part3"],
        ["src", "demo", "part4"],
        ["src", "demo", "part5"],
    ]
    assert commands[-1][2:] == ["pytest", "-q"]
    assert "Демонстрация завершена" in capsys.readouterr().out


def test_main_stops_after_failed_stage(monkeypatch) -> None:
    calls = 0

    def run(_command, check):
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(main.subprocess, "run", run)

    assert main.main() == 7
    assert calls == 1
