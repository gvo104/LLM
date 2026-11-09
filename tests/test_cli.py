import json
from pathlib import Path

from src import cli
from src.config import RuntimeSettings


def test_cli_saves_request_error_when_output_is_requested(
    tmp_path: Path, monkeypatch
) -> None:
    output_path = tmp_path / "result.json"
    monkeypatch.setattr(
        cli,
        "load_runtime_settings",
        lambda: RuntimeSettings(base_url="http://localhost/v1", api_key="test"),
    )

    def rejected_request(_settings):
        raise RuntimeError("rejected")

    monkeypatch.setattr(cli, "_run_part1", rejected_request)

    exit_code = cli.main(["demo", "part1", "--output", str(output_path)])

    assert exit_code == 1
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "command": "part1",
        "error": "rejected",
    }


def test_cli_parses_search_command() -> None:
    args = cli.build_parser().parse_args(["search", "Покажи планшеты"])

    assert args.command == "search"
    assert args.query == "Покажи планшеты"


def test_cli_parses_chat_command() -> None:
    args = cli.build_parser().parse_args(["chat"])

    assert args.command == "chat"


def test_part4_runs_without_runtime_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def fail_if_called():
        raise AssertionError("part4 не должен читать настройки LLM")

    monkeypatch.setattr(cli, "load_runtime_settings", fail_if_called)

    assert cli.main(["demo", "part4"]) == 0
    assert (tmp_path / "reports/results/parser_errors.json").is_file()
    retry_demo_path = tmp_path / "reports/results/retry_demo.json"
    assert retry_demo_path.is_file()
    assert (tmp_path / "logs/pipeline.log").is_file()
    retry_demo = json.loads(retry_demo_path.read_text(encoding="utf-8"))
    assert retry_demo["without_retry"] == {
        "error": "OutputParserException",
        "llm_calls": 1,
    }
    assert retry_demo["retry_success"]["sqlite_calls"] == 1
    assert retry_demo["retry_exhausted"]["sqlite_calls"] == 0
    assert retry_demo["network_error"]["sqlite_calls"] == 0
    assert retry_demo["other_category"]["sqlite_calls"] == 0
