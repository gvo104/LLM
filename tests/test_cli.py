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
