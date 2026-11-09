import pytest

from src.prompt_loader import load_prompt


def test_load_prompt_reads_utf8_template() -> None:
    prompt = load_prompt("structured/system.txt")

    assert "{format_instructions}" in prompt
    assert "каталогу" in prompt


def test_load_prompt_rejects_missing_and_outside_files() -> None:
    with pytest.raises(FileNotFoundError, match="не найден"):
        load_prompt("missing.txt")
    with pytest.raises(ValueError, match="prompts"):
        load_prompt("../README.md")
