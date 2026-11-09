from pathlib import Path

import pytest

from src.config import (
    ConfigurationError,
    RuntimeSettings,
    create_chat_model,
    load_runtime_settings,
)


def test_load_runtime_settings_reads_only_allowed_keys(tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "OPENAI_BASE_URL=http://localhost:35000/v1\nOPENAI_API_KEY=test-token\n",
        encoding="utf-8",
    )

    settings = load_runtime_settings(dotenv_path, environ={})

    assert settings.base_url == "http://localhost:35000/v1"
    assert "test-token" not in repr(settings)


def test_load_runtime_settings_rejects_extra_dotenv_keys(tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "OPENAI_BASE_URL=http://localhost/v1\nOPENAI_API_KEY=test\nLLM_MODEL=qwen\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="разрешены только"):
        load_runtime_settings(dotenv_path, environ={})


@pytest.mark.parametrize(
    "environ, message",
    [
        ({"OPENAI_API_KEY": "token"}, "OPENAI_BASE_URL"),
        ({"OPENAI_BASE_URL": "http://localhost/v1"}, "OPENAI_API_KEY"),
        (
            {"OPENAI_BASE_URL": "not-a-url", "OPENAI_API_KEY": "token"},
            "HTTP",
        ),
    ],
)
def test_load_runtime_settings_validates_required_values(
    environ: dict[str, str], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_runtime_settings(dotenv_path=None, environ=environ)


def test_model_factory_sets_effort_and_thinking_for_every_model() -> None:
    settings = RuntimeSettings(base_url="http://localhost:35000/v1", api_key="test")

    regular = create_chat_model(settings, reasoning_effort="none")
    reasoning = create_chat_model(settings, reasoning_effort="low")

    assert regular.reasoning_effort == "none"
    assert regular.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert regular.max_tokens == 512
    assert reasoning.reasoning_effort == "low"
    assert reasoning.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
