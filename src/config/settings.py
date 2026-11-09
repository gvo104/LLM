from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping
from urllib.parse import urlparse

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI

from .model_config import (
    MAX_TOKENS,
    MODEL_NAME,
    TEMPERATURE,
    TIMEOUT_SECONDS,
    TRANSPORT_MAX_RETRIES,
)

ReasoningEffort = Literal["none", "low", "medium", "high"]
_ALLOWED_ENV_KEYS = {"OPENAI_BASE_URL", "OPENAI_API_KEY"}
_DEFAULT_DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class RuntimeSettings:
    base_url: str
    api_key: str = field(repr=False)


def load_runtime_settings(
    dotenv_path: str | Path | None = _DEFAULT_DOTENV_PATH,
    environ: Mapping[str, str] | None = None,
) -> RuntimeSettings:
    values = dotenv_values(dotenv_path) if dotenv_path is not None else {}
    unexpected_keys = set(values) - _ALLOWED_ENV_KEYS
    if unexpected_keys:
        keys = ", ".join(sorted(unexpected_keys))
        raise ConfigurationError(f"В .env разрешены только: {', '.join(sorted(_ALLOWED_ENV_KEYS))}. Найдены: {keys}.")

    source = dict(values)
    if environ is not None:
        source.update(environ)
    else:
        from os import environ as process_environ

        source.update(process_environ)

    base_url = source.get("OPENAI_BASE_URL", "").strip()
    api_key = source.get("OPENAI_API_KEY", "").strip()
    if not base_url:
        raise ConfigurationError("Не задан OPENAI_BASE_URL.")
    if not api_key:
        raise ConfigurationError("Не задан OPENAI_API_KEY.")

    parsed_url = urlparse(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ConfigurationError("OPENAI_BASE_URL должен быть HTTP(S)-адресом.")
    return RuntimeSettings(base_url=base_url.rstrip("/"), api_key=api_key)


def create_chat_model(
    settings: RuntimeSettings,
    reasoning_effort: ReasoningEffort = "none",
) -> ChatOpenAI:
    if reasoning_effort not in {"none", "low", "medium", "high"}:
        raise ValueError(f"Неподдерживаемый reasoning_effort: {reasoning_effort}")

    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT_SECONDS,
        max_retries=TRANSPORT_MAX_RETRIES,
        reasoning_effort=reasoning_effort,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": reasoning_effort != "none",
            }
        },
    )
