from dataclasses import asdict, dataclass
import json
from time import perf_counter
from typing import Any, Callable

from langchain_core.output_parsers import PydanticOutputParser

from src.catalog import ProductQuery
from src.config.model_config import DEFAULT_REASONING_EFFORT
from src.config.settings import ReasoningEffort, RuntimeSettings, create_chat_model

from .structured import build_product_query_parser, build_product_query_prompt


@dataclass
class ThinkingRun:
    effort: ReasoningEffort
    elapsed_seconds: float | None
    content: Any | None
    content_type: str | None
    parsed_query: dict[str, Any] | None
    parse_error: str | None
    request_error: str | None
    usage_metadata: dict[str, Any] | None
    additional_kwargs: dict[str, Any] | None
    response_metadata: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThinkingComparison:
    query: str
    without_reasoning: ThinkingRun
    with_reasoning: ThinkingRun

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "without_reasoning": self.without_reasoning.to_dict(),
            "with_reasoning": self.with_reasoning.to_dict(),
        }


ModelFactory = Callable[[RuntimeSettings, ReasoningEffort], Any]


def compare_thinking(
    query: str,
    settings: RuntimeSettings,
    reasoning_effort: ReasoningEffort | None = None,
    model_factory: ModelFactory = create_chat_model,
) -> ThinkingComparison:
    enabled_effort = reasoning_effort or DEFAULT_REASONING_EFFORT
    if enabled_effort == "none":
        raise ValueError("Для thinking-сравнения выберите low, medium или high.")

    parser = build_product_query_parser()
    prompt = build_product_query_prompt(parser)
    return ThinkingComparison(
        query=query,
        without_reasoning=_run_once(query, "none", settings, prompt, parser, model_factory),
        with_reasoning=_run_once(
            query, enabled_effort, settings, prompt, parser, model_factory
        ),
    )


def _run_once(
    query: str,
    effort: ReasoningEffort,
    settings: RuntimeSettings,
    prompt,
    parser: PydanticOutputParser,
    model_factory: ModelFactory,
) -> ThinkingRun:
    started_at = perf_counter()
    try:
        response = model_factory(settings, effort).invoke(prompt.invoke({"query": query}))
    except Exception as error:
        return ThinkingRun(
            effort=effort,
            elapsed_seconds=perf_counter() - started_at,
            content=None,
            content_type=None,
            parsed_query=None,
            parse_error=None,
            request_error=str(error),
            usage_metadata=None,
            additional_kwargs=None,
            response_metadata=None,
        )

    content = response.content
    parsed_query, parse_error = _parse_content(parser, content)
    return ThinkingRun(
        effort=effort,
        elapsed_seconds=perf_counter() - started_at,
        content=content,
        content_type=type(content).__name__,
        parsed_query=parsed_query,
        parse_error=parse_error,
        request_error=None,
        usage_metadata=_jsonable(response.usage_metadata),
        additional_kwargs=_jsonable(response.additional_kwargs),
        response_metadata=_jsonable(response.response_metadata),
    )


def _parse_content(
    parser: PydanticOutputParser, content: Any
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(content, str):
        return None, "content не является строкой и не передан в JSON-парсер."
    try:
        result: ProductQuery = parser.parse(content)
        return result.model_dump(mode="json"), None
    except Exception as error:
        return None, str(error)


def _jsonable(value: Any) -> dict[str, Any] | None:
    if not value:
        return None
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))
