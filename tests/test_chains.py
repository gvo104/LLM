import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from src.chains.basic import build_category_chain, build_category_prompt
from src.chains.structured import (
    build_product_query_chain,
    build_product_query_parser,
    build_product_query_prompt,
)
from src.chains.thinking import compare_thinking
from src.config import RuntimeSettings


def test_basic_chain_returns_a_string_while_direct_model_returns_ai_message() -> None:
    llm = FakeListChatModel(responses=["laptop"])
    prompt_value = build_category_prompt().invoke({"query": "Нужен ноутбук", "language": "русском"})

    direct_response = llm.invoke(prompt_value)
    chain_result = build_category_chain(llm).invoke(
        {"query": "Нужен ноутбук", "language": "русском"}
    )

    assert isinstance(direct_response, AIMessage)
    assert isinstance(chain_result, str)


def test_basic_chain_supports_batch() -> None:
    chain = build_category_chain(FakeListChatModel(responses=["laptop", "smartphone", "other"]))

    results = chain.batch(
        [
            {"query": "Ноутбук", "language": "русском"},
            {"query": "Телефон", "language": "русском"},
            {"query": "Холодильник", "language": "русском"},
        ]
    )

    assert sorted(results) == ["laptop", "other", "smartphone"]


def test_structured_chain_returns_product_query() -> None:
    llm = FakeListChatModel(
        responses=['{"category":"laptop","max_price":70000,"sort_by":"price_asc"}']
    )

    result = build_product_query_chain(llm).invoke({"query": "Ноутбуки до 70 тысяч"})

    assert result.category == "laptop"
    assert result.max_price == 70_000
    assert result.model_dump_json() == '{"category":"laptop","max_price":70000,"sort_by":"price_asc"}'


def test_structured_prompt_keeps_format_instructions_in_system_message() -> None:
    messages = build_product_query_prompt().invoke({"query": "Планшет"}).messages

    assert messages[0].type == "system"
    assert "JSON schema" in messages[0].content


def test_parser_rejects_invalid_model_response() -> None:
    parser = build_product_query_parser()

    with pytest.raises(OutputParserException):
        parser.parse('{"category":"camera","max_price":70000,"sort_by":"price_asc"}')


def test_thinking_comparison_keeps_none_and_selected_effort_separate() -> None:
    settings = RuntimeSettings(base_url="http://localhost/v1", api_key="test")

    def fake_factory(_settings, effort):
        return FakeListChatModel(
            responses=[f'{{"category":"tablet","max_price":null,"sort_by":"price_asc"}}']
        )

    comparison = compare_thinking(
        "Покажи планшеты", settings, reasoning_effort="low", model_factory=fake_factory
    )

    assert comparison.without_reasoning.effort == "none"
    assert comparison.with_reasoning.effort == "low"
    assert comparison.without_reasoning.parsed_query == {
        "category": "tablet",
        "max_price": None,
        "sort_by": "price_asc",
    }
