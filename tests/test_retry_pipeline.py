import logging
from pathlib import Path

import pytest

from src.catalog import Product, ProductQuery
from src.retry import RetryingSearchPipeline, RetryExhaustedError, analyze_parser_errors, configure_logging
from src.retry.fakes import FailingChatModel, SequenceChatModel, always_invalid_model, invalid_then_valid_model


class SpyRepository:
    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def build_search_statement(query: ProductQuery) -> tuple[str, tuple[object, ...]]:
        return "SELECT ... WHERE category = ?", (query.category,)

    def search(self, _query: ProductQuery) -> list[Product]:
        self.calls += 1
        return [Product(id=1, name="Товар", category="laptop", price=50_000)]


@pytest.fixture
def logger(tmp_path: Path) -> logging.Logger:
    return configure_logging(tmp_path / "pipeline.log")


def test_parser_error_analysis_has_three_invalid_responses() -> None:
    results = analyze_parser_errors()

    assert [item["scenario"] for item in results] == [
        "invalid_json",
        "invalid_category",
        "invalid_price",
    ]
    assert {item["error_type"] for item in results} == {"OutputParserException"}


def test_retry_runs_search_only_after_success(logger: logging.Logger) -> None:
    model = invalid_then_valid_model()
    repository = SpyRepository()

    response = RetryingSearchPipeline(model, repository, logger).invoke("Ноутбуки до 70 тысяч")

    assert model.calls == 2
    assert repository.calls == 1
    assert response.product_query.category == "laptop"


def test_retry_exhaustion_does_not_run_search(logger: logging.Logger) -> None:
    model = always_invalid_model()
    repository = SpyRepository()

    with pytest.raises(RetryExhaustedError, match="3 попыток"):
        RetryingSearchPipeline(model, repository, logger).invoke("Ноутбуки")

    assert model.calls == 3
    assert repository.calls == 0


def test_network_error_is_not_retried(logger: logging.Logger) -> None:
    model = FailingChatModel(ConnectionError("offline"))
    repository = SpyRepository()

    with pytest.raises(ConnectionError, match="offline"):
        RetryingSearchPipeline(model, repository, logger).invoke("Ноутбуки")

    assert model.calls == 1
    assert repository.calls == 0


def test_other_category_skips_repository(logger: logging.Logger) -> None:
    model = SequenceChatModel(
        ['{"category":"other","max_price":null,"sort_by":"price_asc"}']
    )
    repository = SpyRepository()

    response = RetryingSearchPipeline(model, repository, logger).invoke("Холодильники")

    assert model.calls == 1
    assert repository.calls == 0
    assert response.message == "Категория не поддерживается каталогом."


def test_journal_contains_stages_without_secret(tmp_path: Path) -> None:
    log_path = tmp_path / "pipeline.log"
    logger = configure_logging(log_path)
    RetryingSearchPipeline(invalid_then_valid_model(), SpyRepository(), logger).invoke("Ноутбуки")

    text = log_path.read_text(encoding="utf-8")
    assert "stage=llm" in text
    assert "stage=parse_error" in text
    assert "stage=parse" in text
    assert "stage=sqlite" in text
    assert "OPENAI_API_KEY" not in text
