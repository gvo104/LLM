import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.catalog import Product, ProductQuery
from src.chains.search import build_search_chain


class SpyRepository:
    def __init__(self, products: list[Product]) -> None:
        self.products = products
        self.calls = 0

    def search(self, query: ProductQuery) -> list[Product]:
        self.calls += 1
        return self.products

    @staticmethod
    def build_search_statement(query: ProductQuery) -> tuple[str, tuple[object, ...]]:
        return "SELECT ... WHERE category = ?", (query.category,)


def test_search_pipeline_runs_repository_after_valid_parsing() -> None:
    repository = SpyRepository([Product(id=1, name="Ноутбук", category="laptop", price=50_000)])
    llm = FakeListChatModel(
        responses=['{"category":"laptop","max_price":70000,"sort_by":"price_asc"}']
    )

    response = build_search_chain(llm, repository).invoke("Ноутбуки до 70 тысяч")

    assert repository.calls == 1
    assert response.product_query.category == "laptop"
    assert response.sql == "SELECT ... WHERE category = ?"
    assert response.sql_parameters == ("laptop",)
    assert response.products[0].name == "Ноутбук"


def test_search_pipeline_skips_repository_for_other_category() -> None:
    repository = SpyRepository([])
    llm = FakeListChatModel(
        responses=['{"category":"other","max_price":null,"sort_by":"price_asc"}']
    )

    response = build_search_chain(llm, repository).invoke("Холодильник")

    assert repository.calls == 0
    assert response.message == "Категория не поддерживается каталогом."


def test_search_pipeline_returns_empty_result_message() -> None:
    repository = SpyRepository([])
    llm = FakeListChatModel(
        responses=['{"category":"tablet","max_price":20000,"sort_by":"price_asc"}']
    )

    response = build_search_chain(llm, repository).invoke("Планшеты до 20 тысяч")

    assert repository.calls == 1
    assert response.products == []
    assert response.message == "Товары не найдены."


def test_search_pipeline_does_not_call_repository_after_parse_error() -> None:
    repository = SpyRepository([])
    llm = FakeListChatModel(responses=["не JSON"])

    with pytest.raises(OutputParserException):
        build_search_chain(llm, repository).invoke("Ноутбуки")

    assert repository.calls == 0
