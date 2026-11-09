from langchain_core.messages import AIMessage

from src.catalog import Product, ProductQuery
from src.chains.tool_calling import (
    build_search_tool,
    execute_validated_tool_call,
    request_tool_call,
)


class SpyRepository:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: ProductQuery) -> list[Product]:
        self.calls += 1
        return [Product(id=1, name="Товар", category="laptop", price=50_000)]

    @staticmethod
    def build_search_statement(query: ProductQuery) -> tuple[str, tuple[object, ...]]:
        return "SELECT ... WHERE category = ?", (query.category,)


class ToolCallingFake:
    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.tools = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, _prompt):
        return self.response


def _tool_call(name: str = "search_products", args: dict | None = None) -> dict:
    return {
        "name": name,
        "args": args or {"category": "laptop", "max_price": 70_000, "sort_by": "price_asc"},
        "id": "call_1",
    }


def test_search_tool_uses_product_query_schema() -> None:
    tool = build_search_tool(SpyRepository())

    assert tool.args_schema is ProductQuery
    assert "каталога" in tool.description


def test_request_tool_call_binds_single_search_tool() -> None:
    repository = SpyRepository()
    tool = build_search_tool(repository)
    llm = ToolCallingFake(AIMessage(content="", tool_calls=[_tool_call()]))

    response = request_tool_call(llm, tool, "Ноутбуки до 70 тысяч")

    assert response.tool_calls[0]["name"] == "search_products"
    assert [item.name for item in llm.tools] == ["search_products"]


def test_valid_tool_call_runs_repository_once() -> None:
    repository = SpyRepository()
    result = execute_validated_tool_call(AIMessage(content="", tool_calls=[_tool_call()]), repository)

    assert repository.calls == 1
    assert result.search_response is not None
    assert result.search_response.products[0].name == "Товар"


def test_invalid_tool_args_do_not_run_repository() -> None:
    repository = SpyRepository()

    for args in ({"category": "camera"}, {"category": "laptop", "max_price": "70000"}):
        message = AIMessage(content="", tool_calls=[_tool_call(args=args)])
        result = execute_validated_tool_call(message, repository)

        assert result.search_response is None
        assert "невалидны" in result.message

    assert repository.calls == 0


def test_missing_unknown_and_multiple_tool_calls_do_not_run_repository() -> None:
    repository = SpyRepository()
    cases = [
        AIMessage(content=""),
        AIMessage(content="", tool_calls=[_tool_call(name="other_tool")]),
        AIMessage(content="", tool_calls=[_tool_call(), _tool_call()]),
    ]

    for message in cases:
        result = execute_validated_tool_call(message, repository)
        assert result.search_response is None

    assert repository.calls == 0
