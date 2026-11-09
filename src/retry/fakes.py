from collections.abc import Sequence

from langchain_core.messages import AIMessage

from src.catalog import Product, ProductQuery, ProductRepository

_INVALID_JSON = "это не JSON"
_VALID_LAPTOP_QUERY = (
    '{"category":"laptop","max_price":70000,"sort_by":"price_asc"}'
)


class SequenceChatModel:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, _messages) -> AIMessage:
        response = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return AIMessage(content=response)


class FailingChatModel:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def invoke(self, _messages) -> AIMessage:
        self.calls += 1
        raise self.error


class CountingProductRepository(ProductRepository):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def search(self, query: ProductQuery) -> list[Product]:
        self.calls += 1
        return super().search(query)


def invalid_then_valid_model() -> SequenceChatModel:
    return SequenceChatModel([_INVALID_JSON, _VALID_LAPTOP_QUERY])


def always_invalid_model() -> SequenceChatModel:
    return SequenceChatModel([_INVALID_JSON])
