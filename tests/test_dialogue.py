import logging
from pathlib import Path

from langchain_core.messages import AIMessage

from src.catalog import Product, ProductQuery
from src.dialogue import HistoryDialogueSession, StateDialogueSession, compare_dialogues
from src.retry import RetryingSearchPipeline, configure_logging
from src.retry.fakes import SequenceChatModel


class SpyRepository:
    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def build_search_statement(query: ProductQuery) -> tuple[str, tuple[object, ...]]:
        return "SELECT ... WHERE category = ?", (query.category,)

    def search(self, _query: ProductQuery) -> list[Product]:
        self.calls += 1
        return [Product(id=1, name="Товар", category="laptop", price=50_000)]


class ScriptedChatModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def invoke(self, _messages) -> AIMessage:
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return AIMessage(content=self.responses[index])


class UsageChatModel:
    def __init__(self, response: str) -> None:
        self.response = response

    def invoke(self, _messages) -> AIMessage:
        return AIMessage(
            content=self.response,
            usage_metadata={"input_tokens": 21, "output_tokens": 5, "total_tokens": 26},
        )


def _logger(tmp_path: Path) -> logging.Logger:
    return configure_logging(tmp_path / "pipeline.log")


def _pipeline(model, repository: SpyRepository, tmp_path: Path) -> RetryingSearchPipeline:
    return RetryingSearchPipeline(model, repository, _logger(tmp_path))


def test_state_session_preserves_unspecified_parameters(tmp_path: Path) -> None:
    responses = [
        '{"category":"laptop","max_price":80000,"sort_by":"price_asc"}',
        '{"category":"laptop","max_price":80000,"sort_by":"price_desc"}',
        '{"category":"smartphone","max_price":80000,"sort_by":"price_desc"}',
    ]
    repository = SpyRepository()
    session = StateDialogueSession(_pipeline(SequenceChatModel(responses), repository, tmp_path))

    first = session.handle("Покажи ноутбуки до 80 тысяч")
    second = session.handle("А сначала дорогие?")
    third = session.handle("А что есть среди смартфонов?")

    assert first.product_query == ProductQuery(category="laptop", max_price=80_000)
    assert second.product_query == ProductQuery(
        category="laptop", max_price=80_000, sort_by="price_desc"
    )
    assert third.product_query == ProductQuery(
        category="smartphone", max_price=80_000, sort_by="price_desc"
    )
    assert repository.calls == 3


def test_history_session_keeps_serialized_queries(tmp_path: Path) -> None:
    responses = [
        '{"category":"tablet","max_price":null,"sort_by":"price_asc"}',
        '{"category":"tablet","max_price":60000,"sort_by":"price_asc"}',
    ]
    session = HistoryDialogueSession(
        _pipeline(SequenceChatModel(responses), SpyRepository(), tmp_path)
    )

    session.handle("Покажи планшеты")
    second = session.handle("Только дешевле 60 тысяч")

    assert second.product_query == ProductQuery(category="tablet", max_price=60_000)
    assert len(session.history) == 4
    assert session.history[1].content == ProductQuery(category="tablet").model_dump_json()


def test_invalid_response_keeps_previous_state(tmp_path: Path) -> None:
    model = ScriptedChatModel(
        [
            '{"category":"laptop","max_price":80000,"sort_by":"price_asc"}',
            "не JSON",
        ]
    )
    repository = SpyRepository()
    session = StateDialogueSession(_pipeline(model, repository, tmp_path))

    session.handle("Ноутбуки до 80 тысяч")
    failed = session.handle("Сначала дорогие")

    assert "3 попыток" in failed.error
    assert session.current_query == ProductQuery(category="laptop", max_price=80_000)
    assert repository.calls == 1


def test_other_and_reset_do_not_search(tmp_path: Path) -> None:
    repository = SpyRepository()
    session = StateDialogueSession(
        _pipeline(
            SequenceChatModel(
                ['{"category":"other","max_price":null,"sort_by":"price_asc"}']
            ),
            repository,
            tmp_path,
        )
    )

    turn = session.handle("Холодильники")
    session.reset()

    assert turn.product_query == ProductQuery(category="other")
    assert repository.calls == 0
    assert session.current_query is None
    assert session.turn_number == 0


def test_turn_uses_reported_input_tokens(tmp_path: Path) -> None:
    model = UsageChatModel('{"category":"laptop","max_price":null,"sort_by":"price_asc"}')
    turn = StateDialogueSession(_pipeline(model, SpyRepository(), tmp_path)).handle("Ноутбуки")

    assert turn.input_tokens == 21
    assert turn.search_response.input_tokens == 21


def test_comparison_runs_both_dialogues(tmp_path: Path) -> None:
    queries = [
        '{"category":"laptop","max_price":80000,"sort_by":"price_asc"}',
        '{"category":"laptop","max_price":80000,"sort_by":"price_asc"}',
        '{"category":"laptop","max_price":80000,"sort_by":"price_desc"}',
        '{"category":"laptop","max_price":80000,"sort_by":"price_desc"}',
        '{"category":"smartphone","max_price":80000,"sort_by":"price_desc"}',
        '{"category":"smartphone","max_price":80000,"sort_by":"price_desc"}',
        '{"category":"tablet","max_price":null,"sort_by":"price_asc"}',
        '{"category":"tablet","max_price":null,"sort_by":"price_asc"}',
        '{"category":"tablet","max_price":60000,"sort_by":"price_asc"}',
        '{"category":"tablet","max_price":60000,"sort_by":"price_asc"}',
        '{"category":"laptop","max_price":60000,"sort_by":"price_asc"}',
        '{"category":"laptop","max_price":60000,"sort_by":"price_asc"}',
    ]

    comparison = compare_dialogues(SequenceChatModel(queries), _logger(tmp_path)).to_dict()

    assert [item["name"] for item in comparison["dialogues"]] == ["dialogue_a", "dialogue_b"]
    assert all(
        turn["state_correct"] and turn["history_correct"]
        for dialogue in comparison["dialogues"]
        for turn in dialogue["turns"]
    )
