import logging
from dataclasses import dataclass
from typing import Any

from src.catalog import ProductQuery, ProductRepository
from src.retry import RetryingSearchPipeline

from .history import HistoryDialogueSession
from .session import StateDialogueSession

_DIALOGUES = [
    (
        "dialogue_a",
        [
            ("Покажи ноутбуки до 80 тысяч", ProductQuery(category="laptop", max_price=80_000)),
            (
                "А сначала дорогие?",
                ProductQuery(category="laptop", max_price=80_000, sort_by="price_desc"),
            ),
            (
                "А что есть среди смартфонов?",
                ProductQuery(category="smartphone", max_price=80_000, sort_by="price_desc"),
            ),
        ],
    ),
    (
        "dialogue_b",
        [
            ("Покажи планшеты", ProductQuery(category="tablet")),
            ("Только дешевле 60 тысяч", ProductQuery(category="tablet", max_price=60_000)),
            ("Теперь ноутбуки", ProductQuery(category="laptop", max_price=60_000)),
        ],
    ),
]


@dataclass
class DialogueComparison:
    dialogues: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"dialogues": self.dialogues}


def compare_dialogues(llm, logger: logging.Logger) -> DialogueComparison:
    dialogues = []
    for name, turns in _DIALOGUES:
        state_session = StateDialogueSession(
            RetryingSearchPipeline(llm, ProductRepository(), logger)
        )
        history_session = HistoryDialogueSession(
            RetryingSearchPipeline(llm, ProductRepository(), logger)
        )
        results = []
        for text, expected in turns:
            state_turn = state_session.handle(text)
            history_turn = history_session.handle(text)
            results.append(
                {
                    "user_message": text,
                    "expected_query": expected.model_dump(mode="json"),
                    "state": state_turn.to_dict(),
                    "history": history_turn.to_dict(),
                    "state_correct": state_turn.product_query == expected,
                    "history_correct": history_turn.product_query == expected,
                }
            )
        dialogues.append({"name": name, "turns": results})
    return DialogueComparison(dialogues)
