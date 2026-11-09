from dataclasses import dataclass
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.catalog import ProductQuery
from src.chains.search import SearchResponse
from src.chains.structured import build_product_query_parser
from src.prompt_loader import load_prompt
from src.retry import RetryingSearchPipeline


@dataclass
class DialogueTurn:
    number: int
    user_message: str
    previous_query: ProductQuery | None
    product_query: ProductQuery | None
    search_response: SearchResponse | None
    input_tokens: int | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "user_message": self.user_message,
            "previous_query": self.previous_query.model_dump(mode="json")
            if self.previous_query
            else None,
            "product_query": self.product_query.model_dump(mode="json")
            if self.product_query
            else None,
            "search_response": self.search_response.to_dict() if self.search_response else None,
            "input_tokens": self.input_tokens,
            "error": self.error,
        }


def build_state_prompt(parser) -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", load_prompt("dialogue/state_system.txt")),
            ("human", load_prompt("dialogue/state_human.txt")),
        ]
    )
    return prompt.partial(format_instructions=parser.get_format_instructions())


class StateDialogueSession:
    def __init__(self, pipeline: RetryingSearchPipeline) -> None:
        self.pipeline = pipeline
        self.current_query: ProductQuery | None = None
        self.turn_number = 0

    def handle(self, text: str) -> DialogueTurn:
        self.turn_number += 1
        previous_query = self.current_query
        parser = build_product_query_parser()
        prompt = build_state_prompt(parser)
        prompt_values = {
            "current_query": previous_query.model_dump_json() if previous_query else "null",
            "user_message": text,
        }
        try:
            response = self.pipeline.invoke_with_prompt(
                text,
                prompt,
                prompt_values,
                on_valid_query=self._set_current_query,
            )
        except Exception as error:
            return DialogueTurn(
                self.turn_number,
                text,
                previous_query,
                self.current_query,
                None,
                None,
                str(error),
            )
        return DialogueTurn(
            self.turn_number,
            text,
            previous_query,
            self.current_query,
            response,
            response.input_tokens,
            None,
        )

    def reset(self) -> None:
        self.current_query = None
        self.turn_number = 0

    def _set_current_query(self, query: ProductQuery) -> None:
        self.current_query = query
