from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.catalog import ProductQuery
from src.chains.structured import build_product_query_parser
from src.prompt_loader import load_prompt

from .session import DialogueTurn, StateDialogueSession


def build_history_prompt(parser) -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", load_prompt("dialogue/history_system.txt")),
            MessagesPlaceholder("history"),
            ("human", load_prompt("dialogue/history_human.txt")),
        ]
    )
    return prompt.partial(format_instructions=parser.get_format_instructions())


class HistoryDialogueSession(StateDialogueSession):
    def __init__(self, pipeline) -> None:
        super().__init__(pipeline)
        self.history: list[HumanMessage | AIMessage] = []

    def handle(self, text: str) -> DialogueTurn:
        self.turn_number += 1
        previous_query = self.current_query
        parser = build_product_query_parser()
        try:
            response = self.pipeline.invoke_with_prompt(
                text,
                build_history_prompt(parser),
                {"history": self.history, "user_message": text},
                on_valid_query=lambda query: self._append_turn(text, query),
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
        super().reset()
        self.history.clear()

    def _append_turn(self, text: str, query: ProductQuery) -> None:
        self.current_query = query
        self.history.extend(
            [HumanMessage(content=text), AIMessage(content=query.model_dump_json())]
        )
