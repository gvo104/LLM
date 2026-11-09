import logging
from uuid import uuid4

from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

from src.catalog import ProductQuery, ProductRepository
from src.chains.search import SearchResponse, execute_search
from src.chains.structured import build_product_query_parser, build_product_query_prompt

from .journal import log_event


class RetryExhaustedError(RuntimeError):
    pass


def analyze_parser_errors() -> list[dict[str, str]]:
    parser = build_product_query_parser()
    examples = [
        ("invalid_json", "это не JSON"),
        (
            "invalid_category",
            '{"category":"refrigerator","max_price":null,"sort_by":"price_asc"}',
        ),
        (
            "invalid_price",
            '{"category":"laptop","max_price":-1,"sort_by":"price_asc"}',
        ),
    ]
    results = []
    for name, response in examples:
        try:
            parser.parse(response)
        except Exception as error:
            cause = error.__cause__
            results.append(
                {
                    "scenario": name,
                    "response": response,
                    "error_type": type(error).__name__,
                    "cause": str(cause) if cause else "",
                }
            )
    return results


class RetryingSearchPipeline:
    def __init__(self, llm, repository: ProductRepository, logger: logging.Logger) -> None:
        self.llm = llm
        self.repository = repository
        self.logger = logger

    def invoke(self, user_query: str) -> SearchResponse:
        request_id = uuid4().hex
        attempts = 0
        parser = build_product_query_parser()
        prompt = build_product_query_prompt(parser)

        def mark_attempt(messages):
            nonlocal attempts
            attempts += 1
            log_event(self.logger, request_id, "llm", f"Начало попытки {attempts}.")
            return messages

        def call_model(messages):
            try:
                return self.llm.invoke(messages)
            except Exception as error:
                log_event(self.logger, request_id, "llm_error", type(error).__name__)
                raise

        def parse_response(response):
            try:
                result: ProductQuery = parser.invoke(response)
            except OutputParserException as error:
                log_event(self.logger, request_id, "parse_error", type(error).__name__)
                raise
            log_event(self.logger, request_id, "parse", "Параметры успешно проверены.")
            return result

        retrying_chain = (
            RunnableLambda(mark_attempt)
            | RunnableLambda(call_model)
            | RunnableLambda(parse_response)
        ).with_retry(
            retry_if_exception_type=(OutputParserException,),
            stop_after_attempt=3,
            wait_exponential_jitter=False,
        )

        try:
            product_query = (prompt | retrying_chain).invoke({"query": user_query})
        except OutputParserException as error:
            log_event(self.logger, request_id, "failed", "Попытки разбора исчерпаны.")
            raise RetryExhaustedError(
                "Не удалось распознать запрос после 3 попыток."
            ) from error

        if product_query.category == "other":
            log_event(self.logger, request_id, "sqlite", "Поиск пропущен: неподдерживаемая категория.")
            return execute_search(user_query, product_query, self.repository)

        log_event(self.logger, request_id, "sqlite", "Начало поиска.")
        response = execute_search(user_query, product_query, self.repository)
        log_event(self.logger, request_id, "sqlite", f"Найдено товаров: {len(response.products)}.")
        return response
