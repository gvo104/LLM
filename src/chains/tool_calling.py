from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from src.catalog import ProductQuery, ProductRepository
from src.prompt_loader import load_prompt

from .search import SearchResponse, execute_search


@dataclass
class ToolExecutionResult:
    message: str
    search_response: SearchResponse | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "search_response": self.search_response.to_dict() if self.search_response else None,
        }


def build_search_tool(repository: ProductRepository) -> BaseTool:
    @tool("search_products", args_schema=ProductQuery)
    def search_products(
        category: str,
        max_price: int | None = None,
        sort_by: str = "price_asc",
    ) -> str:
        response = execute_search(
            "",
            ProductQuery(category=category, max_price=max_price, sort_by=sort_by),
            repository,
        )
        return response.message

    return search_products


def request_tool_call(llm, search_tool: BaseTool, query: str) -> AIMessage:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", load_prompt("tool_calling/system.txt")),
            ("human", load_prompt("tool_calling/human.txt")),
        ]
    )
    return llm.bind_tools([search_tool]).invoke(prompt.invoke({"query": query}))


def execute_validated_tool_call(
    message: AIMessage,
    repository: ProductRepository,
) -> ToolExecutionResult:
    tool_calls = message.tool_calls
    if not tool_calls:
        return ToolExecutionResult("Модель не вызвала инструмент.", None)
    if len(tool_calls) != 1:
        return ToolExecutionResult("Ожидается ровно один вызов search_products.", None)

    tool_call = tool_calls[0]
    if tool_call["name"] != "search_products":
        return ToolExecutionResult("Модель вызвала неизвестный инструмент.", None)

    try:
        product_query = ProductQuery.model_validate(tool_call["args"])
    except ValidationError as error:
        return ToolExecutionResult(f"Аргументы инструмента невалидны: {error}", None)

    response = execute_search("", product_query, repository)
    return ToolExecutionResult(response.message, response)
