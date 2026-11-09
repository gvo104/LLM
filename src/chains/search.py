from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from src.catalog import Product, ProductQuery, ProductRepository
from src.prompt_loader import load_prompt

from .structured import build_product_query_chain


@dataclass
class SearchResponse:
    query: str
    product_query: ProductQuery
    products: list[Product]
    message: str
    sql: str | None = None
    sql_parameters: tuple[object, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "parsed_query": self.product_query.model_dump(mode="json"),
            "sql": self.sql,
            "sql_parameters": list(self.sql_parameters),
            "products": [product.model_dump(mode="json") for product in self.products],
            "message": self.message,
        }


def build_search_chain(llm, repository: ProductRepository):
    product_query_chain = RunnableLambda(lambda query: {"query": query}) | build_product_query_chain(llm)
    pipeline = RunnableParallel(
        query=RunnablePassthrough(),
        product_query=product_query_chain,
    )
    return pipeline | RunnableLambda(
        lambda data: execute_search(data["query"], data["product_query"], repository)
    )


def execute_search(
    query: str,
    product_query: ProductQuery,
    repository: ProductRepository,
) -> SearchResponse:
    if product_query.category == "other":
        return SearchResponse(
            query=query,
            product_query=product_query,
            products=[],
            message="Категория не поддерживается каталогом.",
        )

    sql, sql_parameters = repository.build_search_statement(product_query)
    products = repository.search(product_query)
    message = "Товары не найдены." if not products else f"Найдено товаров: {len(products)}."
    return SearchResponse(
        query=query,
        product_query=product_query,
        products=products,
        message=message,
        sql=sql,
        sql_parameters=sql_parameters,
    )


def build_without_database_prompt():
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages(
        [
            ("system", load_prompt("without_database/system.txt")),
            ("human", load_prompt("without_database/human.txt")),
        ]
    )


def ask_without_database(llm, query: str):
    return llm.invoke(build_without_database_prompt().invoke({"query": query}))
