from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.catalog import ProductQuery


def build_product_query_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=ProductQuery)


def build_product_query_prompt(
    parser: PydanticOutputParser | None = None,
) -> ChatPromptTemplate:
    output_parser = parser or build_product_query_parser()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты интерпретируешь запросы к каталогу товаров. Поддерживаются "
                "laptop, smartphone и tablet; неизвестные категории обозначай other. "
                "Если цена не указана, используй null. Если порядок не указан, "
                "используй price_asc. Верни только данные в заданном формате.\n"
                "{format_instructions}",
            ),
            ("human", "{query}"),
        ]
    )
    return prompt.partial(format_instructions=output_parser.get_format_instructions())


def build_product_query_chain(llm):
    parser = build_product_query_parser()
    return build_product_query_prompt(parser) | llm | parser
