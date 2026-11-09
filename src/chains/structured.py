from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.catalog import ProductQuery
from src.prompt_loader import load_prompt


def build_product_query_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=ProductQuery)


def build_product_query_prompt(
    parser: PydanticOutputParser | None = None,
) -> ChatPromptTemplate:
    output_parser = parser or build_product_query_parser()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", load_prompt("structured/system.txt")),
            ("human", load_prompt("structured/human.txt")),
        ]
    )
    return prompt.partial(format_instructions=output_parser.get_format_instructions())


def build_product_query_chain(llm):
    parser = build_product_query_parser()
    return build_product_query_prompt(parser) | llm | parser
