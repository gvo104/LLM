from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def build_category_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты определяешь категорию товара. Поддерживаемые категории: "
                "laptop, smartphone, tablet, other. Отвечай на языке {language}.",
            ),
            ("human", "{query}"),
        ]
    )


def build_category_chain(llm):
    return build_category_prompt() | llm | StrOutputParser()
