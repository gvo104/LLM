from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.prompt_loader import load_prompt


def build_category_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", load_prompt("basic/system.txt")),
            ("human", load_prompt("basic/human.txt")),
        ]
    )


def build_category_chain(llm):
    return build_category_prompt() | llm | StrOutputParser()
