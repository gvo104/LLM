import argparse
import json
from pathlib import Path
from typing import Any

from src.chains.basic import build_category_chain, build_category_prompt
from src.chains.search import ask_without_database, build_search_chain, execute_search
from src.chains.structured import build_product_query_chain
from src.chains.thinking import compare_thinking
from src.chains.tool_calling import (
    build_search_tool,
    execute_validated_tool_call,
    request_tool_call,
)
from src.catalog import ProductQuery, ProductRepository
from src.config import ConfigurationError, create_chat_model, load_runtime_settings

_BASIC_QUERY = "Покажи недорогой ноутбук"
_BATCH_INPUTS = [
    {"query": "Ищу ноутбук для учёбы", "language": "русском"},
    {"query": "Нужен новый смартфон", "language": "русском"},
    {"query": "Хочу большой холодильник", "language": "русском"},
]
_STRUCTURED_QUERIES = [
    "Найди ноутбуки дешевле 70 000 рублей",
    "Покажи самые дорогие смартфоны",
    "Хочу посмотреть планшеты",
    "Покажи ноутбуки до 50 тысяч",
    "Найди холодильники",
    "Покажи что-нибудь недорогое для учёбы",
    "Хочу устройство до шестидесяти тысяч, сначала лучшие",
]
_AMBIGUOUS_QUERIES = _STRUCTURED_QUERIES[-2:]
_PART3_CASES = [
    ("Найди ноутбуки дешевле 70 000 рублей", ProductQuery(category="laptop", max_price=70_000)),
    (
        "Покажи самые дорогие смартфоны",
        ProductQuery(category="smartphone", sort_by="price_desc"),
    ),
    ("Хочу посмотреть планшеты", ProductQuery(category="tablet")),
    ("Покажи ноутбуки до 50 тысяч", ProductQuery(category="laptop", max_price=50_000)),
    ("Найди холодильники", ProductQuery(category="other")),
    ("Найди смартфоны до 30 тысяч", ProductQuery(category="smartphone", max_price=30_000)),
    (
        "Покажи дорогие планшеты до 100 тысяч",
        ProductQuery(category="tablet", max_price=100_000, sort_by="price_desc"),
    ),
    ("Найди ноутбуки дешевле 40 тысяч", ProductQuery(category="laptop", max_price=40_000)),
    ("Найди смартфоны дешевле 20 тысяч", ProductQuery(category="smartphone", max_price=20_000)),
    ("Покажи планшеты до 60 тысяч", ProductQuery(category="tablet", max_price=60_000)),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Демонстрации лабораторной работы №1")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Запустить демонстрацию части лабораторной")
    demo.add_argument("part", choices=["part1", "part2", "part3"])
    demo.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="low",
        help="Уровень reasoning для thinking-сравнения",
    )
    demo.add_argument("--output", type=Path, help="Куда сохранить JSON-результат")
    search = commands.add_parser("search", help="Найти товары по запросу")
    search.add_argument("query")
    search.add_argument("--output", type=Path, help="Куда сохранить JSON-результат")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_runtime_settings()
    except ConfigurationError as error:
        print(f"Ошибка конфигурации: {error}")
        return 2

    try:
        if args.command == "search":
            result = _run_search(settings, args.query)
        elif args.part == "part1":
            result = _run_part1(settings)
        elif args.part == "part2":
            result = _run_part2(settings, args.reasoning_effort)
        else:
            result = _run_part3(settings)
    except Exception as error:
        name = args.query if args.command == "search" else args.part
        result = {"command": name, "error": str(error)}
        _print_and_optionally_save(result, args.output)
        return 1

    _print_and_optionally_save(result, args.output)
    return 0


def _run_part1(settings) -> dict[str, Any]:
    llm = create_chat_model(settings, reasoning_effort="none")
    prompt = build_category_prompt()
    direct_response = llm.invoke(prompt.invoke({"query": _BASIC_QUERY, "language": "русском"}))
    chain = build_category_chain(llm)
    chain_result = chain.invoke({"query": _BASIC_QUERY, "language": "русском"})
    batch_result = chain.batch(_BATCH_INPUTS)
    return {
        "part": "part1",
        "direct_call": {
            "type": type(direct_response).__name__,
            "content": direct_response.content,
        },
        "chain_call": {"type": type(chain_result).__name__, "content": chain_result},
        "batch": batch_result,
    }


def _run_part2(settings, reasoning_effort: str) -> dict[str, Any]:
    llm = create_chat_model(settings, reasoning_effort="none")
    chain = build_product_query_chain(llm)
    structured_results = []
    for query in _STRUCTURED_QUERIES:
        try:
            product_query = chain.invoke({"query": query})
            structured_results.append(
                {
                    "query": query,
                    "json": product_query.model_dump(mode="json"),
                    "valid": True,
                }
            )
        except Exception as error:
            structured_results.append({"query": query, "valid": False, "error": str(error)})

    comparisons = [
        compare_thinking(query, settings, reasoning_effort=reasoning_effort).to_dict()
        for query in _AMBIGUOUS_QUERIES
    ]
    return {
        "part": "part2",
        "structured_queries": structured_results,
        "thinking_comparisons": comparisons,
    }


def _run_search(settings, query: str) -> dict[str, Any]:
    response = build_search_chain(
        create_chat_model(settings, reasoning_effort="none"), ProductRepository()
    ).invoke(query)
    return response.to_dict()


def _run_part3(settings) -> dict[str, Any]:
    repository = ProductRepository()
    llm = create_chat_model(settings, reasoning_effort="none")
    search_chain = build_search_chain(llm, repository)
    search_results = []
    for query, expected in _PART3_CASES:
        try:
            response = search_chain.invoke(query)
            search_results.append(
                {
                    "query": query,
                    "parsed_query": response.product_query.model_dump(mode="json"),
                    "sql": response.sql,
                    "sql_parameters": list(response.sql_parameters),
                    "products": [product.model_dump(mode="json") for product in response.products],
                    "correct": response.product_query == expected,
                    "message": response.message,
                }
            )
        except Exception as error:
            search_results.append({"query": query, "correct": False, "error": str(error)})

    without_database = []
    for query, expected in _PART3_CASES[:3]:
        answer = ask_without_database(llm, query)
        actual = execute_search(query, expected, repository)
        without_database.append(
            {
                "query": query,
                "model_answer": answer.content,
                "catalog_result": actual.to_dict(),
            }
        )

    search_tool = build_search_tool(repository)
    tool_calls = []
    for query, _ in _PART3_CASES[:5]:
        message = request_tool_call(llm, search_tool, query)
        execution = execute_validated_tool_call(message, repository)
        tool_calls.append(
            {
                "query": query,
                "content": message.content,
                "tool_calls": message.tool_calls,
                "execution": execution.to_dict(),
            }
        )

    return {
        "part": "part3",
        "search_results": search_results,
        "without_database": without_database,
        "tool_calls": tool_calls,
    }


def _print_and_optionally_save(result: dict[str, Any], output_path: Path | None) -> None:
    serialized = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    print(serialized)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"Результат сохранён: {output_path}")
