import argparse
import json
from pathlib import Path
from typing import Any

from src.chains.basic import build_category_chain, build_category_prompt
from src.chains.search import ask_without_database, build_search_chain, execute_search
from src.chains.structured import (
    build_product_query_chain,
    build_product_query_parser,
    build_product_query_prompt,
)
from src.chains.thinking import compare_thinking
from src.chains.tool_calling import (
    build_search_tool,
    execute_validated_tool_call,
    request_tool_call,
)
from src.catalog import ProductQuery, ProductRepository
from src.config import ConfigurationError, create_chat_model, load_runtime_settings
from src.dialogue import StateDialogueSession, compare_dialogues
from src.retry import RetryingSearchPipeline, RetryExhaustedError, analyze_parser_errors, configure_logging
from src.retry.fakes import (
    CountingProductRepository,
    FailingChatModel,
    SequenceChatModel,
    always_invalid_model,
    invalid_then_valid_model,
)

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
    demo.add_argument("part", choices=["part1", "part2", "part3", "part4", "part5"])
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
    commands.add_parser("chat", help="Открыть интерактивный диалог с каталогом")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo" and args.part == "part4":
        result = _run_part4()
        _print_and_optionally_save(result, args.output)
        return 0

    try:
        settings = load_runtime_settings()
    except ConfigurationError as error:
        print(f"Ошибка конфигурации: {error}")
        return 2

    if args.command == "chat":
        return _run_chat(settings)

    try:
        if args.command == "search":
            result = _run_search(settings, args.query)
        elif args.part == "part1":
            result = _run_part1(settings)
        elif args.part == "part2":
            result = _run_part2(settings, args.reasoning_effort)
        elif args.part == "part3":
            result = _run_part3(settings)
        else:
            result = _run_part5(settings)
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


def _run_part4() -> dict[str, Any]:
    reports_path = Path("reports/results")
    log_path = Path("logs/pipeline.log")
    logger = configure_logging(log_path)
    parser_errors = analyze_parser_errors()
    _save_json(reports_path / "parser_errors.json", parser_errors)

    without_retry = invalid_then_valid_model()
    parser = build_product_query_parser()
    prompt = build_product_query_prompt(parser)
    try:
        response = without_retry.invoke(prompt.invoke({"query": "Ноутбуки до 70 тысяч"}))
        parser.invoke(response)
        no_retry_result: dict[str, Any] = {"error": None}
    except Exception as error:
        no_retry_result = {"error": type(error).__name__}

    successful_model = invalid_then_valid_model()
    successful_repository = CountingProductRepository()
    successful_pipeline = RetryingSearchPipeline(successful_model, successful_repository, logger)
    successful_response = successful_pipeline.invoke("Ноутбуки до 70 тысяч")

    failed_model = always_invalid_model()
    failed_repository = CountingProductRepository()
    failed_pipeline = RetryingSearchPipeline(failed_model, failed_repository, logger)
    try:
        failed_pipeline.invoke("Ноутбуки до 70 тысяч")
        exhausted_result: dict[str, Any] = {"error": None}
    except RetryExhaustedError as error:
        exhausted_result = {"error": str(error)}

    network_model = FailingChatModel(ConnectionError("offline"))
    network_repository = CountingProductRepository()
    network_pipeline = RetryingSearchPipeline(network_model, network_repository, logger)
    try:
        network_pipeline.invoke("Ноутбуки до 70 тысяч")
        network_result: dict[str, Any] = {"error": None}
    except ConnectionError as error:
        network_result = {"error": str(error)}

    other_model = SequenceChatModel(
        ['{"category":"other","max_price":null,"sort_by":"price_asc"}']
    )
    other_repository = CountingProductRepository()
    other_response = RetryingSearchPipeline(other_model, other_repository, logger).invoke(
        "Найди холодильники"
    )
    retry_demo = {
        "without_retry": {**no_retry_result, "llm_calls": without_retry.calls},
        "retry_success": {
            "llm_calls": successful_model.calls,
            "sqlite_calls": successful_repository.calls,
            "response": successful_response.to_dict(),
        },
        "retry_exhausted": {
            **exhausted_result,
            "llm_calls": failed_model.calls,
            "sqlite_calls": failed_repository.calls,
        },
        "network_error": {
            **network_result,
            "llm_calls": network_model.calls,
            "sqlite_calls": network_repository.calls,
        },
        "other_category": {
            "llm_calls": other_model.calls,
            "sqlite_calls": other_repository.calls,
            "response": other_response.to_dict(),
        },
    }
    _save_json(reports_path / "retry_demo.json", retry_demo)
    return {
        "part": "part4",
        "parser_errors_path": str(reports_path / "parser_errors.json"),
        "retry_demo_path": str(reports_path / "retry_demo.json"),
        "log_path": str(log_path),
        "retry_demo": retry_demo,
    }


def _run_part5(settings) -> dict[str, Any]:
    logger = configure_logging(Path("logs/pipeline.log"))
    comparison = compare_dialogues(
        create_chat_model(settings, reasoning_effort="none"),
        logger,
    )
    return {"part": "part5", "log_path": "logs/pipeline.log", **comparison.to_dict()}


def _run_chat(settings) -> int:
    logger = configure_logging(Path("logs/pipeline.log"))
    session = StateDialogueSession(
        RetryingSearchPipeline(
            create_chat_model(settings, reasoning_effort="none"),
            ProductRepository(),
            logger,
        )
    )
    print("Введите запрос, /state, /reset или /exit.")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            print()
            return 0
        if text == "/exit":
            return 0
        if text == "/state":
            value = session.current_query.model_dump(mode="json") if session.current_query else None
            print(json.dumps(value, ensure_ascii=False))
            continue
        if text == "/reset":
            session.reset()
            print("Состояние диалога очищено.")
            continue
        if not text:
            continue
        print(json.dumps(session.handle(text).to_dict(), ensure_ascii=False, indent=2))


def _print_and_optionally_save(result: dict[str, Any], output_path: Path | None) -> None:
    serialized = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    print(serialized)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"Результат сохранён: {output_path}")


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
