import argparse
import json
from pathlib import Path
from typing import Any

from src.chains.basic import build_category_chain, build_category_prompt
from src.chains.structured import build_product_query_chain
from src.chains.thinking import compare_thinking
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Демонстрации лабораторной работы №1")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Запустить демонстрацию части лабораторной")
    demo.add_argument("part", choices=["part1", "part2"])
    demo.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="low",
        help="Уровень reasoning для thinking-сравнения",
    )
    demo.add_argument("--output", type=Path, help="Куда сохранить JSON-результат")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_runtime_settings()
    except ConfigurationError as error:
        print(f"Ошибка конфигурации: {error}")
        return 2

    try:
        result = _run_part1(settings) if args.part == "part1" else _run_part2(
            settings, args.reasoning_effort
        )
    except Exception as error:
        result = {"part": args.part, "error": str(error)}
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


def _print_and_optionally_save(result: dict[str, Any], output_path: Path | None) -> None:
    serialized = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    print(serialized)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"Результат сохранён: {output_path}")
