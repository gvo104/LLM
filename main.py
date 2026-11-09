import subprocess
import sys


_STEPS = [
    (
        "Часть 1: базовая LCEL-цепочка",
        ["-m", "src", "demo", "part1", "--output", "reports/results/part1.json"],
    ),
    (
        "Часть 2: structured output и thinking",
        [
            "-m",
            "src",
            "demo",
            "part2",
            "--reasoning-effort",
            "low",
            "--output",
            "reports/results/part2.json",
        ],
    ),
    (
        "Часть 3: SQLite и tool calling",
        ["-m", "src", "demo", "part3", "--output", "reports/results/part3.json"],
    ),
    (
        "Часть 4: retry и журнал",
        ["-m", "src", "demo", "part4", "--output", "reports/results/part4.json"],
    ),
    (
        "Часть 5: состояние диалога",
        ["-m", "src", "demo", "part5", "--output", "reports/results/dialogues.json"],
    ),
    ("Offline-тесты", ["-m", "pytest", "-q"]),
]


def main() -> int:
    for title, arguments in _STEPS:
        print(f"\n=== {title} ===", flush=True)
        result = subprocess.run([sys.executable, *arguments], check=False)
        if result.returncode != 0:
            print(f"Демонстрация остановлена: {title} завершилась с кодом {result.returncode}.")
            return result.returncode

    print("\nДемонстрация завершена. Результаты сохранены в reports/results/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
