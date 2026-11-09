# Лабораторная работа №1: LangChain Pipelines

## Структура

```text
.
├── data/products.sqlite3     # каталог из 15 товаров
├── plan/README.md            # краткий план реализации
├── prompts/                  # промпты
├── reports/
│   ├── report.md             # итоговый отчёт
│   └── results/              # фактические JSON-артефакты
├── src/
│   ├── catalog/              # Pydantic-модели и SQLite
│   ├── chains/               # LCEL, structured output и tools
│   ├── config/               # настройки модели и .env
│   ├── dialogue/             # состояние и история диалога
│   └── retry/                # retry, fake-модели и журнал
├── tests/                    # offline unit и integration-тесты
├── main.py                   # единая live-демонстрация
└── requirements.txt
```

## Отчёт

Итоговый отчёт с результатами всех частей лабораторной находится в
[reports/report.md](reports/report.md). Исходные JSON-артефакты демонстраций
собраны в `reports/results/`.

## Окружение

```bash
conda create -n llm-lab01 python=3.11 -y
conda activate llm-lab01
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Локальная конфигурация

Создайте локальный `.env` из шаблона:

```bash
cp .env.example .env
```

В нём остаются только два значения:

```dotenv
OPENAI_BASE_URL=http://10.32.2.11:35000/v1
OPENAI_API_KEY=<персональный_токен>
```

Несекретные параметры модели находятся в
`src/config/model_config.py`.

## Локальные проверки

```bash
python -m pip check
python -m pytest -q
sqlite3 data/products.sqlite3 'PRAGMA integrity_check;'
```

## Online-запуск

Единая демонстрация
последовательно запускает части 1–5 и затем offline-тесты:

```bash
python main.py
```

Результаты сохраняются в `reports/results/`. При ошибке этапа скрипт
останавливается с ненулевым кодом; уже созданные результаты сохраняются.

Отдельные команды:

```bash
python -m src demo part1 --output reports/results/part1.json
python -m src demo part2 --reasoning-effort low --output reports/results/part2.json
python -m src demo part3 --output reports/results/part3.json
python -m src demo part4 --output reports/results/part4.json
python -m src demo part5 --output reports/results/dialogues.json
python -m src search "Покажи ноутбуки до 80 тысяч"
python -m src chat
```

`part4` работает offline. Команды `part1`, `part2`, `part3`, `part5`, `search`
и `chat` требуют доступ к LLM. Обычные цепочки передают
`reasoning_effort="none"`; thinking-сравнение использует выбранный уровень.

## Каталог

```bash
sqlite3 data/products.sqlite3 \
  'SELECT category, COUNT(*) FROM products GROUP BY category;'

sqlite3 data/products.sqlite3 \
  "SELECT name, price FROM products WHERE category = 'laptop' AND price <= 80000 ORDER BY price;"
```
