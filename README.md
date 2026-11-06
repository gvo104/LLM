# Лабораторная работа №1: LangChain Pipelines

Подготовка окружения и демонстрационного каталога товаров для лабораторной
работы по построению пайплайна LangChain и SQLite.

На текущем этапе подготовлены зависимости, база данных, файл локальных
настроек и структура каталогов. Код пайплайна, интеграция с LLM и отчёт будут
добавлены позднее.

## Структура

```text
.
├── .env                        # локальные параметры LLM и пустое поле токена
├── data/
│   └── products.sqlite3   # демонстрационный каталог товаров
├── notebooks/                  # эксперименты и демонстрации результатов
├── reports/                    # будущий отчёт и его вспомогательные материалы
├── src/
│   ├── chains/                 # части 1–2: промпты, LCEL и structured output
│   ├── catalog/                # часть 3: доступ к SQLite и инструмент поиска
│   ├── dialogue/               # часть 5: состояние и история диалога
│   ├── retry/                  # часть 4: retry, тестовый LLM и журнал
│   └── config/                 # чтение переменных окружения и настройки клиента
├── tests/                      # проверки частей 2–5 и тестовые сценарии retry
├── lab01_langchain_pipelines.md
├── README.md
└── requirements.txt
```

База `data/products.sqlite3` содержит таблицу `products` с полями `id`,
`name`, `category` и `price`. В каталоге есть ноутбуки, смартфоны и планшеты;
в каждой категории — минимум четыре товара.

### Связь структуры с заданием

| Каталог | Задания из `lab01_langchain_pipelines.md` | Будущее содержимое |
| --- | --- | --- |
| `src/chains/` | 1.1, 2.1–2.4 | Параметризованная LCEL-цепочка, Pydantic-схема и сравнение режимов размышления. |
| `src/catalog/` и `data/` | 3.1–3.4 | SQLite-каталог, безопасный программный поиск и инструмент `search_products`. |
| `src/retry/` | 4.1–4.4 | Retry цепочки, имитация невалидных ответов и журнал выполнения. |
| `src/dialogue/` | 5.1–5.2 | Сохранение последнего `ProductQuery` и вариант с историей диалога. |
| `src/config/` и `.env` | Подготовка и итоговые требования | Конфигурация OpenAI-совместимого клиента без секретов в коде. |
| `tests/` | 2.3, 3.3, 4.4, 5.2 | Воспроизводимые проверки запросов, поиска и retry. |
| `notebooks/` и `reports/` | Требования к сдаче | Эксперименты, таблицы сравнений и будущий отчёт. |

## Окружение

Для работы используется Conda-окружение с Python 3.11:

```bash
conda create -n llm-lab01 python=3.11 -y
conda activate llm-lab01
python -m pip install --upgrade pip
python -m pip install langchain langchain-core langchain-openai pydantic
python -m pip freeze > requirements.txt
```

Точные версии уже зафиксированы в `requirements.txt`. Чтобы воспроизвести
окружение, выполните:

```bash
conda activate llm-lab01
python -m pip install -r requirements.txt
```

## Локальная конфигурация и токен API

Шаблон `.env` содержит адрес университета, модель и пустое поле
`OPENAI_API_KEY`. Впишите токен только в этот локальный файл. Он исключён из
Git с помощью `.gitignore` и не должен попадать в исходный код, README или
отчёт.

Для запуска из оболочки экспортируйте значение из `.env` вручную либо задайте
переменную напрямую:

```bash
export OPENAI_API_KEY='<ваш_персональный_токен>'
```

Университетский OpenAI-совместимый сервер доступен только из сети университета.

## Проверка каталога

Проверить содержимое базы можно стандартным клиентом SQLite:

```bash
sqlite3 data/products.sqlite3 \
  'SELECT category, COUNT(*) AS count FROM products GROUP BY category;'

sqlite3 data/products.sqlite3 \
  "SELECT name, price FROM products WHERE category = 'laptop' AND price <= 80000 ORDER BY price;"
```
