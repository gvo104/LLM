# Реализованный план лабораторной работы №1

Проект реализован последовательно: сначала подготовлены модели и SQLite,
затем LCEL-цепочки, поиск, retry и состояние диалога. Единый поток данных:

```text
Запрос → prompt → LLM → Pydantic ProductQuery → SQLite → товары
                         ↑ retry только здесь ↑
```

| Этап | Результат |
| --- | --- |
| 1. Домен и SQLite | `ProductQuery`, безопасный параметризованный поиск и каталог из 15 товаров. |
| 2. LCEL | Базовая и structured-цепочки, Pydantic parser, thinking-сравнение. |
| 3. Поиск и tools | Сквозной поиск, `search_products`, 10 контрольных запросов. |
| 4. Retry | Три попытки только для LLM/parser, журнал и offline fake-сценарии. |
| 5. Диалог | Compact state, полная история, интерактивный CLI и сравнение токенов. |
| 6. Финализация | Тесты, фактические JSON-артефакты и итоговый отчёт. |

## Основные решения

- LLM интерпретирует естественный язык, но не создаёт и не выполняет SQL.
- Pydantic валидирует параметры до доступа к SQLite.
- `category="other"` не выполняет поиск в каталоге.
- Обычные запросы используют `reasoning_effort="none"`; эксперимент thinking использует `low`.
- Секреты хранятся только в локальном `.env`.

## Проверка

```bash
conda activate llm-lab01
python -m pip check
python -m pytest -q
sqlite3 data/products.sqlite3 'PRAGMA integrity_check;'
```

Live-демонстрация через VPN или сеть университета:

```bash
python main.py
```
