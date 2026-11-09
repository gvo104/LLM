import sqlite3
from pathlib import Path

import pytest

from src.catalog import ProductQuery, ProductRepository, UnsupportedCategoryError
from src.catalog.models import CatalogDatabaseError
import src.catalog.repository as repository_module


@pytest.fixture
def catalog_database(tmp_path: Path) -> Path:
    database_path = tmp_path / "catalog.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price INTEGER NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO products (id, name, category, price) VALUES (?, ?, ?, ?)",
            [
                (1, "Ноутбук А", "laptop", 45_000),
                (2, "Ноутбук Б", "laptop", 70_000),
                (3, "Ноутбук В", "laptop", 70_000),
                (4, "Смартфон А", "smartphone", 20_000),
                (5, "Смартфон Б", "smartphone", 90_000),
                (6, "Планшет А", "tablet", 50_000),
            ],
        )
    return database_path


def test_search_filters_by_category_and_max_price(catalog_database: Path) -> None:
    repository = ProductRepository(catalog_database)

    products = repository.search(ProductQuery(category="laptop", max_price=70_000))

    assert [product.name for product in products] == ["Ноутбук А", "Ноутбук Б", "Ноутбук В"]


def test_search_orders_by_descending_price(catalog_database: Path) -> None:
    repository = ProductRepository(catalog_database)

    products = repository.search(ProductQuery(category="smartphone", sort_by="price_desc"))

    assert [product.price for product in products] == [90_000, 20_000]


def test_search_returns_an_empty_list_when_no_products_match(catalog_database: Path) -> None:
    repository = ProductRepository(catalog_database)

    assert repository.search(ProductQuery(category="laptop", max_price=40_000)) == []


def test_other_category_does_not_open_the_database(
    catalog_database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = ProductRepository(catalog_database)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("SQLite не должен вызываться для category='other'.")

    monkeypatch.setattr(repository_module.sqlite3, "connect", fail_if_called)

    with pytest.raises(UnsupportedCategoryError):
        repository.search(ProductQuery(category="other"))


def test_missing_catalog_is_reported(tmp_path: Path) -> None:
    repository = ProductRepository(tmp_path / "missing.sqlite3")

    with pytest.raises(CatalogDatabaseError, match="не найден"):
        repository.search(ProductQuery(category="tablet"))
