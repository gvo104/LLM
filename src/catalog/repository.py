"""Безопасный доступ к подготовленному SQLite-каталогу."""

from pathlib import Path
import sqlite3

from .models import (
    CatalogDatabaseError,
    Product,
    ProductQuery,
    UnsupportedCategoryError,
)

_SORT_DIRECTIONS = {"price_asc": "ASC", "price_desc": "DESC"}
_DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[2] / "data" / "products.sqlite3"


class ProductRepository:
    """Выполняет поиск только по валидированному :class:`ProductQuery`."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else _DEFAULT_DATABASE_PATH

    def search(self, query: ProductQuery) -> list[Product]:
        """Возвращает товары, подходящие под параметры поиска."""
        if not isinstance(query, ProductQuery):
            raise TypeError("search ожидает экземпляр ProductQuery.")
        if query.category == "other":
            raise UnsupportedCategoryError(query.category)
        if not self.db_path.is_file():
            raise CatalogDatabaseError(f"Файл каталога не найден: {self.db_path}")

        sql, parameters = self._build_query(query)
        try:
            with sqlite3.connect(self._read_only_uri(), uri=True) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(sql, parameters).fetchall()
        except sqlite3.DatabaseError as error:
            raise CatalogDatabaseError("Не удалось прочитать SQLite-каталог.") from error

        return [Product.model_validate(dict(row)) for row in rows]

    def _read_only_uri(self) -> str:
        return f"{self.db_path.resolve().as_uri()}?mode=ro"

    @staticmethod
    def _build_query(query: ProductQuery) -> tuple[str, tuple[object, ...]]:
        conditions = ["category = ?"]
        parameters: list[object] = [query.category]

        if query.max_price is not None:
            conditions.append("price <= ?")
            parameters.append(query.max_price)

        direction = _SORT_DIRECTIONS[query.sort_by]
        sql = (
            "SELECT id, name, category, price FROM products "
            f"WHERE {' AND '.join(conditions)} "
            f"ORDER BY price {direction}, id ASC"
        )
        return sql, tuple(parameters)
