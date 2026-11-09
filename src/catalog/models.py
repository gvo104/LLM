"""Строгие модели параметров поиска и товаров каталога."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

ProductCategory = Literal["laptop", "smartphone", "tablet"]
QueryCategory = Literal["laptop", "smartphone", "tablet", "other"]
SortOrder = Literal["price_asc", "price_desc"]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class ProductQuery(BaseModel):
    """Проверенные параметры, по которым приложение ищет товары."""

    model_config = ConfigDict(extra="forbid")

    category: QueryCategory = Field(description="Категория товара")
    max_price: NonNegativeInt | None = Field(
        default=None,
        description="Максимальная цена в рублях; null означает отсутствие ограничения",
    )
    sort_by: SortOrder = Field(
        default="price_asc",
        description="Порядок сортировки: по возрастанию или убыванию цены",
    )


class Product(BaseModel):
    """Одна запись, полученная из локального каталога."""

    model_config = ConfigDict(extra="forbid")

    id: Annotated[int, Field(strict=True, ge=1)]
    name: str = Field(min_length=1)
    category: ProductCategory
    price: NonNegativeInt


class UnsupportedCategoryError(ValueError):
    """Запрос относится к товару, которого нет среди поддерживаемых категорий."""

    def __init__(self, category: str) -> None:
        super().__init__(f"Категория '{category}' не поддерживается каталогом.")
        self.category = category


class CatalogDatabaseError(RuntimeError):
    """Каталог отсутствует или не может быть прочитан как SQLite-база."""
