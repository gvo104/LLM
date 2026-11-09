import pytest
from pydantic import ValidationError

from src.catalog.models import Product, ProductQuery


@pytest.mark.parametrize("category", ["laptop", "smartphone", "tablet", "other"])
def test_product_query_accepts_supported_categories(category: str) -> None:
    query = ProductQuery(category=category)

    assert query.category == category
    assert query.max_price is None
    assert query.sort_by == "price_asc"


def test_product_query_accepts_a_strict_non_negative_price() -> None:
    query = ProductQuery(category="laptop", max_price=70_000, sort_by="price_desc")

    assert query.max_price == 70_000
    assert query.sort_by == "price_desc"


@pytest.mark.parametrize(
    "payload",
    [
        {"category": "laptop", "max_price": -1},
        {"category": "laptop", "max_price": "70000"},
        {"category": "camera"},
        {"category": "tablet", "sort_by": "name_asc"},
        {"category": "smartphone", "unexpected": True},
    ],
)
def test_product_query_rejects_invalid_input(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ProductQuery.model_validate(payload)


def test_product_rejects_negative_price() -> None:
    with pytest.raises(ValidationError):
        Product(id=1, name="Тестовый товар", category="laptop", price=-1)
