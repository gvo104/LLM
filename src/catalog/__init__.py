"""Доменная модель и доступ к каталогу товаров."""

from .models import Product, ProductQuery, UnsupportedCategoryError
from .repository import ProductRepository

__all__ = ["Product", "ProductQuery", "ProductRepository", "UnsupportedCategoryError"]
