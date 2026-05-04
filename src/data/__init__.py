"""Módulo de dados - Pipeline de ingestão e processamento."""

from src.data.schemas import Product, ProductEvaluation, SellerInfo
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor

__all__ = [
    "Product",
    "ProductEvaluation",
    "SellerInfo",
    "DataLoader",
    "DataPreprocessor",
]

