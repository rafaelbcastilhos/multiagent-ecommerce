"""
Pré-processador de dados.

Aplica transformações, limpeza e enriquecimento
aos dados carregados.
"""

import logging
import re
from typing import Callable

import numpy as np
import pandas as pd

from src.data.schemas import Product, ProductCategory

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Pré-processador de dados de produtos.

    Aplica transformações e enriquecimento aos dados
    para melhorar a qualidade das análises.
    """

    def __init__(self) -> None:
        """Inicializa o pré-processador."""
        self._transformations: list[Callable[[Product], Product]] = []

    def add_transformation(
        self, transform_fn: Callable[[Product], Product]
    ) -> "DataPreprocessor":
        """
        Adiciona transformação ao pipeline.

        Args:
            transform_fn: Função de transformação.

        Returns:
            Self para encadeamento.
        """
        self._transformations.append(transform_fn)
        return self

    def process(self, products: list[Product]) -> list[Product]:
        """
        Aplica todas as transformações aos produtos.

        Args:
            products: Lista de produtos.

        Returns:
            Lista de produtos processados.
        """
        processed = []
        
        for product in products:
            try:
                result = product
                for transform in self._transformations:
                    result = transform(result)
                processed.append(result)
            except Exception as e:
                logger.warning(f"Erro ao processar {product.product_id}: {e}")
                processed.append(product)  # Mantém original

        return processed

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Limpa e normaliza texto.

        Args:
            text: Texto a limpar.

        Returns:
            Texto limpo.
        """
        if not text:
            return ""
        
        # Remove caracteres especiais excessivos
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        # Remove espaços múltiplos
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def normalize_price(price: float, currency_rate: float = 0.012) -> float:
        """
        Normaliza preço (INR para BRL aproximado).

        Args:
            price: Preço em rupias.
            currency_rate: Taxa de conversão.

        Returns:
            Preço normalizado.
        """
        # Conversão aproximada INR -> BRL
        return round(price * currency_rate, 2)

    @classmethod
    def create_default_pipeline(cls) -> "DataPreprocessor":
        """
        Cria pipeline com transformações padrão.

        Returns:
            Preprocessor configurado.
        """
        preprocessor = cls()
        
        # Transformação para limpar descrições
        def clean_descriptions(product: Product) -> Product:
            return Product(
                **{
                    **product.model_dump(),
                    "title": cls.clean_text(product.title),
                    "description": cls.clean_text(product.description),
                }
            )

        preprocessor.add_transformation(clean_descriptions)
        return preprocessor


