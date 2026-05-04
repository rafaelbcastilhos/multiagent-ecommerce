"""
Carregador de dados do dataset Kaggle.

Responsável por carregar e validar dados do dataset
Flipkart Fashion Products.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from src.data.schemas import (
    DiscountType,
    Product,
    ProductCategory,
    SellerInfo,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Carregador de dados do dataset Flipkart Fashion.

    Lê arquivos JSON/CSV e converte para modelos Pydantic,
    aplicando validações e transformações necessárias.

    Attributes:
        data_path: Caminho para o diretório de dados.
        dataset_file: Nome do arquivo do dataset.
    """

    # Mapeamento de categorias do dataset para enum
    CATEGORY_MAPPING: dict[str, ProductCategory] = {
        "clothing": ProductCategory.CLOTHING,
        "clothing and accessories": ProductCategory.CLOTHING,
        "footwear": ProductCategory.FOOTWEAR,
        "accessories": ProductCategory.ACCESSORIES,
        "sportswear": ProductCategory.SPORTSWEAR,
        "ethnic wear": ProductCategory.ETHNIC_WEAR,
        "western wear": ProductCategory.WESTERN_WEAR,
        "topwear": ProductCategory.CLOTHING,
        "bottomwear": ProductCategory.CLOTHING,
        "innerwear": ProductCategory.CLOTHING,
        "dress": ProductCategory.WESTERN_WEAR,
        "saree": ProductCategory.ETHNIC_WEAR,
        "kurta": ProductCategory.ETHNIC_WEAR,
        "shirts": ProductCategory.CLOTHING,
        "tshirts": ProductCategory.CLOTHING,
        "jeans": ProductCategory.CLOTHING,
        "trousers": ProductCategory.CLOTHING,
        "shoes": ProductCategory.FOOTWEAR,
        "sandals": ProductCategory.FOOTWEAR,
        "watches": ProductCategory.ACCESSORIES,
        "bags": ProductCategory.ACCESSORIES,
        "jewellery": ProductCategory.ACCESSORIES,
        "men": ProductCategory.CLOTHING,
        "women": ProductCategory.CLOTHING,
        "kids": ProductCategory.CLOTHING,
    }

    def __init__(
        self,
        data_path: Path | str = "data/raw",
        dataset_file: str = "flipkart_fashion_products_dataset.json",
    ) -> None:
        """
        Inicializa o carregador.

        Args:
            data_path: Caminho para diretório de dados.
            dataset_file: Nome do arquivo do dataset.
        """
        self.data_path = Path(data_path)
        self.dataset_file = dataset_file
        self._raw_data: list[dict] | None = None
        self._dataframe: pd.DataFrame | None = None

    @property
    def file_path(self) -> Path:
        """Caminho completo do arquivo."""
        return self.data_path / self.dataset_file

    def _load_json(self, force_reload: bool = False) -> list[dict]:
        """
        Carrega dados do arquivo JSON.

        Args:
            force_reload: Se True, recarrega mesmo se já em memória.

        Returns:
            Lista de dicionários com os dados.

        Raises:
            FileNotFoundError: Se arquivo não existe.
        """
        if self._raw_data is not None and not force_reload:
            return self._raw_data

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado: {self.file_path}\n"
                f"Baixe o dataset em: https://www.kaggle.com/datasets/aaditshukla/flipkart-fasion-products-dataset"
            )

        logger.info(f"Carregando dataset JSON: {self.file_path}")
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            self._raw_data = json.load(f)
        
        logger.info(f"Carregados {len(self._raw_data)} registros")
        return self._raw_data

    def load_dataframe(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Carrega dados em DataFrame pandas.

        Args:
            force_reload: Se True, recarrega mesmo se já em memória.

        Returns:
            DataFrame com dados do dataset.

        Raises:
            FileNotFoundError: Se arquivo não existe.
        """
        if self._dataframe is not None and not force_reload:
            return self._dataframe

        raw_data = self._load_json(force_reload)
        self._dataframe = pd.DataFrame(raw_data)
        
        logger.info(f"DataFrame criado com {len(self._dataframe)} linhas")
        return self._dataframe

    def _parse_price(self, price_str: str | float | int | None) -> float:
        """
        Parseia string de preço para float.

        Args:
            price_str: Preço em formato string ou numérico.

        Returns:
            Valor numérico do preço.
        """
        if price_str is None:
            return 0.0
        
        if isinstance(price_str, (int, float)):
            return float(price_str)
        
        # Remove símbolos de moeda, espaços e separadores de milhar
        cleaned = str(price_str).replace("₹", "").replace(",", "").replace(" ", "").strip()
        
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def _parse_discount(self, discount_str: str | None) -> float:
        """
        Parseia string de desconto para percentual.

        Args:
            discount_str: Desconto em formato "69% off" ou similar.

        Returns:
            Percentual de desconto.
        """
        if not discount_str:
            return 0.0
        
        # Extrai números da string (ex: "69% off" -> 69)
        match = re.search(r'(\d+(?:\.\d+)?)', str(discount_str))
        if match:
            return float(match.group(1))
        
        return 0.0

    def _parse_rating(self, rating_str: str | float | None) -> float:
        """
        Parseia string de rating para float.

        Args:
            rating_str: Rating em formato string ou numérico.

        Returns:
            Valor do rating (0-5).
        """
        if rating_str is None:
            return 0.0
        
        try:
            rating = float(str(rating_str).split()[0])
            return min(max(rating, 0.0), 5.0)
        except (ValueError, IndexError):
            return 0.0

    def _parse_category(self, category_str: str, sub_category_str: str = "") -> ProductCategory:
        """
        Mapeia strings de categoria para enum.

        Args:
            category_str: Categoria principal.
            sub_category_str: Subcategoria.

        Returns:
            ProductCategory correspondente.
        """
        # Tenta subcategoria primeiro (mais específica)
        for cat_str in [sub_category_str, category_str]:
            if not cat_str:
                continue
            
            cat_lower = str(cat_str).lower().strip()
            
            # Busca match direto
            if cat_lower in self.CATEGORY_MAPPING:
                return self.CATEGORY_MAPPING[cat_lower]
            
            # Busca match parcial
            for key, value in self.CATEGORY_MAPPING.items():
                if key in cat_lower or cat_lower in key:
                    return value
        
        return ProductCategory.OTHER

    def _parse_discount_type(self, discount: float) -> DiscountType:
        """Determina tipo de desconto baseado no percentual."""
        if discount <= 0:
            return DiscountType.NONE
        if discount >= 50:
            return DiscountType.FLASH_SALE
        if discount >= 30:
            return DiscountType.SEASONAL
        return DiscountType.PERCENTAGE

    def _extract_product_details(self, details: list[dict] | None) -> dict[str, str]:
        """
        Extrai detalhes do produto da lista de dicionários.

        Args:
            details: Lista no formato [{"Key": "Value"}, ...]

        Returns:
            Dicionário com detalhes extraídos.
        """
        if not details:
            return {}
        
        extracted = {}
        for item in details:
            if isinstance(item, dict):
                for key, value in item.items():
                    extracted[key.lower().strip()] = str(value).strip()
        
        return extracted

    def _record_to_product(self, record: dict[str, Any], idx: int) -> Product | None:
        """
        Converte registro do JSON para Product.

        Args:
            record: Registro do dataset.
            idx: Índice do registro.

        Returns:
            Product ou None se inválido.
        """
        try:
            # Extrai preços
            actual_price = self._parse_price(record.get("actual_price"))
            selling_price = self._parse_price(record.get("selling_price"))
            
            # Garante preços válidos
            if actual_price <= 0:
                actual_price = selling_price if selling_price > 0 else 100.0
            if selling_price <= 0:
                selling_price = actual_price

            # Calcula desconto
            discount_str = record.get("discount", "")
            discount_pct = self._parse_discount(discount_str)
            
            # Se desconto não veio no campo, calcula
            if discount_pct == 0 and actual_price > selling_price:
                discount_pct = ((actual_price - selling_price) / actual_price) * 100

            # Extrai detalhes do produto
            product_details = self._extract_product_details(record.get("product_details"))
            
            # Extrai cor e material dos detalhes
            color = product_details.get("color", "")
            material = product_details.get("fabric", product_details.get("material", ""))
            
            # Extrai imagens
            images = record.get("images", [])
            image_url = images[0] if images else ""

            # Cria informações do vendedor
            seller_name = record.get("seller", "")
            seller_info = None
            if seller_name:
                seller_info = SellerInfo(
                    seller_id=f"SELLER_{hash(seller_name) % 100000:05d}",
                    name=str(seller_name)[:100],
                    rating=4.0,  # Valor padrão (não disponível no dataset)
                    total_reviews=100,
                    response_rate=90.0,
                    ship_on_time=95.0,
                    years_active=2,
                )

            # Cria produto
            product = Product(
                product_id=str(record.get("pid", record.get("_id", f"PROD_{idx:06d}"))),
                title=str(record.get("title", "Produto sem título"))[:500],
                description=str(record.get("description", ""))[:2000],
                brand=str(record.get("brand", ""))[:100],
                category=self._parse_category(
                    record.get("category", ""),
                    record.get("sub_category", ""),
                ),
                original_price=actual_price,
                current_price=selling_price,
                discount_percentage=min(discount_pct, 100.0),
                discount_type=self._parse_discount_type(discount_pct),
                rating=self._parse_rating(record.get("average_rating")),
                review_count=0,  # Não disponível no dataset
                image_url=str(image_url),
                color=color[:50],
                size_available=[],  # Não disponível diretamente
                material=material[:100],
                free_shipping=False,  # Não disponível no dataset
                estimated_delivery_days=7,
                return_policy_days=7,
                seller=seller_info,
            )
            return product

        except ValidationError as e:
            logger.warning(f"Validação falhou para registro {idx}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Erro ao processar registro {idx}: {e}")
            return None

    def load_products(
        self,
        limit: int | None = None,
        category_filter: ProductCategory | None = None,
        min_rating: float | None = None,
        only_discounted: bool = False,
    ) -> list[Product]:
        """
        Carrega produtos do dataset.

        Args:
            limit: Limite de produtos a carregar.
            category_filter: Filtrar por categoria específica.
            min_rating: Rating mínimo para filtrar.
            only_discounted: Apenas produtos com desconto.

        Returns:
            Lista de produtos validados.
        """
        raw_data = self._load_json()
        products = []
        
        for idx, record in enumerate(raw_data):
            if limit and len(products) >= limit:
                break
            
            # Filtro de desconto
            if only_discounted:
                discount = self._parse_discount(record.get("discount", ""))
                if discount <= 0:
                    continue
            
            # Filtro de rating
            if min_rating is not None:
                rating = self._parse_rating(record.get("average_rating"))
                if rating < min_rating:
                    continue
            
            product = self._record_to_product(record, idx)
            if product is None:
                continue
            
            # Filtro de categoria
            if category_filter and product.category != category_filter:
                continue
            
            products.append(product)

        logger.info(f"Carregados {len(products)} produtos válidos")
        return products

