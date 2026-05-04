"""
Schemas Pydantic para modelagem de dados.

Define estruturas imutáveis para produtos, vendedores,
avaliações e resultados de análise.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ProductCategory(str, Enum):
    """Categorias de produtos fashion."""

    CLOTHING = "clothing"
    FOOTWEAR = "footwear"
    ACCESSORIES = "accessories"
    SPORTSWEAR = "sportswear"
    ETHNIC_WEAR = "ethnic_wear"
    WESTERN_WEAR = "western_wear"
    OTHER = "other"


class DiscountType(str, Enum):
    """Tipos de desconto disponíveis."""

    PERCENTAGE = "percentage"
    FIXED = "fixed"
    BUNDLE = "bundle"
    FLASH_SALE = "flash_sale"
    SEASONAL = "seasonal"
    NONE = "none"


class SellerInfo(BaseModel):
    """Informações do vendedor."""

    model_config = {"frozen": True}

    seller_id: str = Field(..., description="Identificador único do vendedor")
    name: str = Field(..., description="Nome do vendedor")
    rating: float = Field(default=0.0, ge=0.0, le=5.0, description="Avaliação média do vendedor")
    total_reviews: int = Field(default=0, ge=0, description="Total de avaliações recebidas")
    response_rate: float = Field(default=0.0, ge=0.0, le=100.0, description="Taxa de resposta (%)")
    ship_on_time: float = Field(default=0.0, ge=0.0, le=100.0, description="Taxa de entrega no prazo (%)")
    years_active: int = Field(default=0, ge=0, description="Anos de atividade")

    @property
    def reputation_score(self) -> float:
        """Calcula score de reputação ponderado."""
        return (
            self.rating * 0.4
            + (self.response_rate / 100) * 0.2
            + (self.ship_on_time / 100) * 0.3
            + min(self.years_active / 5, 1.0) * 0.1
        ) * 5


class Product(BaseModel):
    """Modelo de produto do marketplace."""

    model_config = {"frozen": True}

    product_id: str = Field(..., description="Identificador único do produto")
    title: str = Field(..., min_length=1, description="Título do produto")
    description: str = Field(default="", description="Descrição detalhada")
    brand: str = Field(default="", description="Marca do produto")
    category: ProductCategory = Field(default=ProductCategory.OTHER, description="Categoria")

    # Preços
    original_price: float = Field(..., gt=0, description="Preço original")
    current_price: float = Field(..., gt=0, description="Preço atual")
    discount_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Percentual de desconto")
    discount_type: DiscountType = Field(default=DiscountType.NONE, description="Tipo de desconto")

    # Avaliações
    rating: float = Field(default=0.0, ge=0.0, le=5.0, description="Avaliação média")
    review_count: int = Field(default=0, ge=0, description="Número de avaliações")

    # Atributos adicionais
    image_url: str = Field(default="", description="URL da imagem principal")
    color: str = Field(default="", description="Cor do produto")
    size_available: list[str] = Field(default_factory=list, description="Tamanhos disponíveis")
    material: str = Field(default="", description="Material/composição")
    
    # Informações de entrega
    free_shipping: bool = Field(default=False, description="Frete grátis disponível")
    estimated_delivery_days: int = Field(default=7, ge=1, description="Prazo estimado de entrega")
    return_policy_days: int = Field(default=7, ge=0, description="Prazo para devolução")

    # Vendedor
    seller: Optional[SellerInfo] = Field(default=None, description="Informações do vendedor")

    @field_validator("current_price")
    @classmethod
    def validate_current_price(cls, v: float, info) -> float:
        """Valida que preço atual não excede original."""
        original = info.data.get("original_price", v)
        if v > original:
            return original
        return v

    @property
    def savings(self) -> float:
        """Economia em valor absoluto."""
        return self.original_price - self.current_price

    @property
    def is_on_sale(self) -> bool:
        """Verifica se produto está em promoção."""
        return self.discount_percentage > 0 or self.current_price < self.original_price


class BuyerProfile(str, Enum):
    """Perfis psicológicos de compradores."""

    ANXIOUS = "anxious"  # Ansioso: preocupado com prazo, garantias, reputação
    DEMANDING = "demanding"  # Exigente: foco em qualidade, descrição detalhada
    ECONOMIC = "economic"  # Econômico: busca melhor custo-benefício
    IMPULSIVE = "impulsive"  # Impulsivo: atraído por promoções, urgência
    RATIONAL = "rational"  # Racional: análise equilibrada de todos os fatores


class EvaluationDimension(BaseModel):
    """Dimensão individual de avaliação."""

    model_config = {"frozen": True}

    name: str = Field(..., description="Nome da dimensão avaliada")
    score: float = Field(..., ge=0.0, le=10.0, description="Pontuação (0-10)")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Peso da dimensão")
    reasoning: str = Field(default="", description="Justificativa da pontuação")


class ProductEvaluation(BaseModel):
    """Avaliação completa de um produto por um perfil."""

    model_config = {"frozen": True}

    evaluation_id: str = Field(..., description="Identificador único da avaliação")
    product_id: str = Field(..., description="ID do produto avaliado")
    buyer_profile: BuyerProfile = Field(..., description="Perfil do comprador simulado")
    
    # Scores
    overall_score: float = Field(..., ge=0.0, le=10.0, description="Pontuação geral")
    purchase_intention: float = Field(..., ge=0.0, le=1.0, description="Intenção de compra (0-1)")
    
    # Dimensões detalhadas
    dimensions: list[EvaluationDimension] = Field(
        default_factory=list, description="Avaliações por dimensão"
    )
    
    # Feedback qualitativo
    strengths: list[str] = Field(default_factory=list, description="Pontos fortes identificados")
    weaknesses: list[str] = Field(default_factory=list, description="Pontos fracos identificados")
    suggestions: list[str] = Field(default_factory=list, description="Sugestões de melhoria")
    
    # Metadados
    reasoning: str = Field(default="", description="Raciocínio completo do agente")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confiança na avaliação")
    timestamp: datetime = Field(default_factory=datetime.now, description="Data/hora da avaliação")


class AggregatedEvaluation(BaseModel):
    """Avaliação agregada de múltiplos perfis."""

    model_config = {"frozen": True}

    product_id: str = Field(..., description="ID do produto")
    evaluations: list[ProductEvaluation] = Field(..., description="Avaliações individuais")
    
    # Métricas agregadas
    mean_score: float = Field(..., ge=0.0, le=10.0, description="Score médio entre perfis")
    std_score: float = Field(default=0.0, ge=0.0, description="Desvio padrão dos scores")
    min_score: float = Field(..., ge=0.0, le=10.0, description="Menor score")
    max_score: float = Field(..., ge=0.0, le=10.0, description="Maior score")
    
    # Análise de cobertura
    coverage_score: float = Field(
        ..., ge=0.0, le=1.0, 
        description="Score de cobertura (satisfação diversificada)"
    )
    profiles_satisfied: list[BuyerProfile] = Field(
        default_factory=list, description="Perfis satisfeitos (score >= 7)"
    )
    profiles_unsatisfied: list[BuyerProfile] = Field(
        default_factory=list, description="Perfis insatisfeitos (score < 5)"
    )
    
    # Consenso
    consensus_strengths: list[str] = Field(
        default_factory=list, description="Pontos fortes em consenso"
    )
    consensus_weaknesses: list[str] = Field(
        default_factory=list, description="Pontos fracos em consenso"
    )


class CustomerVector(BaseModel):
    """Posição de um cliente no espaço comportamental 3D."""

    model_config = {"frozen": True}

    price_sensitivity: float = Field(..., ge=0.0, le=1.0, description="Sensibilidade ao preço")
    quality_focus: float = Field(..., ge=0.0, le=1.0, description="Foco em qualidade")
    risk_aversion: float = Field(..., ge=0.0, le=1.0, description="Aversão ao risco")

    def to_tuple(self) -> tuple[float, float, float]:
        """Retorna o vetor como tupla (ps, qf, ra)."""
        return (self.price_sensitivity, self.quality_focus, self.risk_aversion)


class BlendedEvaluation(BaseModel):
    """Avaliação personalizada por blending de perfis."""

    model_config = {"frozen": True}

    customer_vector: CustomerVector = Field(..., description="Vetor do cliente no espaço 3D")
    profile_weights: dict[str, float] = Field(..., description="Pesos calculados por perfil")
    blended_score: float = Field(..., ge=0.0, le=10.0, description="Score final ponderado")
    blended_purchase_intention: float = Field(
        ..., ge=0.0, le=1.0, description="Intenção de compra ponderada"
    )
    blended_dimensions: list[EvaluationDimension] = Field(
        default_factory=list, description="Dimensões ponderadas"
    )
    dominant_profiles: list[str] = Field(
        default_factory=list, description="Perfis com maior peso"
    )
    source_evaluation: AggregatedEvaluation = Field(
        ..., description="Avaliação agregada original"
    )


class Recommendation(BaseModel):
    """Recomendação de otimização para o vendedor."""

    model_config = {"frozen": True}

    recommendation_id: str = Field(..., description="Identificador único")
    product_id: str = Field(..., description="Produto relacionado")
    
    # Classificação
    priority: int = Field(..., ge=1, le=5, description="Prioridade (1=urgente, 5=baixa)")
    category: str = Field(..., description="Categoria da recomendação")
    
    # Conteúdo
    title: str = Field(..., description="Título da recomendação")
    description: str = Field(..., description="Descrição detalhada")
    action_items: list[str] = Field(default_factory=list, description="Ações específicas")
    
    # Impacto esperado
    expected_impact: str = Field(..., description="Impacto esperado")
    affected_profiles: list[BuyerProfile] = Field(
        default_factory=list, description="Perfis que serão beneficiados"
    )
    estimated_improvement: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Melhoria estimada (%)"
    )

    # Metadados
    source_evaluations: list[str] = Field(
        default_factory=list, description="IDs das avaliações fonte"
    )

