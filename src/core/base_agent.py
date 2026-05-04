"""
Classe base abstrata para agentes compradores.

Define a interface e comportamento comum para todos os
agentes de simulação de perfis psicológicos.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel, Field

from src.core.llm_client import OllamaClient
from src.data.schemas import (
    BuyerProfile,
    EvaluationDimension,
    Product,
    ProductEvaluation,
)

logger = logging.getLogger(__name__)


class AgentPersonality(BaseModel):
    """Define a personalidade e características do agente."""

    model_config = {"frozen": True}

    profile: BuyerProfile
    name: str
    description: str
    priorities: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    decision_style: str = ""
    price_sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    quality_focus: float = Field(default=0.5, ge=0.0, le=1.0)
    risk_aversion: float = Field(default=0.5, ge=0.0, le=1.0)


class EvaluationCriteria(BaseModel):
    """Critérios de avaliação com pesos específicos do perfil."""

    price_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    quality_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    reputation_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    delivery_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    description_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    promotion_weight: float = Field(default=0.1, ge=0.0, le=1.0)


class BaseAgent(ABC):
    """
    Classe base abstrata para agentes simuladores de compradores.

    Cada agente representa um perfil psicológico específico e avalia
    produtos de acordo com suas características comportamentais.

    Attributes:
        personality: Personalidade e características do agente.
        criteria: Critérios de avaliação com pesos.
        llm_client: Cliente para comunicação com LLM.
    """

    # Atributos de classe para serem sobrescritos
    PROFILE: ClassVar[BuyerProfile]
    PERSONALITY: ClassVar[AgentPersonality]
    CRITERIA: ClassVar[EvaluationCriteria]

    def __init__(self, llm_client: OllamaClient) -> None:
        """
        Inicializa o agente.

        Args:
            llm_client: Cliente Ollama para geração de avaliações.
        """
        self.llm_client = llm_client
        self._validate_class_attributes()

    def _validate_class_attributes(self) -> None:
        """Valida que atributos de classe estão definidos."""
        if not hasattr(self, "PROFILE"):
            raise NotImplementedError("Subclasse deve definir PROFILE")
        if not hasattr(self, "PERSONALITY"):
            raise NotImplementedError("Subclasse deve definir PERSONALITY")
        if not hasattr(self, "CRITERIA"):
            raise NotImplementedError("Subclasse deve definir CRITERIA")

    @property
    def profile(self) -> BuyerProfile:
        """Retorna o perfil do agente."""
        return self.PROFILE

    @property
    def personality(self) -> AgentPersonality:
        """Retorna a personalidade do agente."""
        return self.PERSONALITY

    @property
    def criteria(self) -> EvaluationCriteria:
        """Retorna os critérios de avaliação."""
        return self.CRITERIA

    def _build_system_prompt(self) -> str:
        """
        Constrói o prompt de sistema baseado na personalidade.

        Returns:
            Prompt de sistema para o LLM.
        """
        priorities_str = ", ".join(self.personality.priorities)
        concerns_str = ", ".join(self.personality.concerns)

        return f"""Você é um comprador online com o seguinte perfil psicológico:

**Perfil:** {self.personality.name}
**Descrição:** {self.personality.description}

**Suas principais prioridades ao comprar são:**
{priorities_str}

**Suas principais preocupações são:**
{concerns_str}

**Estilo de decisão:** {self.personality.decision_style}

**Características comportamentais:**
- Sensibilidade a preço: {self.personality.price_sensitivity * 100:.0f}%
- Foco em qualidade: {self.personality.quality_focus * 100:.0f}%
- Aversão a risco: {self.personality.risk_aversion * 100:.0f}%

Você deve avaliar produtos de forma consistente com este perfil, expressando opiniões e preocupações típicas de alguém com essas características.
"""

    def _build_evaluation_prompt(self, product: Product) -> str:
        """
        Constrói o prompt de avaliação do produto.

        Args:
            product: Produto a ser avaliado.

        Returns:
            Prompt formatado para avaliação.
        """
        seller_name = product.seller.name if product.seller else "Não informado"
        has_image = "Sim" if product.image_url else "Não"
        color_info = product.color if product.color else "Não informado"
        material_info = product.material if product.material else "Não informado"

        return f"""Avalie o seguinte produto como se você fosse um comprador real:

**Produto:** {product.title}
**Marca:** {product.brand if product.brand else "Não informada"}
**Vendedor:** {seller_name}
**Categoria:** {product.category.value}

**Preços:**
- Preço original: R$ {product.original_price:.2f}
- Preço com desconto: R$ {product.current_price:.2f}
- Desconto: {product.discount_percentage:.1f}%
- Tipo de promoção: {product.discount_type.value}

**Avaliação dos Compradores:**
- Nota média: {product.rating}/5.0

**Descrição do Produto:**
{product.description if product.description else "Sem descrição disponível"}

**Atributos do Produto:**
- Cor: {color_info}
- Material/Composição: {material_info}
- Foto disponível: {has_image}

---

Com base no seu perfil de comprador, forneça uma avaliação estruturada em JSON com:

1. **overall_score** (0-10): Nota geral do produto para você
2. **purchase_intention** (0-1): Probabilidade de comprar (0=nunca, 1=certamente)
3. **dimensions**: Lista de avaliações por dimensão:
   - price (Preço e custo-benefício considerando desconto aplicado)
   - quality (Qualidade percebida com base em marca, material e descrição)
   - reputation (Credibilidade da marca e nota média dos compradores)
   - delivery (Completude dos atributos e apelo visual do produto)
   - description (Qualidade e riqueza das informações fornecidas)
   - promotion (Atratividade e relevância do desconto oferecido)
4. **strengths**: Lista de pontos fortes do produto (máximo 5)
5. **weaknesses**: Lista de pontos fracos/preocupações (máximo 5)
6. **suggestions**: Sugestões para o vendedor melhorar (máximo 3)
7. **reasoning**: Seu raciocínio completo como comprador

Responda APENAS com JSON válido.
"""

    @abstractmethod
    def _calculate_dimension_weights(self) -> dict[str, float]:
        """
        Define os pesos específicos do perfil para cada dimensão.

        Returns:
            Dicionário com pesos por dimensão.
        """
        pass

    def evaluate(self, product: Product) -> ProductEvaluation:
        """
        Avalia um produto do ponto de vista deste perfil.

        Args:
            product: Produto a ser avaliado.

        Returns:
            ProductEvaluation com avaliação completa.

        Raises:
            ValueError: Se a avaliação falhar.
        """
        logger.info(f"[{self.personality.name}] Avaliando produto: {product.title[:50]}...")

        system_prompt = self._build_system_prompt()
        evaluation_prompt = self._build_evaluation_prompt(product)

        try:
            response = self.llm_client.generate(
                prompt=evaluation_prompt,
                system_prompt=system_prompt,
                format_json=True,
            )

            evaluation_data = self._parse_evaluation_response(response.content)
            
            return ProductEvaluation(
                evaluation_id=str(uuid.uuid4()),
                product_id=product.product_id,
                buyer_profile=self.profile,
                overall_score=evaluation_data.get("overall_score", 5.0),
                purchase_intention=evaluation_data.get("purchase_intention", 0.5),
                dimensions=self._build_dimensions(evaluation_data.get("dimensions", [])),
                strengths=evaluation_data.get("strengths", []),
                weaknesses=evaluation_data.get("weaknesses", []),
                suggestions=evaluation_data.get("suggestions", []),
                reasoning=evaluation_data.get("reasoning", ""),
                confidence=0.85,
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"[{self.personality.name}] Erro na avaliação: {e}")
            raise ValueError(f"Falha na avaliação: {e}") from e

    def _parse_evaluation_response(self, content: str) -> dict:
        """
        Parseia a resposta JSON do LLM.

        Args:
            content: Conteúdo da resposta.

        Returns:
            Dicionário com dados da avaliação.
        """
        import json

        try:
            # Tenta parsear diretamente
            return json.loads(content)
        except json.JSONDecodeError:
            # Tenta extrair JSON de texto misto
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError("Não foi possível extrair JSON da resposta")

    def _build_dimensions(
        self, dimensions_data: list[dict] | dict
    ) -> list[EvaluationDimension]:
        """
        Constrói lista de dimensões de avaliação.

        Args:
            dimensions_data: Dados das dimensões (lista ou dict).

        Returns:
            Lista de EvaluationDimension.
        """
        weights = self._calculate_dimension_weights()
        dimensions = []

        # Normaliza entrada para dicionário
        if isinstance(dimensions_data, list):
            dim_dict = {d.get("name", ""): d for d in dimensions_data}
        else:
            dim_dict = dimensions_data

        for name, weight in weights.items():
            dim_info = dim_dict.get(name, {})
            
            if isinstance(dim_info, dict):
                score = dim_info.get("score", 5.0)
                reasoning = dim_info.get("reasoning", "")
            else:
                score = float(dim_info) if dim_info else 5.0
                reasoning = ""

            dimensions.append(
                EvaluationDimension(
                    name=name,
                    score=min(max(score, 0.0), 10.0),
                    weight=weight,
                    reasoning=reasoning,
                )
            )

        return dimensions

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(profile={self.profile.value})"

