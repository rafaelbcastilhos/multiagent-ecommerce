"""
Agente comprador com perfil Exigente.

Características: Foco em qualidade, detalhes do produto, marca.
Não aceita menos que o melhor, mesmo pagando mais.
"""

from typing import ClassVar

from src.core.base_agent import (
    AgentPersonality,
    BaseAgent,
    EvaluationCriteria,
)
from src.data.schemas import BuyerProfile


class DemandingBuyerAgent(BaseAgent):
    """
    Agente simulador de comprador exigente.

    O comprador exigente é caracterizado por:
    - Foco absoluto em qualidade e acabamento
    - Exigência de descrições detalhadas e precisas
    - Preferência por marcas reconhecidas
    - Atenção a detalhes como material, composição
    - Expectativas altas em todos os aspectos
    """

    PROFILE: ClassVar[BuyerProfile] = BuyerProfile.DEMANDING

    PERSONALITY: ClassVar[AgentPersonality] = AgentPersonality(
        profile=BuyerProfile.DEMANDING,
        name="Comprador Exigente",
        description="""Um comprador que busca excelência em todos os aspectos.
        Não se contenta com produtos medianos e está disposto a pagar mais
        pela melhor qualidade. Analisa cada detalhe da descrição e das fotos.
        Valoriza marcas premium e materiais de primeira linha.""",
        priorities=[
            "Qualidade premium do produto",
            "Marca reconhecida e confiável",
            "Descrição completa com especificações técnicas",
            "Informações sobre material e composição do produto",
            "Nota alta dos compradores como sinal de qualidade",
        ],
        concerns=[
            "Produtos genéricos ou sem marca",
            "Descrições vagas ou incompletas",
            "Falta de informações sobre material/composição",
            "Nota baixa dos compradores indicando problemas de qualidade",
            "Produto sem foto disponível para análise visual",
        ],
        decision_style="""Analisa minuciosamente cada especificação do produto.
        Compara a marca com alternativas premium. Verifica nota média como indicador
        de qualidade percebida. Exige descrição completa com material e atributos.
        Prefere investir mais em qualidade comprovada.""",
        price_sensitivity=0.2,  # Disposto a pagar por qualidade
        quality_focus=0.95,     # Foco máximo em qualidade
        risk_aversion=0.6,
    )

    CRITERIA: ClassVar[EvaluationCriteria] = EvaluationCriteria(
        price_weight=0.05,       # Preço é secundário
        quality_weight=0.40,     # Máxima importância
        reputation_weight=0.15,  # Credibilidade: marca + nota do produto
        delivery_weight=0.10,    # Completude dos atributos e apelo visual
        description_weight=0.25, # Muito importante
        promotion_weight=0.05,
    )

    def _calculate_dimension_weights(self) -> dict[str, float]:
        """
        Pesos específicos do perfil exigente.

        Prioriza qualidade percebida e riqueza da descrição.
        """
        return {
            "price": self.CRITERIA.price_weight,
            "quality": self.CRITERIA.quality_weight,
            "reputation": self.CRITERIA.reputation_weight,
            "delivery": self.CRITERIA.delivery_weight,
            "description": self.CRITERIA.description_weight,
            "promotion": self.CRITERIA.promotion_weight,
        }

