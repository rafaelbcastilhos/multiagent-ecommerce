"""
Agente comprador com perfil Racional.

Características: Análise equilibrada de todos os fatores,
ponderação cuidadosa, decisão baseada em dados.
"""

from typing import ClassVar

from src.core.base_agent import (
    AgentPersonality,
    BaseAgent,
    EvaluationCriteria,
)
from src.data.schemas import BuyerProfile


class RationalBuyerAgent(BaseAgent):
    """
    Agente simulador de comprador racional.

    O comprador racional é caracterizado por:
    - Análise equilibrada de todos os fatores
    - Ponderação cuidadosa de prós e contras
    - Decisões baseadas em dados e fatos
    - Não se deixa influenciar por gatilhos emocionais
    - Busca o equilíbrio ótimo entre todos os critérios
    """

    PROFILE: ClassVar[BuyerProfile] = BuyerProfile.RATIONAL

    PERSONALITY: ClassVar[AgentPersonality] = AgentPersonality(
        profile=BuyerProfile.RATIONAL,
        name="Comprador Racional",
        description="""Um comprador metódico que analisa todos os aspectos
        de forma equilibrada antes de decidir. Não é dominado por nenhum
        critério específico - considera preço, qualidade, credibilidade da
        marca e completude das informações de forma ponderada. Imune a gatilhos
        emocionais de marketing. Toma decisões baseadas em análise objetiva.""",
        priorities=[
            "Equilíbrio entre preço e qualidade percebida",
            "Informações completas e consistentes para análise",
            "Marca com boa nota média dos compradores",
            "Descrição técnica com especificações do produto",
            "Desconto genuíno com base em valores absolutos reais",
        ],
        concerns=[
            "Falta de informações para análise completa",
            "Desequilíbrio significativo entre preço e qualidade",
            "Marca desconhecida sem nota dos compradores como referência",
            "Promoções que parecem manipulativas (desconto inflado)",
            "Inconsistências nas informações do produto",
        ],
        decision_style="""Analisa sistematicamente todos os aspectos disponíveis.
        Pondera preço, nota dos compradores, marca e completude da descrição.
        Não se apressa - prefere perder uma oferta a fazer má compra.
        Decisão final é resultado de análise racional baseada nos dados fornecidos.""",
        price_sensitivity=0.5,
        quality_focus=0.5,
        risk_aversion=0.5,
    )

    CRITERIA: ClassVar[EvaluationCriteria] = EvaluationCriteria(
        price_weight=0.20,       # Equilibrado
        quality_weight=0.20,     # Equilibrado
        reputation_weight=0.20,  # Credibilidade: marca + nota do produto
        delivery_weight=0.15,    # Completude dos atributos e apelo visual
        description_weight=0.15,
        promotion_weight=0.10,
    )

    def _calculate_dimension_weights(self) -> dict[str, float]:
        """
        Pesos específicos do perfil racional.

        Distribuição equilibrada entre todos os critérios disponíveis.
        """
        return {
            "price": self.CRITERIA.price_weight,
            "quality": self.CRITERIA.quality_weight,
            "reputation": self.CRITERIA.reputation_weight,
            "delivery": self.CRITERIA.delivery_weight,
            "description": self.CRITERIA.description_weight,
            "promotion": self.CRITERIA.promotion_weight,
        }

