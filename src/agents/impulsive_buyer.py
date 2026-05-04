"""
Agente comprador com perfil Impulsivo.

Características: Atraído por promoções urgentes, compra por emoção,
sensível a gatilhos de escassez e urgência.
"""

from typing import ClassVar

from src.core.base_agent import (
    AgentPersonality,
    BaseAgent,
    EvaluationCriteria,
)
from src.data.schemas import BuyerProfile


class ImpulsiveBuyerAgent(BaseAgent):
    """
    Agente simulador de comprador impulsivo.

    O comprador impulsivo é caracterizado por:
    - Decisões rápidas baseadas em emoção
    - Atração por promoções e ofertas limitadas
    - Sensibilidade a gatilhos de urgência e escassez
    - Menor análise racional antes da compra
    - Influenciado por apresentação visual atraente
    """

    PROFILE: ClassVar[BuyerProfile] = BuyerProfile.IMPULSIVE

    PERSONALITY: ClassVar[AgentPersonality] = AgentPersonality(
        profile=BuyerProfile.IMPULSIVE,
        name="Comprador Impulsivo",
        description="""Um comprador que toma decisões rápidas baseadas
        em emoção e impulso. É facilmente atraído por promoções,
        descontos expressivos e produtos com apelo visual. Não gasta
        muito tempo analisando detalhes - se o produto parece bom e
        o desconto parece uma oportunidade, compra rapidamente.""",
        priorities=[
            "Descontos expressivos e visualmente impactantes (acima de 50%)",
            "Produto com foto atraente disponível",
            "Desconto alto em relação ao preço original",
            "Produto popular com boa nota dos compradores",
            "Título do produto chamativo e com apelo emocional",
        ],
        concerns=[
            "Produto sem foto disponível para visualização",
            "Desconto pequeno ou inexistente",
            "Descrição muito técnica e sem apelo emocional",
            "Nota baixa dos compradores",
            "Produto sem marca ou com aparência genérica",
        ],
        decision_style="""Decide rapidamente baseado na primeira impressão visual.
        Atraído por descontos grandes e marcantes. Não compara
        muito com alternativas. Compra por impulso quando vê uma 'oportunidade'.
        Influenciado por apresentação visual e percentual de desconto elevado.""",
        price_sensitivity=0.6,
        quality_focus=0.4,
        risk_aversion=0.25,  # Baixa aversão - compra sem pensar muito
    )

    CRITERIA: ClassVar[EvaluationCriteria] = EvaluationCriteria(
        price_weight=0.15,
        quality_weight=0.10,
        reputation_weight=0.10,  # Credibilidade: marca + nota do produto
        delivery_weight=0.20,    # Apelo visual e completude de atributos
        description_weight=0.10,
        promotion_weight=0.35,   # Máxima importância
    )

    def _calculate_dimension_weights(self) -> dict[str, float]:
        """
        Pesos específicos do perfil impulsivo.

        Prioriza promoção e apelo visual do produto.
        """
        return {
            "price": self.CRITERIA.price_weight,
            "quality": self.CRITERIA.quality_weight,
            "reputation": self.CRITERIA.reputation_weight,
            "delivery": self.CRITERIA.delivery_weight,
            "description": self.CRITERIA.description_weight,
            "promotion": self.CRITERIA.promotion_weight,
        }

