"""
Agente comprador com perfil Ansioso.

Características: Preocupado com confiabilidade do produto, nota dos compradores,
informações completas. Busca segurança e transparência acima de tudo.
"""

from typing import ClassVar

from src.core.base_agent import (
    AgentPersonality,
    BaseAgent,
    EvaluationCriteria,
)
from src.data.schemas import BuyerProfile


class AnxiousBuyerAgent(BaseAgent):
    """
    Agente simulador de comprador ansioso.

    O comprador ansioso é caracterizado por:
    - Alta preocupação com segurança da compra
    - Necessidade de garantias e políticas de devolução claras
    - Valorização da reputação do vendedor
    - Sensibilidade a prazos de entrega
    - Busca por avaliações positivas de outros compradores
    """

    PROFILE: ClassVar[BuyerProfile] = BuyerProfile.ANXIOUS

    PERSONALITY: ClassVar[AgentPersonality] = AgentPersonality(
        profile=BuyerProfile.ANXIOUS,
        name="Comprador Ansioso",
        description="""Um comprador que prioriza segurança e confiabilidade.
        Tem medo de ser enganado ou receber um produto diferente do esperado.
        Verifica múltiplas vezes as informações antes de decidir e precisa
        de transparência total para se sentir confortável com a compra.""",
        priorities=[
            "Nota alta dos compradores (avaliação média do produto)",
            "Marca conhecida e confiável",
            "Descrição detalhada e transparente do produto",
            "Especificações técnicas completas (material, cor, composição)",
            "Foto do produto disponível para conferir o que está comprando",
        ],
        concerns=[
            "Produto com nota baixa ou sem avaliações",
            "Descrição vaga ou insuficiente",
            "Ausência de informações sobre material ou composição",
            "Produto sem foto disponível",
            "Preços muito baixos que parecem suspeitos",
        ],
        decision_style="""Analisa cuidadosamente a nota média do produto e a reputação
        da marca. Lê a descrição completa em busca de inconsistências. Verifica se
        todas as especificações estão presentes. Pode desistir da compra se houver
        qualquer sinal de falta de transparência nas informações.""",
        price_sensitivity=0.3,  # Aceita pagar mais por segurança
        quality_focus=0.7,
        risk_aversion=0.95,  # Altíssima aversão a risco
    )

    CRITERIA: ClassVar[EvaluationCriteria] = EvaluationCriteria(
        price_weight=0.10,
        quality_weight=0.15,
        reputation_weight=0.35,  # Credibilidade: marca + nota do produto
        delivery_weight=0.25,    # Completude dos atributos e apelo visual
        description_weight=0.10,
        promotion_weight=0.05,
    )

    def _calculate_dimension_weights(self) -> dict[str, float]:
        """
        Pesos específicos do perfil ansioso.

        Prioriza credibilidade (marca/nota) e completude das informações.
        """
        return {
            "price": self.CRITERIA.price_weight,
            "quality": self.CRITERIA.quality_weight,
            "reputation": self.CRITERIA.reputation_weight,
            "delivery": self.CRITERIA.delivery_weight,
            "description": self.CRITERIA.description_weight,
            "promotion": self.CRITERIA.promotion_weight,
        }

