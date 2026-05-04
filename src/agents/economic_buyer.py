"""
Agente comprador com perfil Econômico.

Características: Busca o melhor custo-benefício, sensível a preços,
analisa descontos reais versus inflados, valoriza promoções genuínas.
"""

from typing import ClassVar

from src.core.base_agent import (
    AgentPersonality,
    BaseAgent,
    EvaluationCriteria,
)
from src.data.schemas import BuyerProfile


class EconomicBuyerAgent(BaseAgent):
    """
    Agente simulador de comprador econômico.

    O comprador econômico é caracterizado por:
    - Alta sensibilidade a preços
    - Busca constante por custo-benefício
    - Análise crítica de descontos (identifica descontos falsos)
    - Comparação com produtos similares
    - Valorização de frete grátis e promoções reais
    """

    PROFILE: ClassVar[BuyerProfile] = BuyerProfile.ECONOMIC

    PERSONALITY: ClassVar[AgentPersonality] = AgentPersonality(
        profile=BuyerProfile.ECONOMIC,
        name="Comprador Econômico",
        description="""Um comprador racional que busca maximizar o valor
        de cada real gasto. Não é necessariamente o mais barato, mas sim
        o melhor custo-benefício. Analisa criticamente promoções para
        identificar descontos reais vs. artificiais. Compara o preço
        com desconto em relação ao preço original para validar a oferta.""",
        priorities=[
            "Melhor relação custo-benefício",
            "Descontos reais e significativos sobre o preço original",
            "Preço de venda abaixo do preço original de forma expressiva",
            "Qualidade adequada compatível com o preço",
            "Promoções genuínas (desconto alto e marca conhecida)",
        ],
        concerns=[
            "Preços originais inflados artificialmente para simular desconto",
            "Desconto percentual alto mas preço absoluto ainda elevado",
            "Produtos muito baratos de qualidade duvidosa",
            "Promoções enganosas sem redução real de valor",
            "Nota baixa indicando má relação custo-benefício",
        ],
        decision_style="""Analisa a proporção entre preço original e preço com desconto.
        Verifica se o desconto é genuíno comparando valores absolutos.
        Avalia a nota do produto para confirmar custo-benefício.
        Compra apenas quando identifica uma oferta claramente vantajosa.""",
        price_sensitivity=0.95,  # Máxima sensibilidade a preço
        quality_focus=0.5,       # Qualidade adequada, não premium
        risk_aversion=0.5,
    )

    CRITERIA: ClassVar[EvaluationCriteria] = EvaluationCriteria(
        price_weight=0.40,       # Máxima importância
        quality_weight=0.15,
        reputation_weight=0.10,  # Credibilidade: marca + nota do produto
        delivery_weight=0.15,    # Completude dos atributos e apelo visual
        description_weight=0.05,
        promotion_weight=0.15,   # Promoções são importantes
    )

    def _calculate_dimension_weights(self) -> dict[str, float]:
        """
        Pesos específicos do perfil econômico.

        Prioriza preço, promoção e credibilidade do produto.
        """
        return {
            "price": self.CRITERIA.price_weight,
            "quality": self.CRITERIA.quality_weight,
            "reputation": self.CRITERIA.reputation_weight,
            "delivery": self.CRITERIA.delivery_weight,
            "description": self.CRITERIA.description_weight,
            "promotion": self.CRITERIA.promotion_weight,
        }

