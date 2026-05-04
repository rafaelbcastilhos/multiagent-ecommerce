"""
Motor de recomendações.

Gera recomendações de otimização para vendedores
baseadas nas avaliações multiagente.
"""

import logging
import uuid
from typing import Callable

from src.data.schemas import (
    AggregatedEvaluation,
    BuyerProfile,
    Product,
    ProductEvaluation,
    Recommendation,
)
from src.evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Motor de geração de recomendações.

    Analisa avaliações multiagente e gera recomendações
    acionáveis para otimização de produtos e promoções.
    """

    def __init__(self) -> None:
        """Inicializa o motor de recomendações."""
        self.metrics = EvaluationMetrics()
        self._rules: list[Callable] = self._initialize_rules()

    def _initialize_rules(self) -> list[Callable]:
        """Inicializa regras de recomendação."""
        return [
            self._check_price_perception,
            self._check_description_quality,
            self._check_reputation_concerns,
            self._check_delivery_issues,
            self._check_promotion_effectiveness,
            self._check_quality_perception,
            self._check_coverage_gaps,
        ]

    def generate_recommendations(
        self,
        product: Product,
        evaluation: AggregatedEvaluation,
    ) -> list[Recommendation]:
        """
        Gera recomendações para um produto.

        Args:
            product: Produto avaliado.
            evaluation: Avaliação agregada.

        Returns:
            Lista de recomendações priorizadas.
        """
        recommendations = []
        
        for rule in self._rules:
            try:
                rec = rule(product, evaluation)
                if rec:
                    recommendations.extend(rec if isinstance(rec, list) else [rec])
            except Exception as e:
                logger.warning(f"Erro na regra {rule.__name__}: {e}")

        # Ordena por prioridade
        recommendations.sort(key=lambda r: r.priority)
        
        logger.info(
            f"Geradas {len(recommendations)} recomendações para {product.product_id}"
        )
        return recommendations

    def _check_price_perception(
        self,
        product: Product,
        evaluation: AggregatedEvaluation,
    ) -> list[Recommendation]:
        """Verifica percepção de preço."""
        recommendations = []
        
        # Identifica perfis com baixo score em preço
        for eval_ in evaluation.evaluations:
            price_dim = next(
                (d for d in eval_.dimensions if d.name == "price"), None
            )
            
            if price_dim and price_dim.score < 5:
                if eval_.buyer_profile == BuyerProfile.ECONOMIC:
                    recommendations.append(
                        Recommendation(
                            recommendation_id=str(uuid.uuid4()),
                            product_id=product.product_id,
                            priority=1,  # Alta prioridade
                            category="pricing",
                            title="Otimizar percepção de custo-benefício",
                            description=f"""O perfil Econômico avaliou o preço negativamente 
                            (score: {price_dim.score:.1f}). Este perfil representa compradores 
                            que buscam maximizar o valor do dinheiro investido.""",
                            action_items=[
                                "Destacar economia comparada ao preço original",
                                "Adicionar comparativo de preço com concorrentes",
                                "Incluir cálculo de custo por uso/dia",
                                "Considerar oferta de frete grátis para aumentar valor percebido",
                            ],
                            expected_impact="Aumento de 15-25% na conversão de compradores econômicos",
                            affected_profiles=[BuyerProfile.ECONOMIC],
                            estimated_improvement=20.0,
                            source_evaluations=[eval_.evaluation_id],
                        )
                    )

        # Desconto pequeno pode ser visto como insignificante
        if 0 < product.discount_percentage < 15:
            recommendations.append(
                Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    product_id=product.product_id,
                    priority=3,
                    category="promotion",
                    title="Desconto pode ser percebido como insignificante",
                    description=f"""Desconto de {product.discount_percentage:.0f}% pode não 
                    gerar impacto emocional suficiente para conversão.""",
                    action_items=[
                        "Considerar aumentar desconto para >= 20%",
                        "Ou remover desconto e ajustar preço base",
                        "Alternativamente, adicionar benefício complementar (frete, brinde)",
                    ],
                    expected_impact="Descontos >= 20% têm maior impacto em decisão de compra",
                    affected_profiles=[BuyerProfile.IMPULSIVE, BuyerProfile.ECONOMIC],
                    estimated_improvement=15.0,
                    source_evaluations=[],
                )
            )

        return recommendations

    def _check_description_quality(
        self,
        product: Product,
        evaluation: AggregatedEvaluation,
    ) -> list[Recommendation]:
        """Verifica qualidade da descrição."""
        recommendations = []
        
        # Verifica dimensão de descrição
        for eval_ in evaluation.evaluations:
            desc_dim = next(
                (d for d in eval_.dimensions if d.name == "description"), None
            )
            
            if desc_dim and desc_dim.score < 6:
                if eval_.buyer_profile == BuyerProfile.DEMANDING:
                    recommendations.append(
                        Recommendation(
                            recommendation_id=str(uuid.uuid4()),
                            product_id=product.product_id,
                            priority=2,
                            category="content",
                            title="Enriquecer descrição para perfil exigente",
                            description=f"""O perfil Exigente avaliou a descrição negativamente 
                            (score: {desc_dim.score:.1f}). Este perfil precisa de informações 
                            detalhadas para tomar decisão.""",
                            action_items=[
                                "Adicionar especificações técnicas detalhadas",
                                "Incluir informações sobre material e composição",
                                "Adicionar tabela de medidas precisa",
                                "Inserir fotos em alta resolução de detalhes",
                                "Descrever processo de fabricação ou origem",
                            ],
                            expected_impact="Aumento de confiança e conversão do perfil exigente",
                            affected_profiles=[BuyerProfile.DEMANDING, BuyerProfile.RATIONAL],
                            estimated_improvement=25.0,
                            source_evaluations=[eval_.evaluation_id],
                        )
                    )
                    break

        # Descrição curta
        if len(product.description) < 100:
            recommendations.append(
                Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    product_id=product.product_id,
                    priority=2,
                    category="content",
                    title="Descrição muito curta",
                    description="""Descrição com menos de 100 caracteres não fornece 
                    informações suficientes para decisão de compra informada.""",
                    action_items=[
                        "Expandir descrição para pelo menos 300 caracteres",
                        "Incluir benefícios do produto",
                        "Adicionar informações de uso e cuidados",
                    ],
                    expected_impact="Maior confiança e menor taxa de devolução",
                    affected_profiles=[BuyerProfile.DEMANDING, BuyerProfile.ANXIOUS],
                    estimated_improvement=20.0,
                    source_evaluations=[],
                )
            )

        return recommendations

    def _check_reputation_concerns(
        self,
        product: Product,
        evaluation: AggregatedEvaluation,
    ) -> list[Recommendation]:
        """Verifica preocupações com reputação."""
        recommendations = []
        
        # Verifica se perfil ansioso está insatisfeito com reputação
        for eval_ in evaluation.evaluations:
            if eval_.buyer_profile == BuyerProfile.ANXIOUS:
                rep_dim = next(
                    (d for d in eval_.dimensions if d.name == "reputation"), None
                )
                
                if rep_dim and rep_dim.score < 7:
                    recommendations.append(
                        Recommendation(
                            recommendation_id=str(uuid.uuid4()),
                            product_id=product.product_id,
                            priority=1,
                            category="trust",
                            title="Fortalecer sinais de confiança",
                            description=f"""O perfil Ansioso avaliou reputação/confiança 
                            negativamente (score: {rep_dim.score:.1f}). Este perfil precisa 
                            de forte sinalização de segurança.""",
                            action_items=[
                                "Destacar política de devolução claramente",
                                "Exibir certificações e selos de qualidade",
                                "Mostrar depoimentos de clientes satisfeitos",
                                "Destacar tempo de atuação e número de vendas",
                                "Garantir resposta rápida a dúvidas",
                            ],
                            expected_impact="Redução de abandono de carrinho por insegurança",
                            affected_profiles=[BuyerProfile.ANXIOUS],
                            estimated_improvement=30.0,
                            source_evaluations=[eval_.evaluation_id],
                        )
                    )

        # Poucas avaliações do produto
        if product.review_count < 10:
            recommendations.append(
                Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    product_id=product.product_id,
                    priority=2,
                    category="trust",
                    title="Aumentar base de avaliações",
                    description=f"""Produto com apenas {product.review_count} avaliações 
                    pode gerar desconfiança, especialmente para compradores ansiosos.""",
                    action_items=[
                        "Solicitar avaliações de compradores anteriores",
                        "Oferecer incentivo para primeiras avaliações",
                        "Destacar qualidade mesmo com poucas avaliações",
                    ],
                    expected_impact="Avaliações aumentam conversão em até 270%",
                    affected_profiles=[BuyerProfile.ANXIOUS, BuyerProfile.RATIONAL],
                    estimated_improvement=25.0,
                    source_evaluations=[],
                )
            )

        return recommendations

    def _check_delivery_issues(
        self,
        product: Product,
        evaluation: AggregatedEvaluation,
    ) -> list[Recommendation]:
        """Verifica problemas de entrega."""
        recommendations = []
        
        # Prazo longo de entrega
        if product.estimated_delivery_days > 10:
            recommendations.append(
                Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    product_id=product.product_id,
                    priority=2,
                    category="logistics",
                    title="Prazo de entrega pode afastar compradores",
                    description=f"""Prazo de {product.estimated_delivery_days} dias é 
                    considerado longo. Compradores ansiosos e impulsivos preferem 
                    entregas rápidas.""",
                    action_items=[
                        "Buscar opções de entrega expressa",
                        "Comunicar claramente o prazo e rastreamento",
                        "Oferecer compensação pelo prazo (desconto, brinde)",
                    ],
                    expected_impact="Entregas mais rápidas aumentam conversão",
                    affected_profiles=[BuyerProfile.ANXIOUS, BuyerProfile.IMPULSIVE],
                    estimated_improvement=15.0,
                    source_evaluations=[],
                )
            )

        # Sem frete grátis
        if not product.free_shipping:
            # Verifica se perfil econômico está insatisfeito
            for eval_ in evaluation.evaluations:
                if eval_.buyer_profile == BuyerProfile.ECONOMIC:
                    delivery_dim = next(
                        (d for d in eval_.dimensions if d.name == "delivery"), None
                    )
                    if delivery_dim and delivery_dim.score < 6:
                        recommendations.append(
                            Recommendation(
                                recommendation_id=str(uuid.uuid4()),
                                product_id=product.product_id,
                                priority=2,
                                category="logistics",
                                title="Considerar frete grátis",
                                description="""Frete pago pode anular percepção de 
                                desconto para compradores econômicos.""",
                                action_items=[
                                    "Avaliar viabilidade de frete grátis",
                                    "Ou incluir frete no preço com desconto aparente maior",
                                    "Oferecer frete grátis acima de valor mínimo",
                                ],
                                expected_impact="Frete grátis pode aumentar conversão em 30%",
                                affected_profiles=[BuyerProfile.ECONOMIC],
                                estimated_improvement=25.0,
                                source_evaluations=[eval_.evaluation_id],
                            )
                        )
                    break

        return recommendations

    def _check_promotion_effectiveness(
        self,
        product: Product,
        evaluation: AggregatedEvaluation,
    ) -> list[Recommendation]:
        """Verifica efetividade da promoção."""
        recommendations = []
        
        # Verifica score de promoção para perfil impulsivo
        for eval_ in evaluation.evaluations:
            if eval_.buyer_profile == BuyerProfile.IMPULSIVE:
                promo_dim = next(
                    (d for d in eval_.dimensions if d.name == "promotion"), None
                )
                
                if promo_dim and promo_dim.score < 7:
                    recommendations.append(
                        Recommendation(
                            recommendation_id=str(uuid.uuid4()),
                            product_id=product.product_id,
                            priority=2,
                            category="promotion",
                            title="Aumentar apelo visual da promoção",
                            description=f"""O perfil Impulsivo avaliou a promoção em 
                            {promo_dim.score:.1f}. Este perfil responde a gatilhos 
                            visuais e urgência.""",
                            action_items=[
                                "Usar badges visuais chamativos (OFERTA, -X%)",
                                "Adicionar senso de urgência (últimas unidades, tempo limitado)",
                                "Destacar economia em valor absoluto",
                                "Usar cores que chamem atenção",
                            ],
                            expected_impact="Gatilhos de urgência aumentam conversão impulsiva",
                            affected_profiles=[BuyerProfile.IMPULSIVE],
                            estimated_improvement=20.0,
                            source_evaluations=[eval_.evaluation_id],
                        )
                    )

        return recommendations

    def _check_quality_perception(
        self,
        product: Product,
        evaluation: AggregatedEvaluation,
    ) -> list[Recommendation]:
        """Verifica percepção de qualidade."""
        recommendations = []
        
        # Verifica score de qualidade para perfil exigente
        for eval_ in evaluation.evaluations:
            if eval_.buyer_profile == BuyerProfile.DEMANDING:
                quality_dim = next(
                    (d for d in eval_.dimensions if d.name == "quality"), None
                )
                
                if quality_dim and quality_dim.score < 7:
                    recommendations.append(
                        Recommendation(
                            recommendation_id=str(uuid.uuid4()),
                            product_id=product.product_id,
                            priority=2,
                            category="quality",
                            title="Melhorar percepção de qualidade",
                            description=f"""O perfil Exigente avaliou qualidade em 
                            {quality_dim.score:.1f}. Necessário reforçar atributos 
                            de qualidade.""",
                            action_items=[
                                "Destacar materiais premium utilizados",
                                "Mostrar detalhes de acabamento nas fotos",
                                "Incluir certificações de qualidade",
                                "Descrever processo de controle de qualidade",
                                "Adicionar comparativo com produtos similares",
                            ],
                            expected_impact="Melhor percepção de qualidade justifica preço",
                            affected_profiles=[BuyerProfile.DEMANDING],
                            estimated_improvement=20.0,
                            source_evaluations=[eval_.evaluation_id],
                        )
                    )

        return recommendations

    def _check_coverage_gaps(
        self,
        product: Product,
        evaluation: AggregatedEvaluation,
    ) -> list[Recommendation]:
        """Verifica gaps de cobertura entre perfis."""
        recommendations = []
        
        # Se há perfis muito insatisfeitos
        if len(evaluation.profiles_unsatisfied) >= 2:
            profiles_list = ", ".join(p.value for p in evaluation.profiles_unsatisfied)
            recommendations.append(
                Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    product_id=product.product_id,
                    priority=1,
                    category="coverage",
                    title="Baixa cobertura de perfis",
                    description=f"""Múltiplos perfis insatisfeitos: {profiles_list}. 
                    Produto pode estar com posicionamento inadequado.""",
                    action_items=[
                        "Revisar posicionamento do produto",
                        "Analisar pontos fracos comuns entre perfis",
                        "Considerar ajustes na oferta ou comunicação",
                    ],
                    expected_impact="Aumentar cobertura de satisfação diversificada",
                    affected_profiles=list(evaluation.profiles_unsatisfied),
                    estimated_improvement=30.0,
                    source_evaluations=[
                        e.evaluation_id for e in evaluation.evaluations
                        if e.buyer_profile in evaluation.profiles_unsatisfied
                    ],
                )
            )

        # Alta variância indica oportunidade de otimização focada
        if evaluation.std_score > 2.0:
            recommendations.append(
                Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    product_id=product.product_id,
                    priority=2,
                    category="coverage",
                    title="Alta variação entre perfis - oportunidade de nicho",
                    description=f"""Desvio padrão de {evaluation.std_score:.1f} indica 
                    que alguns perfis gostam muito enquanto outros não. Pode ser 
                    oportunidade de posicionamento de nicho.""",
                    action_items=[
                        "Identificar e focar no perfil mais satisfeito",
                        "Ou ajustar para maior equilíbrio entre perfis",
                        "Considerar variações do produto para diferentes perfis",
                    ],
                    expected_impact="Clareza de posicionamento melhora conversão",
                    affected_profiles=[],
                    estimated_improvement=15.0,
                    source_evaluations=[],
                )
            )

        return recommendations

    def summarize_recommendations(
        self,
        recommendations: list[Recommendation],
    ) -> dict:
        """
        Cria resumo das recomendações.

        Args:
            recommendations: Lista de recomendações.

        Returns:
            Resumo estruturado.
        """
        if not recommendations:
            return {"message": "Nenhuma recomendação gerada"}

        by_priority = {1: [], 2: [], 3: [], 4: [], 5: []}
        by_category = {}
        
        for rec in recommendations:
            by_priority[rec.priority].append(rec.title)
            
            if rec.category not in by_category:
                by_category[rec.category] = []
            by_category[rec.category].append(rec.title)

        return {
            "total_recommendations": len(recommendations),
            "by_priority": {
                "urgent": by_priority[1],
                "high": by_priority[2],
                "medium": by_priority[3],
                "low": by_priority[4] + by_priority[5],
            },
            "by_category": by_category,
            "top_actions": [r.action_items[0] for r in recommendations[:3] if r.action_items],
            "estimated_total_improvement": sum(r.estimated_improvement for r in recommendations) / len(recommendations),
        }

