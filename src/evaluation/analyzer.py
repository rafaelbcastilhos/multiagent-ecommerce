"""
Analisador de resultados das avaliações.

Gera relatórios e visualizações a partir das
avaliações multiagente.
"""

import logging
from collections import Counter
from datetime import datetime
from typing import Any

from src.data.schemas import (
    AggregatedEvaluation,
    BuyerProfile,
    Product,
    ProductEvaluation,
)
from src.evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """
    Analisador de resultados de avaliações.

    Gera relatórios, rankings e insights a partir
    das avaliações multiagente.
    """

    def __init__(self) -> None:
        """Inicializa o analisador."""
        self.metrics = EvaluationMetrics()

    def generate_product_report(
        self,
        product: Product,
        evaluation: AggregatedEvaluation,
    ) -> dict[str, Any]:
        """
        Gera relatório detalhado de um produto.

        Args:
            product: Produto avaliado.
            evaluation: Avaliação agregada.

        Returns:
            Relatório estruturado.
        """
        coverage = self.metrics.calculate_coverage(evaluation)
        consensus = self.metrics.calculate_consensus(evaluation)
        effectiveness = self.metrics.calculate_promotion_effectiveness(evaluation)

        return {
            "product": {
                "id": product.product_id,
                "title": product.title,
                "brand": product.brand,
                "category": product.category.value,
                "current_price": product.current_price,
                "original_price": product.original_price,
                "discount": product.discount_percentage,
                "rating": product.rating,
                "reviews": product.review_count,
            },
            "overall_analysis": {
                "mean_score": round(evaluation.mean_score, 2),
                "score_range": f"{evaluation.min_score:.1f} - {evaluation.max_score:.1f}",
                "coverage_score": round(coverage.coverage_rate * 100, 1),
                "risk_score": round(coverage.risk_rate * 100, 1),
                "consensus_level": round(consensus.agreement_score * 100, 1),
            },
            "profile_results": {
                eval_.buyer_profile.value: {
                    "score": round(eval_.overall_score, 2),
                    "purchase_intention": round(eval_.purchase_intention * 100, 1),
                    "status": self._get_satisfaction_status(eval_.overall_score),
                    "top_concerns": eval_.weaknesses[:2],
                    "top_strengths": eval_.strengths[:2],
                }
                for eval_ in evaluation.evaluations
            },
            "consensus": {
                "strengths": evaluation.consensus_strengths,
                "weaknesses": evaluation.consensus_weaknesses,
                "disagreement_areas": consensus.disagreement_areas,
            },
            "promotion_effectiveness": {
                "overall_appeal": round(effectiveness.overall_appeal, 2),
                "conversion_potential": round(effectiveness.conversion_potential * 100, 1),
                "improvement_potential": round(effectiveness.improvement_potential * 100, 1),
            },
            "metadata": {
                "profiles_analyzed": len(evaluation.evaluations),
                "generated_at": datetime.now().isoformat(),
            },
        }

    def _get_satisfaction_status(self, score: float) -> str:
        """Converte score em status textual."""
        if score >= 8:
            return "Muito Satisfeito"
        if score >= 7:
            return "Satisfeito"
        if score >= 5:
            return "Neutro"
        if score >= 3:
            return "Insatisfeito"
        return "Muito Insatisfeito"

    def rank_products(
        self,
        evaluations: list[tuple[Product, AggregatedEvaluation]],
        sort_by: str = "mean_score",
    ) -> list[dict]:
        """
        Cria ranking de produtos.

        Args:
            evaluations: Lista de (produto, avaliação).
            sort_by: Critério de ordenação.

        Returns:
            Lista ordenada com rankings.
        """
        rankings = []
        
        for product, evaluation in evaluations:
            coverage = self.metrics.calculate_coverage(evaluation)
            
            rankings.append({
                "rank": 0,  # Será preenchido após ordenação
                "product_id": product.product_id,
                "title": product.title[:50],
                "mean_score": evaluation.mean_score,
                "coverage_score": coverage.coverage_rate,
                "satisfied_profiles": len(evaluation.profiles_satisfied),
                "unsatisfied_profiles": len(evaluation.profiles_unsatisfied),
                "min_score": evaluation.min_score,
                "max_score": evaluation.max_score,
            })

        # Ordena pelo critério
        sort_key = sort_by if sort_by in rankings[0] else "mean_score"
        rankings.sort(key=lambda x: x[sort_key], reverse=True)
        
        # Adiciona ranking
        for i, item in enumerate(rankings, 1):
            item["rank"] = i

        return rankings

    def identify_top_performers(
        self,
        evaluations: list[tuple[Product, AggregatedEvaluation]],
        by_profile: BuyerProfile | None = None,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Identifica produtos com melhor desempenho.

        Args:
            evaluations: Lista de (produto, avaliação).
            by_profile: Filtrar por perfil específico.
            top_n: Número de top performers.

        Returns:
            Lista dos melhores produtos.
        """
        scored_products = []
        
        for product, evaluation in evaluations:
            if by_profile:
                # Busca score específico do perfil
                profile_eval = next(
                    (e for e in evaluation.evaluations if e.buyer_profile == by_profile),
                    None
                )
                score = profile_eval.overall_score if profile_eval else 0
            else:
                score = evaluation.mean_score
            
            scored_products.append({
                "product": product,
                "evaluation": evaluation,
                "score": score,
            })

        scored_products.sort(key=lambda x: x["score"], reverse=True)
        
        return [
            {
                "product_id": item["product"].product_id,
                "title": item["product"].title,
                "score": round(item["score"], 2),
                "profile": by_profile.value if by_profile else "all",
            }
            for item in scored_products[:top_n]
        ]

    def identify_improvement_opportunities(
        self,
        evaluations: list[tuple[Product, AggregatedEvaluation]],
    ) -> list[dict]:
        """
        Identifica produtos com maior potencial de melhoria.

        Produtos com alta variância entre perfis ou
        com perfis específicos insatisfeitos.

        Args:
            evaluations: Lista de (produto, avaliação).

        Returns:
            Lista de oportunidades de melhoria.
        """
        opportunities = []
        
        for product, evaluation in evaluations:
            # Alta variância indica que alguns perfis gostam, outros não
            if evaluation.std_score > 1.5:
                opportunities.append({
                    "product_id": product.product_id,
                    "title": product.title,
                    "opportunity_type": "high_variance",
                    "description": "Alta variação entre perfis - oportunidade de otimização focada",
                    "satisfied_profiles": [p.value for p in evaluation.profiles_satisfied],
                    "unsatisfied_profiles": [p.value for p in evaluation.profiles_unsatisfied],
                    "priority": "high" if len(evaluation.profiles_unsatisfied) >= 2 else "medium",
                })
            
            # Produtos com perfis específicos muito insatisfeitos
            for eval_ in evaluation.evaluations:
                if eval_.overall_score < 4:
                    opportunities.append({
                        "product_id": product.product_id,
                        "title": product.title,
                        "opportunity_type": "profile_specific",
                        "description": f"Perfil {eval_.buyer_profile.value} muito insatisfeito",
                        "target_profile": eval_.buyer_profile.value,
                        "current_score": eval_.overall_score,
                        "main_concerns": eval_.weaknesses[:3],
                        "priority": "high",
                    })

        # Ordena por prioridade
        priority_order = {"high": 0, "medium": 1, "low": 2}
        opportunities.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))
        
        return opportunities

    def analyze_profile_patterns(
        self,
        evaluations: list[tuple[Product, AggregatedEvaluation]],
    ) -> dict[str, Any]:
        """
        Analisa padrões de comportamento por perfil.

        Args:
            evaluations: Lista de (produto, avaliação).

        Returns:
            Análise de padrões por perfil.
        """
        profile_data: dict[BuyerProfile, dict] = {}
        
        for _, evaluation in evaluations:
            for eval_ in evaluation.evaluations:
                profile = eval_.buyer_profile
                
                if profile not in profile_data:
                    profile_data[profile] = {
                        "scores": [],
                        "intentions": [],
                        "all_strengths": [],
                        "all_weaknesses": [],
                    }
                
                profile_data[profile]["scores"].append(eval_.overall_score)
                profile_data[profile]["intentions"].append(eval_.purchase_intention)
                profile_data[profile]["all_strengths"].extend(eval_.strengths)
                profile_data[profile]["all_weaknesses"].extend(eval_.weaknesses)

        # Processa dados
        patterns = {}
        for profile, data in profile_data.items():
            scores = data["scores"]
            intentions = data["intentions"]
            
            # Conta menções mais frequentes
            strength_counts = Counter(
                s.lower().strip() for s in data["all_strengths"]
            ).most_common(5)
            weakness_counts = Counter(
                w.lower().strip() for w in data["all_weaknesses"]
            ).most_common(5)

            patterns[profile.value] = {
                "average_score": sum(scores) / len(scores) if scores else 0,
                "average_intention": sum(intentions) / len(intentions) if intentions else 0,
                "score_std": (
                    (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5
                    if len(scores) > 1 else 0
                ),
                "most_valued": [s[0] for s in strength_counts],
                "main_concerns": [w[0] for w in weakness_counts],
                "total_evaluations": len(scores),
            }

        return patterns

