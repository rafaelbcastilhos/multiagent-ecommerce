"""
Métricas de avaliação do sistema multiagente.

Define métricas para avaliar a qualidade das análises
e a cobertura de perfis.
"""

import statistics
from dataclasses import dataclass
from typing import Callable

from src.data.schemas import (
    AggregatedEvaluation,
    BuyerProfile,
    ProductEvaluation,
)


@dataclass(frozen=True)
class CoverageMetrics:
    """Métricas de cobertura de perfis."""

    total_profiles: int
    satisfied_profiles: int
    unsatisfied_profiles: int
    neutral_profiles: int
    coverage_rate: float  # % satisfeitos
    risk_rate: float      # % insatisfeitos
    diversity_score: float  # Variação entre perfis


@dataclass(frozen=True)
class ConsensusMetrics:
    """Métricas de consenso entre agentes."""

    agreement_score: float  # Quão similares são as avaliações
    disagreement_areas: list[str]  # Dimensões com maior divergência
    strongest_consensus: list[str]  # Pontos de maior acordo
    weakest_consensus: list[str]   # Pontos de maior discordância


@dataclass(frozen=True)
class PromotionEffectiveness:
    """Métricas de efetividade da promoção."""

    overall_appeal: float  # Apelo geral (0-10)
    profile_specific_appeal: dict[str, float]  # Apelo por perfil
    conversion_potential: float  # Potencial de conversão estimado
    improvement_potential: float  # Potencial de melhoria


class EvaluationMetrics:
    """
    Calculadora de métricas para avaliações multiagente.

    Fornece métricas quantitativas para análise de resultados
    das avaliações de produtos.
    """

    SATISFACTION_THRESHOLD: float = 7.0  # Score >= 7 = satisfeito
    DISSATISFACTION_THRESHOLD: float = 5.0  # Score < 5 = insatisfeito

    def calculate_coverage(
        self,
        evaluation: AggregatedEvaluation,
    ) -> CoverageMetrics:
        """
        Calcula métricas de cobertura de perfis.

        Args:
            evaluation: Avaliação agregada.

        Returns:
            CoverageMetrics com métricas calculadas.
        """
        total = len(evaluation.evaluations)
        satisfied = len(evaluation.profiles_satisfied)
        unsatisfied = len(evaluation.profiles_unsatisfied)
        neutral = total - satisfied - unsatisfied

        # Calcula diversidade (baseado em desvio padrão normalizado)
        scores = [e.overall_score for e in evaluation.evaluations]
        if len(scores) > 1:
            std = statistics.stdev(scores)
            # Normaliza: std de 0 = alta diversidade (todos iguais seria baixa)
            diversity = 1 - (std / 5)  # Max std teórico ~5
        else:
            diversity = 0.0

        return CoverageMetrics(
            total_profiles=total,
            satisfied_profiles=satisfied,
            unsatisfied_profiles=unsatisfied,
            neutral_profiles=neutral,
            coverage_rate=satisfied / total if total > 0 else 0,
            risk_rate=unsatisfied / total if total > 0 else 0,
            diversity_score=max(0, min(1, diversity)),
        )

    def calculate_consensus(
        self,
        evaluation: AggregatedEvaluation,
    ) -> ConsensusMetrics:
        """
        Calcula métricas de consenso entre agentes.

        Args:
            evaluation: Avaliação agregada.

        Returns:
            ConsensusMetrics com análise de consenso.
        """
        evaluations = evaluation.evaluations
        
        if len(evaluations) < 2:
            return ConsensusMetrics(
                agreement_score=1.0,
                disagreement_areas=[],
                strongest_consensus=evaluation.consensus_strengths,
                weakest_consensus=[],
            )

        # Calcula agreement score baseado na variância
        scores = [e.overall_score for e in evaluations]
        variance = statistics.variance(scores)
        # Normaliza: variância 0 = agreement 1, variância alta = agreement baixo
        agreement = max(0, 1 - (variance / 25))  # Max variância teórica = 25

        # Identifica dimensões com maior divergência
        dimension_variances = self._calculate_dimension_variances(evaluations)
        disagreement_areas = sorted(
            dimension_variances.keys(),
            key=lambda x: dimension_variances[x],
            reverse=True,
        )[:3]

        return ConsensusMetrics(
            agreement_score=agreement,
            disagreement_areas=disagreement_areas,
            strongest_consensus=evaluation.consensus_strengths[:3],
            weakest_consensus=evaluation.consensus_weaknesses[:3],
        )

    def _calculate_dimension_variances(
        self,
        evaluations: list[ProductEvaluation],
    ) -> dict[str, float]:
        """Calcula variância por dimensão."""
        dimension_scores: dict[str, list[float]] = {}
        
        for eval_ in evaluations:
            for dim in eval_.dimensions:
                if dim.name not in dimension_scores:
                    dimension_scores[dim.name] = []
                dimension_scores[dim.name].append(dim.score)

        variances = {}
        for name, scores in dimension_scores.items():
            if len(scores) > 1:
                variances[name] = statistics.variance(scores)
            else:
                variances[name] = 0.0

        return variances

    def calculate_promotion_effectiveness(
        self,
        evaluation: AggregatedEvaluation,
    ) -> PromotionEffectiveness:
        """
        Calcula efetividade da promoção.

        Args:
            evaluation: Avaliação agregada.

        Returns:
            PromotionEffectiveness com métricas.
        """
        # Apelo geral
        overall = evaluation.mean_score
        
        # Apelo por perfil
        profile_appeal = {
            e.buyer_profile.value: e.overall_score
            for e in evaluation.evaluations
        }
        
        # Potencial de conversão (média ponderada das intenções de compra)
        intentions = [e.purchase_intention for e in evaluation.evaluations]
        conversion = statistics.mean(intentions) if intentions else 0.0
        
        # Potencial de melhoria (gap até score máximo)
        improvement = (10 - overall) / 10

        return PromotionEffectiveness(
            overall_appeal=overall,
            profile_specific_appeal=profile_appeal,
            conversion_potential=conversion,
            improvement_potential=improvement,
        )


