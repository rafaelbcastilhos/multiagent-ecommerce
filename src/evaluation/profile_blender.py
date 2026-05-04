"""
Blending ponderado de perfis por espaço vetorial.

Calcula pesos de afinidade entre um vetor de cliente e os
centroides dos perfis usando Inverse Distance Weighting (IDW),
e produz uma avaliação interpolada.
"""

import logging
import math

from src.data.schemas import (
    AggregatedEvaluation,
    BlendedEvaluation,
    BuyerProfile,
    CustomerVector,
    EvaluationDimension,
    ProductEvaluation,
)

logger = logging.getLogger(__name__)

# Centroides dos perfis no espaço 3D (price_sensitivity, quality_focus, risk_aversion)
PROFILE_CENTROIDS: dict[BuyerProfile, tuple[float, float, float]] = {
    BuyerProfile.ANXIOUS: (0.30, 0.70, 0.95),
    BuyerProfile.DEMANDING: (0.20, 0.95, 0.60),
    BuyerProfile.ECONOMIC: (0.95, 0.50, 0.50),
    BuyerProfile.IMPULSIVE: (0.60, 0.40, 0.25),
    BuyerProfile.RATIONAL: (0.50, 0.50, 0.50),
}

# Limiar mínimo de peso para considerar um perfil como dominante
DOMINANCE_THRESHOLD = 0.15


def _euclidean_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Calcula distância euclidiana entre dois pontos 3D."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


class ProfileBlender:
    """
    Interpola avaliações de múltiplos perfis usando IDW.

    Dado um vetor de cliente no espaço comportamental 3D,
    calcula pesos de afinidade com cada perfil e produz
    uma avaliação ponderada.

    Attributes:
        exponent: Expoente k do IDW. k=1 suave, k=2 concentrado.
        epsilon: Distância mínima para evitar divisão por zero.
    """

    def __init__(self, exponent: float = 2.0, epsilon: float = 1e-6) -> None:
        self.exponent = exponent
        self.epsilon = epsilon

    def calculate_weights(
        self,
        customer: CustomerVector,
        centroids: dict[BuyerProfile, tuple[float, float, float]] | None = None,
    ) -> dict[BuyerProfile, float]:
        """
        Calcula pesos IDW do cliente para cada perfil.

        Args:
            customer: Vetor do cliente no espaço 3D.
            centroids: Centroides dos perfis (usa padrão se None).

        Returns:
            Dicionário {perfil: peso}, soma = 1.0.
        """
        centroids = centroids or PROFILE_CENTROIDS
        customer_point = customer.to_tuple()

        distances: dict[BuyerProfile, float] = {}
        for profile, centroid in centroids.items():
            d = _euclidean_distance(customer_point, centroid)
            distances[profile] = d

        # Verifica se o cliente coincide com algum centroide
        for profile, d in distances.items():
            if d < self.epsilon:
                return {p: (1.0 if p == profile else 0.0) for p in centroids}

        # IDW: w_i = d_i^(-k) / sum(d_j^(-k))
        inv_distances = {
            profile: d ** (-self.exponent) for profile, d in distances.items()
        }
        total = sum(inv_distances.values())

        weights = {profile: inv_d / total for profile, inv_d in inv_distances.items()}

        logger.debug(
            "Pesos calculados: %s",
            {p.value: f"{w:.3f}" for p, w in weights.items()},
        )

        return weights

    def blend(
        self,
        aggregated: AggregatedEvaluation,
        customer: CustomerVector,
    ) -> BlendedEvaluation:
        """
        Produz avaliação ponderada a partir de avaliações agregadas.

        Args:
            aggregated: Avaliação agregada com resultados dos 5 agentes.
            customer: Vetor do cliente no espaço 3D.

        Returns:
            BlendedEvaluation com scores interpolados.
        """
        weights = self.calculate_weights(customer)

        # Mapeia avaliações por perfil
        eval_by_profile: dict[BuyerProfile, ProductEvaluation] = {
            e.buyer_profile: e for e in aggregated.evaluations
        }

        # Filtra pesos para perfis que possuem avaliação
        active_weights = {
            p: w for p, w in weights.items() if p in eval_by_profile
        }

        # Renormaliza se algum perfil não tem avaliação
        if len(active_weights) < len(weights):
            total = sum(active_weights.values())
            if total > 0:
                active_weights = {p: w / total for p, w in active_weights.items()}

        # Score geral ponderado
        blended_score = sum(
            active_weights[p] * eval_by_profile[p].overall_score
            for p in active_weights
        )

        # Intenção de compra ponderada
        blended_intention = sum(
            active_weights[p] * eval_by_profile[p].purchase_intention
            for p in active_weights
        )

        # Dimensões ponderadas
        blended_dimensions = self._blend_dimensions(active_weights, eval_by_profile)

        # Perfis dominantes (peso > threshold)
        dominant = sorted(active_weights, key=active_weights.get, reverse=True)
        dominant_profiles = [
            p.value for p in dominant if active_weights[p] >= DOMINANCE_THRESHOLD
        ]

        profile_weights_str = {p.value: round(w, 4) for p, w in active_weights.items()}

        logger.info(
            "Blending concluído: score=%.2f, dominantes=%s",
            blended_score,
            dominant_profiles,
        )

        return BlendedEvaluation(
            customer_vector=customer,
            profile_weights=profile_weights_str,
            blended_score=round(blended_score, 2),
            blended_purchase_intention=round(blended_intention, 4),
            blended_dimensions=blended_dimensions,
            dominant_profiles=dominant_profiles,
            source_evaluation=aggregated,
        )

    def _blend_dimensions(
        self,
        weights: dict[BuyerProfile, float],
        eval_by_profile: dict[BuyerProfile, ProductEvaluation],
    ) -> list[EvaluationDimension]:
        """Calcula dimensões ponderadas entre perfis."""
        # Coleta todas as dimensões existentes
        dimension_names: list[str] = []
        for p in weights:
            for dim in eval_by_profile[p].dimensions:
                if dim.name not in dimension_names:
                    dimension_names.append(dim.name)

        blended: list[EvaluationDimension] = []
        for dim_name in dimension_names:
            weighted_score = 0.0
            weighted_weight = 0.0

            for profile, w in weights.items():
                for dim in eval_by_profile[profile].dimensions:
                    if dim.name == dim_name:
                        weighted_score += w * dim.score
                        weighted_weight += w * dim.weight
                        break

            blended.append(
                EvaluationDimension(
                    name=dim_name,
                    score=round(min(max(weighted_score, 0.0), 10.0), 2),
                    weight=round(min(max(weighted_weight, 0.0), 1.0), 4),
                    reasoning=f"Score ponderado por blending (IDW k={self.exponent})",
                )
            )

        return blended
