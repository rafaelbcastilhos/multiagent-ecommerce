"""
Orquestrador de agentes multiagente.

Coordena a execução de múltiplos agentes para avaliar
produtos e agregar resultados.
"""

import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.anxious_buyer import AnxiousBuyerAgent
from src.agents.demanding_buyer import DemandingBuyerAgent
from src.agents.economic_buyer import EconomicBuyerAgent
from src.agents.impulsive_buyer import ImpulsiveBuyerAgent
from src.agents.rational_buyer import RationalBuyerAgent
from src.core.base_agent import BaseAgent
from src.core.llm_client import OllamaClient
from src.data.schemas import (
    AggregatedEvaluation,
    BlendedEvaluation,
    BuyerProfile,
    CustomerVector,
    Product,
    ProductEvaluation,
)
from src.evaluation.profile_blender import ProfileBlender

logger = logging.getLogger(__name__)


# Registro de todos os agentes disponíveis
AVAILABLE_AGENTS: dict[BuyerProfile, type[BaseAgent]] = {
    BuyerProfile.ANXIOUS: AnxiousBuyerAgent,
    BuyerProfile.DEMANDING: DemandingBuyerAgent,
    BuyerProfile.ECONOMIC: EconomicBuyerAgent,
    BuyerProfile.IMPULSIVE: ImpulsiveBuyerAgent,
    BuyerProfile.RATIONAL: RationalBuyerAgent,
}


class AgentOrchestrator:
    """
    Orquestrador para coordenação de múltiplos agentes.

    Gerencia a execução paralela/sequencial de agentes,
    agregação de resultados e análise de consenso.

    Attributes:
        llm_client: Cliente Ollama compartilhado.
        agents: Dicionário de agentes instanciados.
        max_workers: Número máximo de workers paralelos.
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        profiles: list[BuyerProfile] | None = None,
        max_workers: int = 3,
    ) -> None:
        """
        Inicializa o orquestrador.

        Args:
            llm_client: Cliente Ollama para os agentes.
            profiles: Lista de perfis a serem usados (None = todos).
            max_workers: Máximo de agentes executando em paralelo.
        """
        self.llm_client = llm_client
        self.max_workers = max_workers
        
        # Instancia agentes selecionados
        selected_profiles = profiles or list(AVAILABLE_AGENTS.keys())
        self.agents: dict[BuyerProfile, BaseAgent] = {}
        
        for profile in selected_profiles:
            if profile in AVAILABLE_AGENTS:
                agent_class = AVAILABLE_AGENTS[profile]
                self.agents[profile] = agent_class(llm_client)
                logger.info(f"Agente inicializado: {profile.value}")

    def evaluate_product(
        self,
        product: Product,
        parallel: bool = True,
    ) -> AggregatedEvaluation:
        """
        Avalia um produto com todos os agentes.

        Args:
            product: Produto a ser avaliado.
            parallel: Se True, executa agentes em paralelo.

        Returns:
            AggregatedEvaluation com resultados agregados.
        """
        logger.info(f"Iniciando avaliação multiagente: {product.title[:50]}...")
        
        if parallel:
            evaluations = self._evaluate_parallel(product)
        else:
            evaluations = self._evaluate_sequential(product)

        return self._aggregate_evaluations(product.product_id, evaluations)

    def evaluate_product_blended(
        self,
        product: Product,
        customer_vector: CustomerVector,
        parallel: bool = True,
        idw_exponent: float = 2.0,
    ) -> BlendedEvaluation:
        """
        Avalia produto e aplica blending personalizado.

        Executa a avaliação com todos os agentes e interpola os
        resultados segundo a posição do cliente no espaço vetorial.

        Args:
            product: Produto a ser avaliado.
            customer_vector: Vetor comportamental do cliente.
            parallel: Se True, executa agentes em paralelo.
            idw_exponent: Expoente k do IDW (1=suave, 2=concentrado).

        Returns:
            BlendedEvaluation com scores interpolados.
        """
        logger.info(
            "Avaliação blended: %s | vetor=%s | k=%.1f",
            product.title[:40],
            customer_vector.to_tuple(),
            idw_exponent,
        )

        aggregated = self.evaluate_product(product, parallel=parallel)
        blender = ProfileBlender(exponent=idw_exponent)
        return blender.blend(aggregated, customer_vector)

    def _evaluate_sequential(self, product: Product) -> list[ProductEvaluation]:
        """Executa avaliações sequencialmente."""
        evaluations = []
        
        for profile, agent in self.agents.items():
            try:
                evaluation = agent.evaluate(product)
                evaluations.append(evaluation)
                logger.info(
                    f"[{profile.value}] Score: {evaluation.overall_score:.1f}"
                )
            except Exception as e:
                logger.error(f"[{profile.value}] Erro: {e}")
                continue

        return evaluations

    def _evaluate_parallel(self, product: Product) -> list[ProductEvaluation]:
        """Executa avaliações em paralelo."""
        evaluations = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_profile = {
                executor.submit(agent.evaluate, product): profile
                for profile, agent in self.agents.items()
            }
            
            for future in as_completed(future_to_profile):
                profile = future_to_profile[future]
                try:
                    evaluation = future.result()
                    evaluations.append(evaluation)
                    logger.info(
                        f"[{profile.value}] Score: {evaluation.overall_score:.1f}"
                    )
                except Exception as e:
                    logger.error(f"[{profile.value}] Erro: {e}")

        return evaluations

    def _aggregate_evaluations(
        self,
        product_id: str,
        evaluations: list[ProductEvaluation],
    ) -> AggregatedEvaluation:
        """
        Agrega avaliações de múltiplos agentes.

        Args:
            product_id: ID do produto avaliado.
            evaluations: Lista de avaliações individuais.

        Returns:
            AggregatedEvaluation com métricas agregadas.
        """
        if not evaluations:
            raise ValueError("Nenhuma avaliação para agregar")

        scores = [e.overall_score for e in evaluations]
        
        # Calcula métricas estatísticas
        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
        min_score = min(scores)
        max_score = max(scores)
        
        # Identifica perfis satisfeitos/insatisfeitos
        profiles_satisfied = [
            e.buyer_profile for e in evaluations if e.overall_score >= 7.0
        ]
        profiles_unsatisfied = [
            e.buyer_profile for e in evaluations if e.overall_score < 5.0
        ]
        
        # Calcula score de cobertura (% de perfis satisfeitos)
        coverage_score = len(profiles_satisfied) / len(evaluations)
        
        # Encontra consensos
        consensus_strengths = self._find_consensus(
            [e.strengths for e in evaluations], threshold=0.5
        )
        consensus_weaknesses = self._find_consensus(
            [e.weaknesses for e in evaluations], threshold=0.5
        )

        return AggregatedEvaluation(
            product_id=product_id,
            evaluations=evaluations,
            mean_score=mean_score,
            std_score=std_score,
            min_score=min_score,
            max_score=max_score,
            coverage_score=coverage_score,
            profiles_satisfied=profiles_satisfied,
            profiles_unsatisfied=profiles_unsatisfied,
            consensus_strengths=consensus_strengths,
            consensus_weaknesses=consensus_weaknesses,
        )

    def _find_consensus(
        self,
        items_lists: list[list[str]],
        threshold: float = 0.5,
    ) -> list[str]:
        """
        Encontra itens mencionados por múltiplos agentes.

        Args:
            items_lists: Lista de listas de itens.
            threshold: Proporção mínima para consenso.

        Returns:
            Lista de itens em consenso.
        """
        if not items_lists:
            return []

        # Conta ocorrências normalizadas (lowercase)
        item_counts: dict[str, int] = {}
        for items in items_lists:
            seen = set()  # Evita contar duplicatas do mesmo agente
            for item in items:
                normalized = item.lower().strip()
                if normalized not in seen:
                    item_counts[normalized] = item_counts.get(normalized, 0) + 1
                    seen.add(normalized)

        # Filtra por threshold
        min_count = int(len(items_lists) * threshold)
        consensus = [
            item for item, count in item_counts.items()
            if count >= max(min_count, 2)
        ]

        return consensus[:5]  # Limita a 5 itens

    def evaluate_products_batch(
        self,
        products: list[Product],
        parallel_agents: bool = True,
        parallel_products: bool = False,
        max_product_workers: int = 2,
    ) -> list[AggregatedEvaluation]:
        """
        Avalia múltiplos produtos em lote.

        Args:
            products: Lista de produtos.
            parallel_agents: Se True, agentes executam em paralelo por produto.
            parallel_products: Se True, múltiplos produtos são avaliados em paralelo.
            max_product_workers: Máximo de produtos processados simultaneamente.

        Returns:
            Lista de avaliações agregadas na mesma ordem dos produtos de entrada.
        """
        total = len(products)

        if not parallel_products:
            results = []
            for i, product in enumerate(products, 1):
                logger.info(f"Processando produto {i}/{total}")
                try:
                    result = self.evaluate_product(product, parallel=parallel_agents)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Erro no produto {product.product_id}: {e}")
            return results

        # Processamento paralelo de produtos
        logger.info(f"Processando {total} produtos em paralelo (workers={max_product_workers})")
        results: list[AggregatedEvaluation | None] = [None] * total

        def _process(index: int, product: Product) -> tuple[int, AggregatedEvaluation | None]:
            logger.info(f"Processando produto {index + 1}/{total}: {product.title[:40]}...")
            try:
                return index, self.evaluate_product(product, parallel=parallel_agents)
            except Exception as e:
                logger.error(f"Erro no produto {product.product_id}: {e}")
                return index, None

        with ThreadPoolExecutor(max_workers=max_product_workers) as executor:
            futures = {
                executor.submit(_process, i, product): i
                for i, product in enumerate(products)
            }
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return [r for r in results if r is not None]

    def get_agent_profiles(self) -> list[BuyerProfile]:
        """Retorna lista de perfis ativos."""
        return list(self.agents.keys())

    def get_agent(self, profile: BuyerProfile) -> BaseAgent | None:
        """Retorna agente específico por perfil."""
        return self.agents.get(profile)

