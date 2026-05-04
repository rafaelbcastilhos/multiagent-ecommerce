"""
Ponto de entrada principal da aplicação.

Sistema Multiagente para Avaliação de Promoções
com Perfis Psicológicos de Compradores.
"""

import argparse
import logging
import random
import sys
from pathlib import Path

from config.settings import SUPPORTED_MODELS, get_settings
from src.core.llm_client import OllamaClient
from src.core.orchestrator import AgentOrchestrator
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.schemas import BuyerProfile
from src.database import (
    DatabaseConnection,
    EvaluationRepository,
    MongoEvaluationRepository,
)
from src.evaluation.analyzer import ResultAnalyzer
from src.recommendations.engine import RecommendationEngine
from src.recommendations.seller_message import SellerMessageGenerator

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """Cria diretórios necessários."""
    dirs = [
        "data/raw",
        "data/processed",
        "data/cache",
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def open_mongo_repository(settings) -> MongoEvaluationRepository | None:
    """
    Abre o repositório MongoDB e valida conectividade.

    Returns:
        Repositório pronto para uso, ou None se o servidor não estiver disponível
        (a execução prossegue salvando apenas no SQLite).
    """
    repo = MongoEvaluationRepository(
        uri=settings.mongo.uri,
        database=settings.mongo.database,
        collection=settings.mongo.reports_collection,
        server_selection_timeout_ms=settings.mongo.server_selection_timeout_ms,
    )
    try:
        repo.ping()
        repo.ensure_indexes()
        return repo
    except Exception as e:
        logger.error(
            f"MongoDB indisponível em {settings.mongo.uri}: {e}. "
            "Os relatórios não serão persistidos no Mongo nesta execução."
        )
        repo.close()
        return None


def check_ollama_connection(client: OllamaClient) -> bool:
    """Verifica conexão com Ollama."""
    if not client.is_available():
        logger.error(
            "Ollama não está disponível. Verifique se o servidor está rodando:\n"
            "  1. Instale Ollama: https://ollama.ai\n"
            "  2. Inicie o servidor: ollama serve\n"
            "  3. Baixe um modelo: ollama pull llama3.2"
        )
        return False
    
    models = client.list_models()
    logger.info(f"Ollama disponível. Modelos: {models}")
    return True


def run_demo(
    parallel: bool = True,
    parallel_products: bool = False,
    model: str | None = None,
    sample: int | None = None,
    seed: int = 42,
) -> None:
    """
    Executa o sistema sobre todo o dataset.

    Args:
        parallel: Executar agentes em paralelo por produto.
        parallel_products: Avaliar múltiplos produtos em paralelo.
        model: Modelo LLM a utilizar (sobrescreve configuração).
        sample: Número de produtos a amostrar aleatoriamente (None = todos).
        seed: Semente para reprodutibilidade da amostragem.
    """
    settings = get_settings()
    active_model = model or settings.ollama.model

    logger.info("=" * 60)
    logger.info("Sistema Multiagente - Avaliação de Promoções")
    logger.info(f"Modelo: {active_model}")
    logger.info("=" * 60)

    # Inicializa cliente LLM
    with OllamaClient(
        host=settings.ollama.host,
        model=active_model,
        temperature=settings.ollama.temperature,
        timeout=settings.ollama.timeout,
        max_retries=settings.ollama.max_retries,
        retry_backoff=settings.ollama.retry_backoff,
    ) as llm_client:

        if not check_ollama_connection(llm_client):
            return

        # Carrega dados
        logger.info("\n📦 Carregando dados...")
        loader = DataLoader(
            data_path=settings.data.raw_path,
            dataset_file=settings.data.dataset_file,
        )
        
        try:
            products = loader.load_products()
        except FileNotFoundError as e:
            logger.error(str(e))
            logger.info("\n📥 Para executar a demo, baixe o dataset:")
            logger.info("   https://www.kaggle.com/datasets/aaditshukla/flipkart-fasion-products-dataset")
            logger.info(f"   E salve em: {settings.data.raw_path}/{settings.data.dataset_file}")
            return

        if not products:
            logger.error("Nenhum produto carregado")
            return

        # Pré-processa
        preprocessor = DataPreprocessor.create_default_pipeline()
        products = preprocessor.process(products)

        if sample and sample < len(products):
            random.seed(seed)
            products = random.sample(products, sample)
            logger.info(f"✅ {len(products)} produtos selecionados (amostra aleatória, seed={seed})")
        else:
            logger.info(f"✅ {len(products)} produtos carregados")

        # Inicializa orquestrador
        logger.info("\n🤖 Inicializando agentes...")
        orchestrator = AgentOrchestrator(
            llm_client=llm_client,
            max_workers=settings.agent.max_concurrent_agents,
        )

        profiles = orchestrator.get_agent_profiles()
        logger.info(f"✅ {len(profiles)} perfis ativos: {[p.value for p in profiles]}")
        if parallel_products:
            logger.info(f"⚡ Modo paralelo de produtos ativado (requer OLLAMA_NUM_PARALLEL > 1)")

        # Componentes de análise
        analyzer = ResultAnalyzer()
        recommendation_engine = RecommendationEngine()
        seller_message_generator = SellerMessageGenerator(llm_client=llm_client)

        # Inicializa banco de dados relacional (SQLite)
        db = DatabaseConnection(settings.database.path)
        db.create_tables()
        repo = EvaluationRepository(db)
        run_id = repo.save_run(
            llm_model=active_model,
            llm_temperature=settings.ollama.temperature,
            profiles_used=[p.value for p in profiles],
        )
        logger.info(f"🗄️ Run registrado no banco: {run_id}")

        # Inicializa banco de dados não relacional (MongoDB)
        mongo_repo = open_mongo_repository(settings)

        # Avalia todos os produtos (paralelo ou sequencial)
        logger.info("\n🔍 Iniciando avaliações multiagente...")
        evaluations_map = orchestrator.evaluate_products_batch(
            products,
            parallel_agents=parallel,
            parallel_products=parallel_products,
            max_product_workers=settings.agent.max_concurrent_agents,
        )

        # Pós-processamento: relatórios e recomendações
        for product, evaluation in zip(products[:len(evaluations_map)], evaluations_map):
            logger.info(f"\n{'='*60}")
            logger.info(f"Produto: {product.title[:50]}...")
            logger.info(f"Preço: R$ {product.current_price:.2f} (desconto: {product.discount_percentage:.0f}%)")
            logger.info("=" * 60)

            try:
                report = analyzer.generate_product_report(product, evaluation)

                logger.info(f"\n📊 Resultados:")
                logger.info(f"   Score Médio: {evaluation.mean_score:.1f}/10")
                logger.info(f"   Cobertura: {evaluation.coverage_score*100:.0f}%")
                logger.info(f"   Perfis Satisfeitos: {[p.value for p in evaluation.profiles_satisfied]}")
                logger.info(f"   Perfis Insatisfeitos: {[p.value for p in evaluation.profiles_unsatisfied]}")

                recommendations = recommendation_engine.generate_recommendations(
                    product, evaluation
                )

                if recommendations:
                    logger.info(f"\n💡 Top Recomendações (heurísticas):")
                    for rec in recommendations[:3]:
                        logger.info(f"   [{rec.priority}] {rec.title}")
                        if rec.action_items:
                            logger.info(f"       → {rec.action_items[0]}")

                seller_recommendation: dict | None = None
                try:
                    seller_recommendation = seller_message_generator.generate(report)
                    logger.info(
                        "\n📝 Mensagem ao vendedor gerada "
                        f"({len(seller_recommendation['message'])} chars)."
                    )
                except Exception as e:
                    logger.warning(f"Falha ao gerar mensagem ao vendedor: {e}")

                # Persiste no banco relacional (estrutura analítica)
                repo.save_report(run_id, report)

                # Persiste no banco não relacional (relatório completo + mensagem)
                if mongo_repo is not None:
                    mongo_repo.save_report(run_id, report, seller_recommendation)

            except Exception as e:
                logger.error(f"Erro ao processar relatório do produto {product.product_id}: {e}")

        repo.update_run_product_count(run_id)
        db.close()
        if mongo_repo is not None:
            mongo_repo.close()

        logger.info("\n" + "=" * 60)
        logger.info("✅ Processamento concluído!")
        logger.info(f"🗄️ SQLite: {settings.database.path}")
        if mongo_repo is not None:
            logger.info(
                f"🍃 MongoDB: {settings.mongo.uri} "
                f"db={settings.mongo.database} coll={settings.mongo.reports_collection}"
            )
        logger.info("=" * 60)


def run_single_product(product_id: str, model: str | None = None) -> None:
    """Avalia um produto específico."""
    settings = get_settings()
    active_model = model or settings.ollama.model

    with OllamaClient(
        host=settings.ollama.host,
        model=active_model,
        timeout=settings.ollama.timeout,
        max_retries=settings.ollama.max_retries,
        retry_backoff=settings.ollama.retry_backoff,
    ) as llm_client:

        if not check_ollama_connection(llm_client):
            return

        loader = DataLoader()
        products = loader.load_products()
        
        product = next((p for p in products if p.product_id == product_id), None)
        
        if not product:
            logger.error(f"Produto não encontrado: {product_id}")
            return

        orchestrator = AgentOrchestrator(llm_client=llm_client)
        evaluation = orchestrator.evaluate_product(product)

        analyzer = ResultAnalyzer()
        report = analyzer.generate_product_report(product, evaluation)

        # Mensagem ao vendedor (LLM)
        seller_recommendation: dict | None = None
        try:
            seller_recommendation = SellerMessageGenerator(llm_client=llm_client).generate(report)
        except Exception as e:
            logger.warning(f"Falha ao gerar mensagem ao vendedor: {e}")

        # Persiste no banco relacional
        db = DatabaseConnection(settings.database.path)
        db.create_tables()
        repo = EvaluationRepository(db)
        profiles = orchestrator.get_agent_profiles()
        run_id = repo.save_run(
            llm_model=active_model,
            llm_temperature=settings.ollama.temperature,
            profiles_used=[p.value for p in profiles],
            notes=f"single product: {product_id}",
        )
        repo.save_report(run_id, report)
        repo.update_run_product_count(run_id)
        db.close()

        # Persiste no MongoDB
        mongo_repo = open_mongo_repository(settings)
        if mongo_repo is not None:
            mongo_repo.save_report(run_id, report, seller_recommendation)
            mongo_repo.close()

        logger.info(f"🗄️ SQLite: {settings.database.path}")
        logger.info(
            f"🍃 MongoDB: {settings.mongo.uri} "
            f"db={settings.mongo.database} coll={settings.mongo.reports_collection}"
        )

        # Imprime mensagem ao vendedor para inspeção
        if seller_recommendation:
            print("\n" + "=" * 60)
            print("MENSAGEM AO VENDEDOR")
            print("=" * 60)
            print(seller_recommendation["message"])


def run_compare(
    models: list[str] | None = None,
    parallel: bool = True,
    parallel_products: bool = False,
    sample: int | None = None,
    seed: int = 42,
) -> None:
    """
    Executa todos os produtos com múltiplos modelos e exibe tabela comparativa.

    Args:
        models: Lista de modelos a comparar (padrão: SUPPORTED_MODELS).
        parallel: Executar agentes em paralelo por produto.
        parallel_products: Avaliar múltiplos produtos em paralelo.
        sample: Número de produtos a amostrar aleatoriamente (None = todos).
        seed: Semente para reprodutibilidade da amostragem.
    """
    models_to_run = models or SUPPORTED_MODELS
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("Comparação entre modelos")
    logger.info(f"Modelos: {models_to_run}")
    logger.info("=" * 60)

    # Carrega produtos uma única vez
    loader = DataLoader(
        data_path=settings.data.raw_path,
        dataset_file=settings.data.dataset_file,
    )
    try:
        products = loader.load_products()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    preprocessor = DataPreprocessor.create_default_pipeline()
    products = preprocessor.process(products)

    if sample and sample < len(products):
        random.seed(seed)
        products = random.sample(products, sample)
        logger.info(f"✅ {len(products)} produtos selecionados (amostra aleatória, seed={seed})")
    else:
        logger.info(f"✅ {len(products)} produtos carregados")

    analyzer = ResultAnalyzer()
    summary: dict[str, list[dict]] = {}

    # Inicializa banco relacional (SQLite) e não relacional (MongoDB)
    db = DatabaseConnection(settings.database.path)
    db.create_tables()
    repo = EvaluationRepository(db)
    mongo_repo = open_mongo_repository(settings)

    for model_name in models_to_run:
        logger.info(f"\n>>> Rodando com modelo: {model_name}")
        summary[model_name] = []

        with OllamaClient(
            host=settings.ollama.host,
            model=model_name,
            temperature=settings.ollama.temperature,
            timeout=settings.ollama.timeout,
            max_retries=settings.ollama.max_retries,
            retry_backoff=settings.ollama.retry_backoff,
        ) as llm_client:
            if not check_ollama_connection(llm_client):
                logger.warning(f"Modelo {model_name} indisponível, pulando.")
                continue

            orchestrator = AgentOrchestrator(
                llm_client=llm_client,
                max_workers=settings.agent.max_concurrent_agents,
            )
            seller_message_generator = SellerMessageGenerator(llm_client=llm_client)

            profiles = orchestrator.get_agent_profiles()
            run_id = repo.save_run(
                llm_model=model_name,
                llm_temperature=settings.ollama.temperature,
                profiles_used=[p.value for p in profiles],
                notes=f"compare run",
            )
            logger.info(f"🗄️ Run registrado no banco: {run_id}")

            evaluations = orchestrator.evaluate_products_batch(
                products,
                parallel_agents=parallel,
                parallel_products=parallel_products,
                max_product_workers=settings.agent.max_concurrent_agents,
            )
            for product, evaluation in zip(products[:len(evaluations)], evaluations):
                try:
                    report = analyzer.generate_product_report(product, evaluation)

                    seller_recommendation: dict | None = None
                    try:
                        seller_recommendation = seller_message_generator.generate(report)
                    except Exception as e:
                        logger.warning(
                            f"Falha ao gerar mensagem ao vendedor "
                            f"({model_name}/{product.product_id}): {e}"
                        )

                    repo.save_report(run_id, report)
                    if mongo_repo is not None:
                        mongo_repo.save_report(run_id, report, seller_recommendation)

                    summary[model_name].append({
                        "product_id": product.product_id,
                        "title": product.title[:40],
                        "mean_score": round(evaluation.mean_score, 2),
                        "coverage_pct": round(evaluation.coverage_score * 100, 1),
                    })
                except Exception as e:
                    logger.error(f"Erro em {model_name} / {product.product_id}: {e}")

            repo.update_run_product_count(run_id)

    db.close()
    if mongo_repo is not None:
        mongo_repo.close()

    # Exibe tabela comparativa
    logger.info("\n" + "=" * 60)
    logger.info("RESUMO COMPARATIVO")
    logger.info("=" * 60)
    for model_name, results in summary.items():
        if not results:
            continue
        avg_score = sum(r["mean_score"] for r in results) / len(results)
        avg_coverage = sum(r["coverage_pct"] for r in results) / len(results)
        logger.info(f"\nModelo: {model_name}")
        logger.info(f"  Score médio:    {avg_score:.2f}/10")
        logger.info(f"  Cobertura média: {avg_coverage:.1f}%")
        for r in results:
            logger.info(f"  [{r['product_id']}] {r['title'][:35]} → {r['mean_score']}/10  cob:{r['coverage_pct']}%")

    logger.info(f"\n🗄️ SQLite: {settings.database.path}")
    if mongo_repo is not None:
        logger.info(
            f"🍃 MongoDB: {settings.mongo.uri} "
            f"db={settings.mongo.database} coll={settings.mongo.reports_collection}"
        )


def main() -> None:
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Sistema Multiagente para Avaliação de Promoções"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")
    
    # Argumento global de modelo (disponível em todos os subcomandos via parents)
    model_parent = argparse.ArgumentParser(add_help=False)
    model_parent.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help=f"Modelo Ollama a utilizar. Suportados: {', '.join(SUPPORTED_MODELS)}",
    )

    # Argumentos de amostragem (compartilhado por demo e compare)
    sample_parent = argparse.ArgumentParser(add_help=False)
    sample_parent.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Selecionar N produtos aleatoriamente do dataset",
    )
    sample_parent.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Semente para reprodutibilidade da amostragem (default: 42)",
    )

    # Comando demo
    demo_parser = subparsers.add_parser("demo", help="Executa o sistema sobre todo o dataset", parents=[model_parent, sample_parent])
    demo_parser.add_argument(
        "--sequential",
        action="store_true",
        help="Executar agentes sequencialmente"
    )
    demo_parser.add_argument(
        "--parallel-products",
        action="store_true",
        help="Avaliar múltiplos produtos em paralelo (requer OLLAMA_NUM_PARALLEL > 1)"
    )

    # Comando evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Avalia produto específico", parents=[model_parent])
    eval_parser.add_argument(
        "product_id",
        help="ID do produto a avaliar"
    )

    # Comando compare
    compare_parser = subparsers.add_parser("compare", help="Compara múltiplos modelos sobre todo o dataset", parents=[sample_parent])
    compare_parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help=f"Modelos a comparar (default: {' '.join(SUPPORTED_MODELS)})"
    )
    compare_parser.add_argument(
        "--sequential",
        action="store_true",
        help="Executar agentes sequencialmente"
    )
    compare_parser.add_argument(
        "--parallel-products",
        action="store_true",
        help="Avaliar múltiplos produtos em paralelo (requer OLLAMA_NUM_PARALLEL > 1)"
    )

    # Comando check
    subparsers.add_parser("check", help="Verifica conexão com Ollama")

    args = parser.parse_args()

    setup_directories()

    match args.command:
        case "demo":
            run_demo(
                parallel=not args.sequential,
                parallel_products=args.parallel_products,
                model=args.model,
                sample=args.sample,
                seed=args.seed,
            )
        case "evaluate":
            run_single_product(args.product_id, model=args.model)
        case "compare":
            run_compare(
                models=args.models,
                parallel=not args.sequential,
                parallel_products=args.parallel_products,
                sample=args.sample,
                seed=args.seed,
            )
        case "check":
            settings = get_settings()
            with OllamaClient(host=settings.ollama.host) as client:
                if check_ollama_connection(client):
                    logger.info("✅ Sistema pronto para uso!")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

