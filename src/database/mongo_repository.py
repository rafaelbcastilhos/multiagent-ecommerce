"""
Repositório MongoDB para persistência de relatórios de avaliação.

Substitui a geração de arquivos JSON em outputs/reports/ por documentos
MongoDB que preservam a mesma estrutura de atributos do relatório original
e adicionam a recomendação textual gerada para o vendedor.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


class MongoEvaluationRepository:
    """
    Persiste relatórios de avaliação em uma coleção MongoDB.

    Cada documento corresponde a um par (run_id, product_id) e mantém a
    mesma estrutura aninhada usada nos relatórios JSON: product, overall_analysis,
    profile_results, consensus, promotion_effectiveness, metadata. Adiciona
    o campo seller_recommendation com a mensagem gerada para o vendedor.
    """

    def __init__(
        self,
        uri: str,
        database: str,
        collection: str,
        server_selection_timeout_ms: int = 3000,
    ) -> None:
        self._client: MongoClient = MongoClient(
            uri,
            serverSelectionTimeoutMS=server_selection_timeout_ms,
        )
        self._collection: Collection = self._client[database][collection]
        self._db_name = database
        self._coll_name = collection

    def ping(self) -> None:
        """Valida conectividade com o servidor; lança em caso de falha."""
        self._client.admin.command("ping")
        logger.info(
            f"MongoDB conectado: db='{self._db_name}' collection='{self._coll_name}'"
        )

    def ensure_indexes(self) -> None:
        """Garante índice único em (run_id, product.id) e índices de consulta."""
        self._collection.create_index(
            [("run_id", 1), ("product.id", 1)],
            unique=True,
            name="uniq_run_product",
        )
        self._collection.create_index([("run_id", 1)], name="run_id_idx")
        self._collection.create_index([("product.id", 1)], name="product_id_idx")

    def save_report(
        self,
        run_id: str,
        report: dict[str, Any],
        seller_recommendation: dict[str, Any] | None = None,
    ) -> str:
        """
        Persiste um relatório como documento MongoDB.

        Args:
            run_id: Identificador do run de execução.
            report: Estrutura gerada por ResultAnalyzer.generate_product_report.
            seller_recommendation: Mensagem de recomendação para o vendedor
                (campos: message, model, generated_at). Pode ser None.

        Returns:
            _id do documento persistido (compound: "<run_id>:<product_id>").
        """
        product_id = report["product"]["id"]
        doc_id = f"{run_id}:{product_id}"

        document: dict[str, Any] = {
            "_id": doc_id,
            "run_id": run_id,
            "saved_at": datetime.now().isoformat(),
            **deepcopy(report),
        }
        if seller_recommendation is not None:
            document["seller_recommendation"] = seller_recommendation

        self._collection.replace_one({"_id": doc_id}, document, upsert=True)
        logger.info(f"Relatório persistido em MongoDB: {doc_id}")
        return doc_id

    def get_report(self, run_id: str, product_id: str) -> dict[str, Any] | None:
        """Retorna um relatório específico ou None."""
        return self._collection.find_one({"run_id": run_id, "product.id": product_id})

    def get_reports_by_run(self, run_id: str) -> list[dict[str, Any]]:
        """Retorna todos os relatórios de um run, do maior para o menor mean_score."""
        cursor = self._collection.find({"run_id": run_id}).sort(
            "overall_analysis.mean_score", -1
        )
        return list(cursor)

    def close(self) -> None:
        """Encerra a conexão com o servidor MongoDB."""
        try:
            self._client.close()
            logger.info("Conexão MongoDB encerrada.")
        except PyMongoError as e:
            logger.warning(f"Erro ao encerrar conexão MongoDB: {e}")

    def __enter__(self) -> "MongoEvaluationRepository":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
