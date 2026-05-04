"""
Repositório de dados do sistema de avaliações multiagente.

Implementa operações de persistência para runs de execução,
produtos, avaliações agregadas e resultados por perfil psicológico.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.database.connection import DatabaseConnection

logger = logging.getLogger(__name__)


class EvaluationRepository:
    """
    Repositório para persistência de avaliações multiagente.

    Encapsula todas as operações de leitura e escrita no banco de dados,
    expondo uma interface de alto nível desacoplada do SQL subjacente.

    Attributes:
        db: Gerenciador de conexão com o banco de dados.
    """

    def __init__(self, db: DatabaseConnection) -> None:
        """
        Inicializa o repositório.

        Args:
            db: Instância de DatabaseConnection já inicializada.
        """
        self.db = db

    # ------------------------------------------------------------------ #
    # Runs                                                                 #
    # ------------------------------------------------------------------ #

    def save_run(
        self,
        llm_model: str,
        llm_temperature: float = 0.7,
        profiles_used: list[str] | None = None,
        notes: str = "",
        run_id: str | None = None,
    ) -> str:
        """
        Registra um novo run de avaliação.

        Args:
            llm_model: Nome do modelo LLM utilizado (ex: "llama3.2").
            llm_temperature: Temperatura de geração configurada.
            profiles_used: Lista de perfis psicológicos analisados.
            notes: Observações opcionais sobre o run.
            run_id: ID customizado; gerado automaticamente (UUID4) se None.

        Returns:
            run_id do registro criado.
        """
        run_id = run_id or str(uuid.uuid4())
        profiles_json = json.dumps(profiles_used or [])

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO evaluation_runs
                    (run_id, llm_model, llm_temperature, profiles_used, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    llm_model,
                    llm_temperature,
                    profiles_json,
                    notes,
                    datetime.now().isoformat(),
                ),
            )

        logger.info(f"Run registrado: {run_id} [{llm_model}]")
        return run_id

    def update_run_product_count(self, run_id: str) -> None:
        """
        Atualiza o contador de produtos avaliados em um run.

        Args:
            run_id: ID do run a ser atualizado.
        """
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE evaluation_runs
                SET product_count = (
                    SELECT COUNT(*) FROM product_evaluations WHERE run_id = ?
                )
                WHERE run_id = ?
                """,
                (run_id, run_id),
            )

    # ------------------------------------------------------------------ #
    # Relatórios                                                           #
    # ------------------------------------------------------------------ #

    def save_report(self, run_id: str, report: dict[str, Any]) -> int:
        """
        Persiste um relatório JSON completo em uma transação atômica.

        Insere ou atualiza o produto e registra a avaliação agregada,
        os resultados por perfil (com concerns e strengths) e os itens
        de consenso em uma única transação.

        Args:
            run_id: ID do run ao qual este relatório pertence.
            report: Dicionário no formato gerado por ResultAnalyzer.generate_product_report.

        Returns:
            ID do registro em product_evaluations criado.

        Raises:
            sqlite3.IntegrityError: Se a combinação (run_id, product_id) já existir.
            KeyError: Se campos obrigatórios estiverem ausentes no relatório.
        """
        product_data = report["product"]
        overall = report["overall_analysis"]
        promotion = report["promotion_effectiveness"]
        metadata = report.get("metadata", {})

        min_score, max_score = self._parse_score_range(overall.get("score_range", ""))

        with self.db.transaction() as conn:
            self._upsert_product(conn, product_data)

            cursor = conn.execute(
                """
                INSERT INTO product_evaluations (
                    run_id, product_id,
                    mean_score, min_score, max_score,
                    coverage_score, risk_score, consensus_level,
                    overall_appeal, conversion_potential, improvement_potential,
                    profiles_analyzed, generated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    product_data["id"],
                    overall.get("mean_score"),
                    min_score,
                    max_score,
                    overall.get("coverage_score"),
                    overall.get("risk_score"),
                    overall.get("consensus_level"),
                    promotion.get("overall_appeal"),
                    promotion.get("conversion_potential"),
                    promotion.get("improvement_potential"),
                    metadata.get("profiles_analyzed"),
                    metadata.get("generated_at"),
                ),
            )
            product_eval_id: int = cursor.lastrowid  # type: ignore[assignment]

            self._insert_profile_evaluations(
                conn, product_eval_id, report.get("profile_results", {})
            )

            consensus = report.get("consensus", {})
            self._insert_consensus_items(
                conn, product_eval_id, "strength", consensus.get("strengths", [])
            )
            self._insert_consensus_items(
                conn, product_eval_id, "weakness", consensus.get("weaknesses", [])
            )
            self._insert_consensus_items(
                conn,
                product_eval_id,
                "disagreement_area",
                consensus.get("disagreement_areas", []),
            )

        logger.info(
            f"Relatório salvo: produto={product_data['id']} run={run_id} id={product_eval_id}"
        )
        return product_eval_id

    def import_reports_from_directory(
        self,
        reports_dir: Path | str,
        run_id: str,
        skip_existing: bool = True,
    ) -> dict[str, int]:
        """
        Importa todos os relatórios JSON de um diretório para um run.

        Ignora o arquivo statistical_analysis.json automaticamente.

        Args:
            reports_dir: Diretório contendo os arquivos de relatório.
            run_id: ID do run de destino (deve ser criado previamente com save_run).
            skip_existing: Se True, ignora arquivos já importados neste run
                           em vez de lançar erro.

        Returns:
            Dicionário com contagens finais:
            {"imported": N, "skipped": N, "failed": N}.
        """
        reports_dir = Path(reports_dir)
        counts: dict[str, int] = {"imported": 0, "skipped": 0, "failed": 0}

        json_files = sorted(
            f for f in reports_dir.glob("*.json") if f.stem != "statistical_analysis"
        )

        logger.info(f"Importando {len(json_files)} relatório(s) de '{reports_dir}'")

        for json_file in json_files:
            try:
                with open(json_file, encoding="utf-8") as f:
                    report = json.load(f)

                if "product" not in report or "overall_analysis" not in report:
                    logger.warning(f"Formato inválido, ignorando: {json_file.name}")
                    counts["failed"] += 1
                    continue

                self.save_report(run_id, report)
                counts["imported"] += 1

            except sqlite3.IntegrityError:
                if skip_existing:
                    logger.debug(f"Já importado, ignorando: {json_file.name}")
                    counts["skipped"] += 1
                else:
                    raise
            except Exception as e:
                logger.error(f"Erro ao importar '{json_file.name}': {e}")
                counts["failed"] += 1

        self.update_run_product_count(run_id)
        logger.info(f"Importação concluída: {counts}")
        return counts

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def get_runs(self) -> list[dict[str, Any]]:
        """
        Retorna todos os runs registrados, do mais recente ao mais antigo.

        Returns:
            Lista de dicionários com os dados de cada run.
        """
        with self.db.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM evaluation_runs ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_product_evaluations(self, run_id: str) -> list[dict[str, Any]]:
        """
        Retorna as avaliações de produtos de um run, com dados do produto.

        Args:
            run_id: ID do run desejado.

        Returns:
            Lista de avaliações ordenadas por mean_score decrescente.
        """
        with self.db.transaction() as conn:
            rows = conn.execute(
                """
                SELECT pe.*, p.title, p.brand, p.category
                FROM product_evaluations pe
                JOIN products p ON p.product_id = pe.product_id
                WHERE pe.run_id = ?
                ORDER BY pe.mean_score DESC
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_profile_evaluations(self, product_evaluation_id: int) -> list[dict[str, Any]]:
        """
        Retorna avaliações por perfil de uma product_evaluation.

        Args:
            product_evaluation_id: ID do registro em product_evaluations.

        Returns:
            Lista de avaliações por perfil com concerns e strengths aninhados.
        """
        with self.db.transaction() as conn:
            profiles = conn.execute(
                "SELECT * FROM profile_evaluations WHERE product_evaluation_id = ?",
                (product_evaluation_id,),
            ).fetchall()

            result = []
            for profile in profiles:
                profile_dict = dict(profile)

                concerns = conn.execute(
                    "SELECT concern FROM profile_concerns WHERE profile_evaluation_id = ? ORDER BY position",
                    (profile_dict["id"],),
                ).fetchall()

                strengths = conn.execute(
                    "SELECT strength FROM profile_strengths WHERE profile_evaluation_id = ? ORDER BY position",
                    (profile_dict["id"],),
                ).fetchall()

                profile_dict["concerns"] = [r["concern"] for r in concerns]
                profile_dict["strengths"] = [r["strength"] for r in strengths]
                result.append(profile_dict)

        return result

    def compare_runs(self, run_id_a: str, run_id_b: str) -> list[dict[str, Any]]:
        """
        Compara resultados de dois runs para os produtos em comum.

        Args:
            run_id_a: ID do run de referência (ex: modelo anterior).
            run_id_b: ID do run de comparação (ex: modelo novo).

        Returns:
            Lista de produtos com scores de ambos os runs e o delta,
            ordenada pela maior diferença absoluta.
        """
        with self.db.transaction() as conn:
            rows = conn.execute(
                """
                SELECT
                    p.product_id,
                    p.title,
                    a.mean_score     AS score_a,
                    b.mean_score     AS score_b,
                    (b.mean_score - a.mean_score) AS score_delta,
                    a.coverage_score AS coverage_a,
                    b.coverage_score AS coverage_b
                FROM products p
                JOIN product_evaluations a
                    ON a.product_id = p.product_id AND a.run_id = ?
                JOIN product_evaluations b
                    ON b.product_id = p.product_id AND b.run_id = ?
                ORDER BY ABS(b.mean_score - a.mean_score) DESC
                """,
                (run_id_a, run_id_b),
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Helpers privados                                                     #
    # ------------------------------------------------------------------ #

    def _upsert_product(
        self, conn: sqlite3.Connection, product: dict[str, Any]
    ) -> None:
        """
        Insere ou atualiza um produto dentro de uma transação ativa.

        Args:
            conn: Conexão ativa da transação corrente.
            product: Dados do produto no formato do relatório JSON.
        """
        conn.execute(
            """
            INSERT INTO products
                (product_id, title, brand, category,
                 current_price, original_price, discount_pct, rating, review_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_id) DO UPDATE SET
                title          = excluded.title,
                brand          = excluded.brand,
                category       = excluded.category,
                current_price  = excluded.current_price,
                original_price = excluded.original_price,
                discount_pct   = excluded.discount_pct,
                rating         = excluded.rating,
                review_count   = excluded.review_count
            """,
            (
                product["id"],
                product.get("title", ""),
                product.get("brand", ""),
                product.get("category", ""),
                product.get("current_price"),
                product.get("original_price"),
                product.get("discount"),
                product.get("rating"),
                product.get("reviews", 0),
            ),
        )

    def _insert_profile_evaluations(
        self,
        conn: sqlite3.Connection,
        product_eval_id: int,
        profile_results: dict[str, Any],
    ) -> None:
        """
        Insere avaliações por perfil com concerns e strengths.

        Args:
            conn: Conexão ativa da transação corrente.
            product_eval_id: FK para product_evaluations.
            profile_results: Dicionário de perfis do relatório JSON.
        """
        for profile_name, data in profile_results.items():
            cursor = conn.execute(
                """
                INSERT INTO profile_evaluations
                    (product_evaluation_id, profile, score, purchase_intention, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    product_eval_id,
                    profile_name,
                    data.get("score"),
                    data.get("purchase_intention"),
                    data.get("status"),
                ),
            )
            profile_eval_id: int = cursor.lastrowid  # type: ignore[assignment]

            for i, concern in enumerate(data.get("top_concerns", [])):
                conn.execute(
                    "INSERT INTO profile_concerns (profile_evaluation_id, position, concern) VALUES (?, ?, ?)",
                    (profile_eval_id, i, concern),
                )

            for i, strength in enumerate(data.get("top_strengths", [])):
                conn.execute(
                    "INSERT INTO profile_strengths (profile_evaluation_id, position, strength) VALUES (?, ?, ?)",
                    (profile_eval_id, i, strength),
                )

    def _insert_consensus_items(
        self,
        conn: sqlite3.Connection,
        product_eval_id: int,
        item_type: str,
        items: list[str],
    ) -> None:
        """
        Insere itens de consenso de um tipo específico.

        Args:
            conn: Conexão ativa da transação corrente.
            product_eval_id: FK para product_evaluations.
            item_type: Tipo do item ('strength', 'weakness', 'disagreement_area').
            items: Lista de valores a inserir.
        """
        for i, value in enumerate(items):
            conn.execute(
                "INSERT INTO consensus_items (product_evaluation_id, type, position, value) VALUES (?, ?, ?, ?)",
                (product_eval_id, item_type, i, value),
            )

    @staticmethod
    def _parse_score_range(score_range: str) -> tuple[float | None, float | None]:
        """
        Parseia a string de intervalo de score do relatório.

        Args:
            score_range: String no formato "4.0 - 7.0".

        Returns:
            Tupla (min_score, max_score) ou (None, None) se inválida.
        """
        try:
            parts = score_range.split(" - ")
            return float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            return None, None
