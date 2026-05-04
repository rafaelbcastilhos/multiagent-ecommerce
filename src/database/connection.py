"""
Gerenciador de conexão com banco de dados SQLite.

Fornece interface de contexto para transações atômicas
e inicialização automática do schema.
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.database.schema import ALL_INDEXES, ALL_TABLES

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Gerenciador de conexão SQLite com suporte a transações.

    Mantém uma única conexão reutilizável e expõe um context manager
    para transações com commit/rollback automático.

    Attributes:
        db_path: Caminho do arquivo SQLite.
    """

    def __init__(self, db_path: Path | str) -> None:
        """
        Inicializa o gerenciador.

        Args:
            db_path: Caminho para o arquivo do banco de dados.
                     O diretório pai é criado automaticamente se não existir.
        """
        self.db_path = Path(db_path)
        self._connection: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection:
        """
        Retorna a conexão ativa, criando-a se necessário.

        Returns:
            Conexão SQLite configurada.
        """
        if self._connection is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")

            logger.info(f"Conexão SQLite estabelecida: {self.db_path}")

        return self._connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """
        Context manager para execução de operações em transação atômica.

        Realiza commit ao final do bloco. Em caso de exceção,
        executa rollback e re-lança o erro.

        Yields:
            Conexão SQLite ativa dentro da transação.

        Raises:
            Exception: Qualquer exceção ocorrida durante a transação,
                       após o rollback ser executado.

        Example:
            with db.transaction() as conn:
                conn.execute("INSERT INTO ...", (...))
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def create_tables(self) -> None:
        """
        Cria todas as tabelas e índices se ainda não existirem.

        Deve ser chamado uma vez na inicialização da aplicação.
        É idempotente — seguro para múltiplas chamadas.
        """
        with self.transaction() as conn:
            for ddl in ALL_TABLES:
                conn.execute(ddl)
            for idx in ALL_INDEXES:
                conn.execute(idx)

        logger.info("Schema do banco de dados inicializado.")

    def close(self) -> None:
        """Encerra a conexão com o banco de dados."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info("Conexão SQLite encerrada.")

    def __enter__(self) -> "DatabaseConnection":
        self._get_connection()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
