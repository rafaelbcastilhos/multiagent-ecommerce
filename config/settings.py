"""
Configurações globais do sistema.

Utiliza pydantic-settings para gerenciamento de configurações
com suporte a variáveis de ambiente.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


SUPPORTED_MODELS: list[str] = [
    "llama3.2",
    "qwen2.5:7b",
]


class OllamaSettings(BaseSettings):
    """Configurações do cliente Ollama."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    host: str = Field(default="http://localhost:11434", description="URL do servidor Ollama")
    model: str = Field(default="llama3.2", description="Modelo LLM a ser utilizado")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperatura para geração")
    max_tokens: int = Field(default=2048, ge=1, description="Máximo de tokens na resposta")
    timeout: int = Field(default=600, ge=1, description="Timeout em segundos por requisição")
    max_retries: int = Field(default=3, ge=1, description="Tentativas totais em caso de timeout/conexão")
    retry_backoff: float = Field(default=2.0, ge=1.0, description="Base do backoff exponencial entre retries")


class DataSettings(BaseSettings):
    """Configurações de dados."""

    model_config = SettingsConfigDict(env_prefix="DATA_")

    raw_path: Path = Field(default=Path("data/raw"), description="Caminho dos dados brutos")
    processed_path: Path = Field(default=Path("data/processed"), description="Caminho dos dados processados")
    cache_path: Path = Field(default=Path("data/cache"), description="Caminho do cache")
    dataset_file: str = Field(default="flipkart_fashion_products_dataset.json", description="Nome do arquivo do dataset")


class AgentSettings(BaseSettings):
    """Configurações dos agentes."""

    model_config = SettingsConfigDict(env_prefix="AGENT_")

    max_concurrent_agents: int = Field(default=5, ge=1, le=10, description="Máximo de agentes concorrentes")
    evaluation_rounds: int = Field(default=3, ge=1, description="Rodadas de avaliação por agente")
    consensus_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Limiar para consenso")


class DatabaseSettings(BaseSettings):
    """Configurações do banco de dados SQLite."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    path: Path = Field(
        default=Path("data/evaluations.db"),
        description="Caminho do arquivo SQLite",
    )


class MongoSettings(BaseSettings):
    """Configurações do MongoDB (persistência de relatórios e recomendações)."""

    model_config = SettingsConfigDict(env_prefix="MONGO_")

    uri: str = Field(
        default="mongodb://localhost:27017",
        description="URI de conexão com o MongoDB",
    )
    database: str = Field(
        default="tcc_evaluations",
        description="Nome do banco de dados",
    )
    reports_collection: str = Field(
        default="evaluation_reports",
        description="Nome da coleção de relatórios",
    )
    server_selection_timeout_ms: int = Field(
        default=3000,
        ge=100,
        description="Timeout em ms para seleção de servidor",
    )


class LoggingSettings(BaseSettings):
    """Configurações de logging."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Nível de logging"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Formato do log",
    )
    file_path: Path | None = Field(default=None, description="Arquivo de log (opcional)")


class Settings(BaseSettings):
    """Configurações principais da aplicação."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-configurações
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    mongo: MongoSettings = Field(default_factory=MongoSettings)

    # Configurações gerais
    project_name: str = Field(default="TCC-MultiAgent-Buyers", description="Nome do projeto")
    debug: bool = Field(default=False, description="Modo debug")
    vertical: Literal["fashion", "sports", "electronics"] = Field(
        default="fashion", description="Vertical de produtos"
    )


def get_settings() -> Settings:
    """
    Retorna instância singleton das configurações.

    Returns:
        Settings: Objeto de configurações da aplicação.
    """
    return Settings()

