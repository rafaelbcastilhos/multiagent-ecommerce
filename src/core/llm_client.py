"""
Cliente para integração com Ollama LLM.

Fornece interface simplificada para comunicação com modelos
LLM locais via Ollama API.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
)


@dataclass(frozen=True)
class LLMResponse:
    """Resposta do modelo LLM."""

    content: str
    model: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    done: bool


class OllamaClient:
    """
    Cliente para comunicação com Ollama API.

    Suporta geração de texto com modelos locais como Llama, Mistral, etc.

    Attributes:
        host: URL do servidor Ollama.
        model: Nome do modelo a ser utilizado.
        temperature: Temperatura para geração (criatividade).
        max_tokens: Máximo de tokens na resposta.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 120,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        """
        Inicializa o cliente Ollama.

        Args:
            host: URL do servidor Ollama.
            model: Modelo LLM a ser utilizado.
            temperature: Temperatura para geração (0.0-2.0).
            max_tokens: Máximo de tokens na resposta.
            timeout: Timeout em segundos para requisições.
            max_retries: Tentativas totais em caso de timeout/conexão.
            retry_backoff: Base em segundos para backoff exponencial
                (2.0 => espera 2s, 4s, 8s entre tentativas).
        """
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._client = httpx.Client(timeout=timeout)

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Fecha o cliente HTTP."""
        self._client.close()

    def is_available(self) -> bool:
        """
        Verifica se o servidor Ollama está disponível.

        Returns:
            True se o servidor está respondendo, False caso contrário.
        """
        try:
            response = self._client.get(f"{self.host}/api/tags")
            return response.status_code == 200
        except httpx.RequestError as e:
            logger.warning(f"Ollama não disponível: {e}")
            return False

    def list_models(self) -> list[str]:
        """
        Lista modelos disponíveis no Ollama.

        Returns:
            Lista de nomes dos modelos instalados.

        Raises:
            ConnectionError: Se não conseguir conectar ao Ollama.
        """
        try:
            response = self._client.get(f"{self.host}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except httpx.RequestError as e:
            raise ConnectionError(f"Erro ao listar modelos: {e}") from e

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        format_json: bool = False,
    ) -> LLMResponse:
        """
        Gera texto usando o modelo LLM.

        Args:
            prompt: Prompt do usuário.
            system_prompt: Prompt de sistema (contexto/instruções).
            temperature: Override da temperatura padrão.
            max_tokens: Override do máximo de tokens.
            format_json: Se True, força resposta em formato JSON.

        Returns:
            LLMResponse com o texto gerado e metadados.

        Raises:
            ConnectionError: Se não conseguir conectar ao Ollama.
            ValueError: Se a resposta for inválida.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        if format_json:
            payload["format"] = "json"

        logger.debug(f"Enviando prompt para {self.model}: {prompt[:100]}...")
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.post(
                    f"{self.host}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                return LLMResponse(
                    content=data.get("response", ""),
                    model=data.get("model", self.model),
                    total_tokens=data.get("eval_count", 0) + data.get("prompt_eval_count", 0),
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    completion_tokens=data.get("eval_count", 0),
                    done=data.get("done", True),
                )

            except _RETRYABLE_EXCEPTIONS as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                wait = self.retry_backoff ** attempt
                logger.warning(
                    f"Ollama timeout/conexão (tentativa {attempt}/{self.max_retries}): {e}. "
                    f"Retry em {wait:.1f}s."
                )
                time.sleep(wait)
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama retornou erro HTTP {e.response.status_code}: {e}")
                raise ConnectionError(f"Ollama HTTP {e.response.status_code}: {e}") from e
            except httpx.RequestError as e:
                logger.error(f"Erro de conexão com Ollama: {e}")
                raise ConnectionError(f"Erro ao conectar com Ollama: {e}") from e
            except json.JSONDecodeError as e:
                logger.error(f"Resposta inválida do Ollama: {e}")
                raise ValueError(f"Resposta JSON inválida: {e}") from e

        logger.error(f"Ollama falhou após {self.max_retries} tentativas: {last_exc}")
        raise ConnectionError(
            f"Ollama indisponível após {self.max_retries} tentativas: {last_exc}"
        ) from last_exc


