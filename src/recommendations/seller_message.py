"""
Gerador de mensagens de recomendação textuais para o vendedor.

Consome os campos top_concerns e top_strengths do relatório multiagente
e usa o LLM para sintetizar uma mensagem acionável que ajude o vendedor
a aumentar a qualidade da publicação no marketplace.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.core.llm_client import OllamaClient

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "Você é um especialista em e-commerce que ajuda vendedores a otimizar "
    "publicações de produtos em marketplaces (Mercado Livre, Amazon, Shopee). "
    "Sua resposta deve ser objetiva, profissional e baseada exclusivamente nos dados fornecidos. "
    "Não invente atributos do produto. Responda em português."
)


class SellerMessageGenerator:
    """
    Gera uma mensagem de recomendação para o vendedor a partir do relatório.

    Usa os pontos fracos (top_concerns) e fortes (top_strengths) de cada perfil
    psicológico para sintetizar um diagnóstico, ações priorizadas e pontos a
    manter.
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> None:
        self.llm = llm_client
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, report: dict[str, Any]) -> dict[str, Any]:
        """
        Gera a mensagem de recomendação para o vendedor.

        Args:
            report: Relatório no formato de ResultAnalyzer.generate_product_report.

        Returns:
            Dicionário com message, model, generated_at e os insumos usados.
        """
        prompt = self._build_prompt(report)

        response = self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        message = response.content.strip()
        logger.info(
            f"Mensagem ao vendedor gerada para produto={report['product']['id']} "
            f"({len(message)} chars)"
        )

        return {
            "message": message,
            "model": response.model,
            "generated_at": datetime.now().isoformat(),
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
        }

    @staticmethod
    def _build_prompt(report: dict[str, Any]) -> str:
        product = report.get("product", {})
        overall = report.get("overall_analysis", {})
        promotion = report.get("promotion_effectiveness", {})
        consensus = report.get("consensus", {})
        profile_results: dict[str, Any] = report.get("profile_results", {})

        concerns_lines: list[str] = []
        strengths_lines: list[str] = []
        profile_summary_lines: list[str] = []

        for profile, data in profile_results.items():
            score = data.get("score")
            status = data.get("status", "")
            intention = data.get("purchase_intention")
            profile_summary_lines.append(
                f"- {profile}: score {score}/10, intenção {intention}%, status: {status}"
            )
            for concern in data.get("top_concerns", []) or []:
                concerns_lines.append(f"- [{profile}] {concern}")
            for strength in data.get("top_strengths", []) or []:
                strengths_lines.append(f"- [{profile}] {strength}")

        disagreement_areas = consensus.get("disagreement_areas") or []
        consensus_strengths = consensus.get("strengths") or []
        consensus_weaknesses = consensus.get("weaknesses") or []

        return f"""Analise os dados abaixo e gere uma MENSAGEM DE RECOMENDAÇÃO ao vendedor para aumentar a qualidade da publicação deste produto.

DADOS DO PRODUTO
- Título: {product.get("title", "N/A")}
- Marca: {product.get("brand", "N/A")}
- Categoria: {product.get("category", "N/A")}
- Preço atual: R$ {product.get("current_price", 0):.2f}
- Preço original: R$ {product.get("original_price", 0):.2f}
- Desconto: {product.get("discount", 0):.0f}%
- Avaliação média: {product.get("rating", "N/A")}
- Número de reviews: {product.get("reviews", 0)}

INDICADORES AGREGADOS
- Score médio: {overall.get("mean_score", 0):.2f}/10 (faixa: {overall.get("score_range", "N/A")})
- Cobertura de perfis satisfeitos: {overall.get("coverage_score", 0):.0f}%
- Risco: {overall.get("risk_score", 0):.0f}%
- Nível de consenso entre perfis: {overall.get("consensus_level", 0):.0f}%
- Apelo geral da promoção: {promotion.get("overall_appeal", 0):.2f}/10
- Potencial de conversão: {promotion.get("conversion_potential", 0):.0f}%
- Potencial de melhoria: {promotion.get("improvement_potential", 0):.0f}%

RESULTADOS POR PERFIL DE COMPRADOR
{chr(10).join(profile_summary_lines) if profile_summary_lines else "- (nenhum)"}

PONTOS FRACOS APONTADOS (top_concerns por perfil)
{chr(10).join(concerns_lines) if concerns_lines else "- (nenhum)"}

PONTOS FORTES APONTADOS (top_strengths por perfil)
{chr(10).join(strengths_lines) if strengths_lines else "- (nenhum)"}

CONSENSO ENTRE PERFIS
- Forças em consenso: {", ".join(consensus_strengths) if consensus_strengths else "nenhuma"}
- Fraquezas em consenso: {", ".join(consensus_weaknesses) if consensus_weaknesses else "nenhuma"}
- Áreas de discordância: {", ".join(disagreement_areas) if disagreement_areas else "nenhuma"}

INSTRUÇÕES DE SAÍDA
Escreva uma mensagem direta para o vendedor (máximo 220 palavras) com EXATAMENTE estas duas seções, nesta ordem:

1) Abertura positiva (1-2 frases) — começa reconhecendo os pontos fortes reais do produto, derivados dos top_strengths. Use uma estrutura concessiva, por exemplo: "Apesar de ser uma peça de qualidade, com [força 1] e [força 2]..." ou "Mesmo apresentando [força], existem oportunidades para...". A frase deve fluir como um pivô natural para as melhorias listadas a seguir.

2) Top 3 oportunidades de melhoria — bloco com cabeçalho "Oportunidades de melhoria:" seguido de 3 itens numerados (1., 2., 3.). Cada item: ação concreta e específica + ganho esperado em uma frase. Priorize ações que atacam os top_concerns mais recorrentes entre os perfis.

REGRAS OBRIGATÓRIAS — SIGA TODAS SEM EXCEÇÃO
- A PRIMEIRA palavra da resposta DEVE ser a primeira palavra da abertura positiva (ex.: "Apesar", "Embora", "Mesmo"). NUNCA inicie com:
    * Saudações: "Olá", "Olá, vendedor", "Olá, [Nome do Vendedor]", "Prezado", "Caro", "Bom dia", "Boa tarde", "Oi".
    * Títulos ou cabeçalhos: "**Mensagem de Recomendação**", "Mensagem de Recomendação:", "Análise:", "Recomendação:", "Sumário:".
    * Qualquer linha em negrito, em maiúsculas ou entre asteriscos antes da abertura.
- A ÚLTIMA linha da resposta DEVE ser o terceiro item da lista de melhorias. NUNCA termine com:
    * Despedidas: "Atenciosamente", "Cordialmente", "Att.", "Abraços", "Obrigado", "Espero que ajude".
    * Assinaturas: "[Seu Nome]", "[Nome]", "Equipe ...", qualquer placeholder entre colchetes.
    * Frases de fechamento meta: "Essa é uma análise que pode ajudar a melhorar a publicação do produto e aumentar as vendas.", "Boa sorte com as vendas".
- NÃO exiba números de notas, scores ou avaliações no texto final. NÃO escreva valores como "6.5/10", "score médio de X", "nota 4.2", "70% de conversão", "35% de potencial de melhoria", nem cite quantidade de reviews. Use os números apenas internamente para priorizar; descreva qualitativamente (ex.: "boa aceitação geral", "baixa cobertura entre perfis", "alto potencial de ganho").
- NÃO use emojis. NÃO inclua disclaimers.
- NÃO cite nomes técnicos de perfis psicológicos (Ansioso, Exigente, Econômico, Impulsivo, Racional). Traduza em comportamento (ex.: "compradores que comparam preço", "compradores que precisam de detalhes técnicos").
- Use APENAS as informações fornecidas acima — não invente atributos do produto.

EXEMPLO DE INÍCIO VÁLIDO:
"Apesar de ser uma peça com material de boa qualidade e design atraente, há ajustes que podem aumentar a conversão da publicação."

EXEMPLO DE INÍCIO INVÁLIDO:
"**Mensagem de Recomendação**\n\nOlá, vendedor"

EXEMPLO DE FIM VÁLIDO:
"3. Inclua uma tabela de medidas detalhada para reduzir dúvidas e devoluções."

EXEMPLO DE FIM INVÁLIDO:
"3. \n\nAtenciosamente,\n[Seu Nome]"
"""
