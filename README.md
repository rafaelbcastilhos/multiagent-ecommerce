# 🛒 Sistema multi-agente baseado em LLMs para avaliação de promoções em e-commerce

**Simulação de perfis psicológicos de compradores**

Trabalho de Conclusão de Curso — MBA em Inteligência Artificial e Big Data
Instituto de Ciências Matemáticas e de Computação — ICMC/USP, São Carlos, 2026
Autor: **Rafael Begnini de Castilhos** — Orientador: **Prof. Dr. Fernando Santos Osório**

---

## 📋 Objetivo

### Objetivo geral

Desenvolver um sistema multi-agente baseado em inteligência artificial capaz de simular perfis psicológicos variados de compradores para avaliar anúncios e promoções no segmento de moda, utilizando dados públicos e reais de um marketplace. O resultado é um mecanismo capaz de sugerir recomendações para o vendedor, visando aumentar a cobertura de satisfação de diferentes perfis de compradores e contribuir para estratégias promocionais mais inteligentes, personalizadas e alinhadas à complexidade comportamental do comércio eletrônico contemporâneo.

### Objetivos específicos

- Modelar cinco perfis psicológicos distintos baseados na literatura de comportamento do consumidor;
- Implementar uma arquitetura multi-agente integrada a LLMs locais via Ollama;
- Gerar recomendações acionáveis para otimização de publicações em marketplace.

---

## 🏗️ Arquitetura

O sistema é estruturado em **oito camadas funcionais**, cada uma com responsabilidades bem definidas e interfaces tipadas:

| # | Camada | Componentes | Responsabilidade |
|---|--------|-------------|------------------|
| 1 | **Entrada** | `DataLoader`, `DataPreprocessor` | Carrega o dataset, normaliza campos, infere atributos derivados |
| 2 | **Orquestração** | `AgentOrchestrator` | Instancia agentes e coordena execução paralela ou sequencial |
| 3 | **Agentes** | 5 perfis derivados de `BaseAgent` | Avaliação multicritério condicionada por perfil psicológico |
| 4 | **Inferência** | `OllamaClient` | Comunicação HTTP com o Ollama (saída JSON estruturada) |
| 5 | **Avaliação** | `EvaluationMetrics`, `ResultAnalyzer` | Métricas de cobertura, consenso e efetividade promocional |
| 6 | **Blending** | `ProfileBlender`, `CustomerVector` | Interpolação vetorial (IDW) para clientes em comportamentos híbridos |
| 7 | **Recomendações** | `RecommendationEngine`, `SellerMessageGenerator` | Heurísticas determinísticas + mensagem narrativa ao vendedor |
| 8 | **Saída** | `AggregatedEvaluation`, `Recommendation`, `SellerRecommendation` | Materialização dos resultados (relatório, recomendações, mensagem) |

```
tcc/
├── src/
│   ├── agents/              # Agentes com perfis psicológicos
│   │   ├── anxious_buyer.py     # Perfil ansioso
│   │   ├── demanding_buyer.py   # Perfil exigente
│   │   ├── economic_buyer.py    # Perfil econômico
│   │   ├── impulsive_buyer.py   # Perfil impulsivo
│   │   └── rational_buyer.py    # Perfil racional
│   │
│   ├── core/                # Componentes centrais
│   │   ├── llm_client.py        # Cliente Ollama
│   │   ├── base_agent.py        # Classe base dos agentes
│   │   └── orchestrator.py      # Orquestrador multiagente
│   │
│   ├── data/                # Pipeline de dados
│   │   ├── schemas.py           # Modelos Pydantic
│   │   ├── loader.py            # Carregador de dados
│   │   └── preprocessor.py      # Pré-processamento
│   │
│   ├── database/            # Persistência (MongoDB)
│   │   ├── connection.py        # Conexão com MongoDB
│   │   ├── mongo_repository.py  # Repositório de relatórios
│   │   ├── repository.py        # Interface de repositório
│   │   └── schema.py            # Schemas de persistência
│   │
│   ├── evaluation/          # Sistema de avaliação
│   │   ├── metrics.py           # Métricas de cobertura
│   │   ├── analyzer.py          # Análise de resultados
│   │   └── profile_blender.py   # Combinação de perfis
│   │
│   └── recommendations/     # Motor de recomendações
│       ├── engine.py            # Gerador de recomendações
│       └── seller_message.py    # Mensagens para o vendedor
│
├── config/                  # Configurações
│   └── settings.py              # Pydantic Settings
│
├── data/                    # Dados
│   ├── raw/                     # Dados brutos (dataset)
│   ├── processed/               # Dados processados
│   └── cache/                   # Cache local
│
├── outputs/                 # Saídas
│   ├── reports/                 # Relatórios JSON
│   └── recommendations/         # Recomendações
│
├── main.py                  # Ponto de entrada
├── pyproject.toml           # Dependências e configuração do projeto
├── poetry.lock              # Lock de dependências
└── README.md
```

---

## 🧠 Perfis Psicológicos

Cada perfil é caracterizado por **três parâmetros comportamentais contínuos** no intervalo `[0, 1]` — sensibilidade ao preço, foco em qualidade e aversão ao risco — e por um **vetor de pesos** sobre seis dimensões de avaliação do produto.

### Parâmetros comportamentais

| Perfil | Sensibilidade ao Preço | Foco em Qualidade | Aversão ao Risco | Característica dominante |
|--------|------------------------|-------------------|------------------|--------------------------|
| **Ansioso** | Baixa (0,30) | Alta (0,70) | Muito alta (0,95) | Reputação, garantias, prazo |
| **Exigente** | Muito baixa (0,20) | Muito alta (0,95) | Média (0,60) | Qualidade premium, descrição detalhada |
| **Econômico** | Muito alta (0,95) | Média (0,50) | Média (0,50) | Preço, custo-benefício, desconto |
| **Impulsivo** | Média (0,60) | Baixa (0,40) | Muito baixa (0,25) | Promoção, gatilhos visuais, urgência |
| **Racional** | Média (0,50) | Média (0,50) | Média (0,50) | Análise equilibrada de todos os fatores |

### Pesos por dimensão de avaliação

| Dimensão | Ansioso | Exigente | Econômico | Impulsivo | Racional |
|----------|:-------:|:--------:|:---------:|:---------:|:--------:|
| Preço             | 10% | 5%  | **40%** | 15% | 20% |
| Qualidade         | 15% | **40%** | 15% | 10% | 20% |
| Credibilidade     | **35%** | 15% | 10% | 10% | 20% |
| Completude visual | 25% | 10% | 15% | 20% | 15% |
| Descrição         | 10% | **25%** | 5%  | 10% | 15% |
| Promoção          | 5%  | 5%  | 15% | **35%** | 10% |

> Os parâmetros e pesos são aplicados na construção do *system prompt* enviado ao LLM, condicionando o agente a raciocinar a partir das prioridades características do perfil.

---

## 🎯 Blending Vetorial de Perfis

Compradores reais não se enquadram rigidamente em um único perfil — posicionam-se ao longo de um espectro contínuo. O `ProfileBlender` trata os cinco perfis como **âncoras (centroides) num espaço tridimensional** e interpola as avaliações por **Inverse Distance Weighting (IDW)**:

$$w_i = \frac{d(c, p_i)^{-k}}{\sum_{j=1}^{N} d(c, p_j)^{-k}}$$

- `c` é o vetor comportamental do cliente em `[0, 1]³`
- `p_i` é o centroide do perfil `i`
- `k` é o expoente de concentração (`k=1` distribui suavemente, `k→∞` aproxima do perfil mais próximo)

O blending opera **sem chamadas adicionais ao LLM** — reaproveita as avaliações já produzidas pelos cinco agentes, ponderando-as pelos pesos `w_i`. Perfis com `w_i ≥ 0,15` são classificados como dominantes para fins de interpretação qualitativa.

---

## 💾 Persistência híbrida

O sistema combina dois bancos de dados com finalidades complementares:

- **SQLite** — armazena a estrutura analítica normalizada (`evaluation_runs`, `product_evaluations`, `profile_evaluations`, `consensus_items`, etc.) ideal para consultas comparativas e *joins* entre execuções.
- **MongoDB** — armazena o relatório completo como documento aninhado (`EvaluationReport`), preservando a estrutura original do `AggregatedEvaluation` e incorporando a `SellerRecommendation` gerada pelo LLM.

Essa divisão evita fragmentar campos textuais variáveis em tabelas relacionais e elimina o custo de reconstruir o relatório por *joins* sempre que ele precisa ser exibido na íntegra.

---

## 🚀 Instalação

### 1. Pré-requisitos

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation) para gerenciamento de dependências
- [Ollama](https://ollama.ai) instalado e rodando
- (Opcional) MongoDB local ou remoto, caso deseje persistir os relatórios documentais

### 2. Clonar e instalar

```bash
# Clone o repositório
git clone <seu-repo>
cd tcc

# Instale as dependências (Poetry cuida do virtualenv automaticamente)
poetry install --no-root

# Ative o shell do ambiente quando quiser rodar comandos diretamente
poetry shell
```

> Todas as dependências do projeto são definidas no `pyproject.toml` e travadas no `poetry.lock`. Não há mais `requirements.txt`.

### 3. Configurar Ollama

Os experimentos de referência utilizam dois modelos executados localmente — **Llama 3.2** (Meta AI) e **Qwen 2.5 7B** (Alibaba Cloud) — para permitir avaliação comparativa entre arquiteturas.

```bash
# Modelos utilizados nos experimentos comparativos
ollama pull llama3.2
ollama pull qwen2.5:7b

# Verificar se estão disponíveis
ollama list
```

A inferência local foi escolhida por: ausência de custos por requisição, privacidade dos dados, controle sobre a versão do modelo (reprodutibilidade) e independência de serviços externos.

### 4. Baixar Dataset

Baixe o dataset do Kaggle:
- [Flipkart Fashion Products Dataset](https://www.kaggle.com/datasets/aaditshukla/flipkart-fasion-products-dataset)

Salve em `data/raw/flipkart_fashion_products_dataset.json`

### 5. Configurar ambiente

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite conforme necessário
nano .env
```

---

## 💻 Uso

> Os exemplos abaixo assumem que o virtualenv do Poetry está ativo (`poetry shell`). Caso contrário, prefixe os comandos com `poetry run`.

### Verificar conexão com Ollama

```bash
python main.py check
```

### Executar demonstração

```bash
# Demo sobre o dataset completo
python main.py demo

# Execução sequencial (mais lenta, menos recursos)
python main.py demo --sequential
```

### Avaliar produto específico

```bash
python main.py evaluate PROD_000001
```

### Comparar múltiplos modelos

Compara os modelos definidos (default: `llama3.2` e `qwen2.5:7b`) sobre uma amostra reproduzível do dataset.

```bash
# Amostra de 30 produtos com seed fixa, avaliando produtos em paralelo
OLLAMA_NUM_PARALLEL=3 python main.py compare \
    --sample 30 \
    --seed 42 \
    --parallel-products

# Especificando os modelos a comparar
python main.py compare --models llama3.2 mistral --sample 50
```

> A flag `--parallel-products` exige `OLLAMA_NUM_PARALLEL > 1` para que o Ollama atenda múltiplos produtos simultaneamente.

---

## 📊 Saídas

### Relatório de Produto

```json
{
  "product": {
    "id": "PROD_000001",
    "title": "Camiseta Premium Cotton",
    "current_price": 89.90,
    "discount": 35
  },
  "overall_analysis": {
    "mean_score": 7.2,
    "coverage_score": 60,
    "consensus_level": 78
  },
  "profile_results": {
    "anxious": {"score": 6.5, "status": "Neutro"},
    "demanding": {"score": 8.1, "status": "Satisfeito"},
    "economic": {"score": 7.8, "status": "Satisfeito"},
    "impulsive": {"score": 7.5, "status": "Satisfeito"},
    "rational": {"score": 6.1, "status": "Neutro"}
  }
}
```

### Recomendações

```json
{
  "priority": 1,
  "category": "trust",
  "title": "Fortalecer sinais de confiança",
  "action_items": [
    "Destacar política de devolução claramente",
    "Exibir certificações e selos de qualidade",
    "Mostrar depoimentos de clientes satisfeitos"
  ],
  "affected_profiles": ["anxious"],
  "estimated_improvement": 30
}
```

---

## 🔧 Configurações

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `OLLAMA_HOST` | URL do servidor Ollama | `http://localhost:11434` |
| `OLLAMA_MODEL` | Modelo LLM | `llama3.2` |
| `OLLAMA_TEMPERATURE` | Criatividade (0-2) | `0.7` |
| `AGENT_MAX_CONCURRENT_AGENTS` | Agentes paralelos | `3` |

---

## 📈 Métricas

### Cobertura de Perfis
- **Coverage Rate**: % de perfis satisfeitos (score ≥ 7)
- **Risk Rate**: % de perfis insatisfeitos (score < 5)
- **Diversity Score**: Variação entre perfis

### Consenso
- **Agreement Score**: Similaridade entre avaliações
- **Disagreement Areas**: Dimensões com maior divergência

### Efetividade da Promoção
- **Overall Appeal**: Apelo geral (0-10)
- **Conversion Potential**: Potencial de conversão estimado

---

## 🛠️ Tecnologias

- **Python 3.10+**: Linguagem principal
- **Poetry**: Gerenciamento de dependências e virtualenv
- **Pydantic v2 / pydantic-settings**: *Schemas* imutáveis (`frozen=True`) e configuração
- **Ollama**: Servidor local de LLMs (Llama 3.2, Qwen 2.5, Mistral)
- **httpx**: Cliente HTTP para o Ollama com *timeouts* configuráveis
- **SQLite**: Estrutura analítica normalizada para consultas comparativas
- **MongoDB / PyMongo**: Persistência documental dos relatórios completos
- **Pandas / NumPy**: Carregamento e manipulação inicial do *dataset*


