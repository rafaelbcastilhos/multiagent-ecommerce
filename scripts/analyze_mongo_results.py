"""
Análise dos resultados gravados no MongoDB (`tcc_evaluations.evaluation_reports`).

Gera nuvens de palavras, gráficos comparativos entre os perfis psicológicos e os
modelos LLM, além de visualizações textuais (consenso, recomendações ao vendedor)
para enriquecer o capítulo de Resultados da monografia.

Uso:
    python scripts/analyze_mongo_results.py
    python scripts/analyze_mongo_results.py --uri mongodb://localhost:27017 --db tcc_evaluations

Saídas:
    outputs/charts/mongo_*.png
    outputs/tables/mongo_*.csv
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from pymongo import MongoClient
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "outputs" / "tables"
CHARTS_DIR = ROOT / "outputs" / "charts"

PROFILE_LABELS = {
    "anxious": "Ansioso",
    "demanding": "Exigente",
    "economic": "Econômico",
    "impulsive": "Impulsivo",
    "rational": "Racional",
}
PROFILE_ORDER = ["anxious", "demanding", "economic", "impulsive", "rational"]
PROFILE_COLORS = {
    "anxious": "#E57373",
    "demanding": "#7E57C2",
    "economic": "#26A69A",
    "impulsive": "#FFB300",
    "rational": "#42A5F5",
}
STATUS_ORDER = [
    "Muito Insatisfeito",
    "Insatisfeito",
    "Neutro",
    "Satisfeito",
    "Muito Satisfeito",
]
STATUS_COLORS = {
    "Muito Insatisfeito": "#8B0000",
    "Insatisfeito": "#E57373",
    "Neutro": "#BDBDBD",
    "Satisfeito": "#81C784",
    "Muito Satisfeito": "#1B5E20",
}
MODEL_COLORS = {"llama3.2": "#1f77b4", "qwen2.5:7b": "#ff7f0e"}

PT_STOPWORDS = {
    "a", "à", "às", "ao", "aos", "as", "o", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos", "em", "na", "no", "nas", "nos", "por",
    "pelo", "pela", "pelos", "pelas", "para", "com", "sem", "sob", "sobre",
    "entre", "ate", "até", "e", "ou", "mas", "que", "se", "como", "quando",
    "porque", "porquê", "qual", "quais", "quem", "onde", "ja", "já", "nao",
    "não", "sim", "muito", "muita", "muitos", "muitas", "pouco", "pouca",
    "ser", "é", "foi", "são", "era", "eram", "estar", "está", "estão", "ter",
    "tem", "têm", "tinha", "havia", "há", "esse", "essa", "esses", "essas",
    "este", "esta", "estes", "estas", "isso", "isto", "aquele", "aquela",
    "aqueles", "aquelas", "aquilo", "seu", "sua", "seus", "suas", "meu",
    "minha", "meus", "minhas", "nosso", "nossa", "nossos", "nossas", "lhe",
    "lhes", "me", "te", "se", "nos", "vos", "ele", "ela", "eles", "elas",
    "também", "tambem", "ainda", "ja", "já", "mais", "menos", "muito", "bem",
    "mal", "tudo", "todo", "toda", "todos", "todas", "outra", "outro",
    "outras", "outros", "etc", "ex", "r$", "rs", "produto", "produtos",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )
    sns.set_style("whitegrid", {"axes.spines.top": False, "axes.spines.right": False})


def save_table(df: pd.DataFrame, name: str) -> None:
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=False, float_format="%.3f")
    print(f"  tabela: {path.relative_to(ROOT)}")


def save_figure(fig: plt.Figure, name: str) -> None:
    path = CHARTS_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  figura: {path.relative_to(ROOT)}")


def load_dataframes(uri: str, db_name: str) -> dict[str, pd.DataFrame]:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client[db_name]
    docs = list(db["evaluation_reports"].find({}, {"_id": 0}))

    product_rows = []
    profile_rows = []
    consensus_rows = []
    recommendation_rows = []

    for d in docs:
        run_id = d.get("run_id")
        model = d.get("seller_recommendation", {}).get("model")
        product = d.get("product", {})
        overall = d.get("overall_analysis", {})
        promo = d.get("promotion_effectiveness", {})

        product_rows.append(
            {
                "run_id": run_id,
                "llm_model": model,
                "product_id": product.get("id"),
                "title": product.get("title"),
                "brand": product.get("brand"),
                "category": product.get("category"),
                "current_price": product.get("current_price"),
                "original_price": product.get("original_price"),
                "discount_pct": product.get("discount"),
                "rating": product.get("rating"),
                "reviews": product.get("reviews"),
                "mean_score": overall.get("mean_score"),
                "coverage_score": overall.get("coverage_score"),
                "risk_score": overall.get("risk_score"),
                "consensus_level": overall.get("consensus_level"),
                "overall_appeal": promo.get("overall_appeal"),
                "conversion_potential": promo.get("conversion_potential"),
                "improvement_potential": promo.get("improvement_potential"),
            }
        )

        for profile_key, payload in (d.get("profile_results") or {}).items():
            if not isinstance(payload, dict):
                continue
            profile_rows.append(
                {
                    "run_id": run_id,
                    "llm_model": model,
                    "product_id": product.get("id"),
                    "profile": profile_key,
                    "score": payload.get("score"),
                    "purchase_intention": payload.get("purchase_intention"),
                    "status": payload.get("status"),
                    "concerns": payload.get("top_concerns") or [],
                    "strengths": payload.get("top_strengths") or [],
                }
            )

        consensus = d.get("consensus") or {}
        consensus_rows.append(
            {
                "run_id": run_id,
                "llm_model": model,
                "product_id": product.get("id"),
                "strengths": consensus.get("strengths") or [],
                "weaknesses": consensus.get("weaknesses") or [],
                "disagreement_areas": consensus.get("disagreement_areas") or [],
            }
        )

        recommendation_rows.append(
            {
                "run_id": run_id,
                "llm_model": model,
                "product_id": product.get("id"),
                "message": d.get("seller_recommendation", {}).get("message", ""),
                "prompt_tokens": d.get("seller_recommendation", {}).get("prompt_tokens"),
                "completion_tokens": d.get("seller_recommendation", {}).get(
                    "completion_tokens"
                ),
            }
        )

    return {
        "products": pd.DataFrame(product_rows),
        "profiles": pd.DataFrame(profile_rows),
        "consensus": pd.DataFrame(consensus_rows),
        "recommendations": pd.DataFrame(recommendation_rows),
    }


# ---------------------------------------------------------------------------
# Helpers de texto / nuvens de palavras
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-]+")


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in TOKEN_RE.findall(text or "")]


def clean_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in PT_STOPWORDS and len(t) > 2]


def build_wordcloud(text: str, color: str | None = None) -> WordCloud:
    return WordCloud(
        width=1200,
        height=600,
        background_color="white",
        stopwords=PT_STOPWORDS,
        collocations=False,
        colormap="viridis" if color is None else None,
        color_func=(lambda *args, **kwargs: color) if color else None,
        max_words=120,
        min_font_size=10,
        prefer_horizontal=0.9,
    ).generate(text)


def join_phrases(items: list[list[str]]) -> str:
    flat: list[str] = []
    for lst in items:
        for phrase in lst:
            if isinstance(phrase, str) and phrase.strip():
                flat.extend(clean_tokens(tokenize(phrase)))
    return " ".join(flat)


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------
def table_top_terms_por_perfil(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile in PROFILE_ORDER:
        sub = profiles[profiles["profile"] == profile]
        concerns = clean_tokens(
            tokenize(" ".join(p for lst in sub["concerns"] for p in lst))
        )
        strengths = clean_tokens(
            tokenize(" ".join(p for lst in sub["strengths"] for p in lst))
        )
        for term, count in Counter(concerns).most_common(10):
            rows.append(
                {"Perfil": PROFILE_LABELS[profile], "Tipo": "Preocupação", "Termo": term, "Frequência": count}
            )
        for term, count in Counter(strengths).most_common(10):
            rows.append(
                {"Perfil": PROFILE_LABELS[profile], "Tipo": "Ponto forte", "Termo": term, "Frequência": count}
            )
    df = pd.DataFrame(rows)
    save_table(df, "mongo_tabela_top_termos_por_perfil")
    return df


def table_consenso_termos(consensus: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in [
        ("strengths", "Ponto forte (consenso)"),
        ("weaknesses", "Fraqueza (consenso)"),
        ("disagreement_areas", "Discordância"),
    ]:
        tokens = clean_tokens(
            tokenize(" ".join(p for lst in consensus[col] for p in lst))
        )
        for term, count in Counter(tokens).most_common(15):
            rows.append({"Tipo": label, "Termo": term, "Frequência": count})
    df = pd.DataFrame(rows)
    save_table(df, "mongo_tabela_termos_consenso")
    return df


def table_recomendacao_metricas(recs: pd.DataFrame) -> pd.DataFrame:
    df = recs.copy()
    df["caracteres"] = df["message"].fillna("").str.len()
    df["palavras"] = df["message"].fillna("").apply(lambda s: len(tokenize(s)))
    out = (
        df.groupby("llm_model")
        .agg(
            n=("message", "size"),
            caracteres_medios=("caracteres", "mean"),
            palavras_medias=("palavras", "mean"),
            prompt_tokens_medio=("prompt_tokens", "mean"),
            completion_tokens_medio=("completion_tokens", "mean"),
        )
        .round(1)
        .reset_index()
    )
    out.columns = [
        "Modelo",
        "n",
        "Caracteres médios",
        "Palavras médias",
        "Prompt tokens (média)",
        "Completion tokens (média)",
    ]
    save_table(out, "mongo_tabela_recomendacao_metricas")
    return out


# ---------------------------------------------------------------------------
# Nuvens de palavras
# ---------------------------------------------------------------------------
def fig_wordcloud_geral(profiles: pd.DataFrame) -> None:
    concerns_text = join_phrases(profiles["concerns"].tolist())
    strengths_text = join_phrases(profiles["strengths"].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(build_wordcloud(strengths_text, color="#2E7D32"), interpolation="bilinear")
    axes[0].set_title("Pontos fortes mais citados (todos os perfis)")
    axes[0].axis("off")

    axes[1].imshow(build_wordcloud(concerns_text, color="#C62828"), interpolation="bilinear")
    axes[1].set_title("Preocupações mais citadas (todos os perfis)")
    axes[1].axis("off")

    fig.suptitle("Análise textual agregada — preocupações × pontos fortes")
    save_figure(fig, "mongo_fig_wordcloud_geral")


def fig_wordcloud_por_perfil(profiles: pd.DataFrame, kind: str) -> None:
    """kind: 'concerns' ou 'strengths'."""
    title_map = {"concerns": "Preocupações", "strengths": "Pontos fortes"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, profile in zip(axes, PROFILE_ORDER):
        sub = profiles[profiles["profile"] == profile]
        text = join_phrases(sub[kind].tolist())
        if not text.strip():
            ax.text(0.5, 0.5, "(sem dados)", ha="center", va="center")
            ax.axis("off")
            continue
        wc = build_wordcloud(text, color=PROFILE_COLORS[profile])
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(PROFILE_LABELS[profile])
        ax.axis("off")

    axes[-1].axis("off")
    fig.suptitle(f"{title_map[kind]} por perfil de comprador", fontsize=13)
    save_figure(fig, f"mongo_fig_wordcloud_{kind}_por_perfil")


def fig_wordcloud_consenso(consensus: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, title, color in [
        (axes[0], "strengths", "Pontos fortes em consenso", "#2E7D32"),
        (axes[1], "weaknesses", "Fraquezas em consenso", "#C62828"),
        (axes[2], "disagreement_areas", "Áreas de divergência", "#6A1B9A"),
    ]:
        text = join_phrases(consensus[col].tolist())
        if not text.strip():
            ax.text(0.5, 0.5, "(sem dados)", ha="center", va="center")
            ax.axis("off")
            continue
        ax.imshow(build_wordcloud(text, color=color), interpolation="bilinear")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Consenso entre perfis — análise textual")
    save_figure(fig, "mongo_fig_wordcloud_consenso")


def fig_wordcloud_recomendacao(recs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, model in zip(axes, sorted(recs["llm_model"].unique())):
        text = " ".join(recs[recs["llm_model"] == model]["message"].dropna())
        text = " ".join(clean_tokens(tokenize(text)))
        ax.imshow(
            build_wordcloud(text, color=MODEL_COLORS.get(model)),
            interpolation="bilinear",
        )
        ax.set_title(f"Mensagens ao vendedor — {model}")
        ax.axis("off")
    fig.suptitle("Vocabulário das recomendações geradas pelos modelos")
    save_figure(fig, "mongo_fig_wordcloud_recomendacao")


# ---------------------------------------------------------------------------
# Visualizações adicionais
# ---------------------------------------------------------------------------
def fig_radar_perfis(profiles: pd.DataFrame) -> None:
    """Radar chart médio (score, intenção, satisfação) por perfil e modelo."""
    status_to_num = {
        "Muito Insatisfeito": 1,
        "Insatisfeito": 2,
        "Neutro": 3,
        "Satisfeito": 4,
        "Muito Satisfeito": 5,
    }
    df = profiles.copy()
    df["status_num"] = df["status"].map(status_to_num)
    agg = (
        df.groupby(["llm_model", "profile"])
        .agg(
            score=("score", "mean"),
            intention=("purchase_intention", "mean"),
            status=("status_num", "mean"),
        )
        .reset_index()
    )
    # normaliza para 0-1
    agg["score_n"] = agg["score"] / 10
    agg["intention_n"] = agg["intention"] / 100
    agg["status_n"] = (agg["status"] - 1) / 4

    metrics = ["score_n", "intention_n", "status_n"]
    metric_labels = ["Score (0-10)", "Intenção (%)", "Satisfação (1-5)"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5.5), subplot_kw={"projection": "polar"}
    )
    for ax, model in zip(axes, sorted(agg["llm_model"].unique())):
        sub = agg[agg["llm_model"] == model]
        for profile in PROFILE_ORDER:
            row = sub[sub["profile"] == profile]
            if row.empty:
                continue
            values = row[metrics].iloc[0].tolist()
            values += values[:1]
            ax.plot(
                angles,
                values,
                color=PROFILE_COLORS[profile],
                linewidth=2,
                label=PROFILE_LABELS[profile],
            )
            ax.fill(angles, values, color=PROFILE_COLORS[profile], alpha=0.12)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
        ax.set_title(model, pad=18)
    axes[-1].legend(
        loc="upper right", bbox_to_anchor=(1.55, 1.05), frameon=False, fontsize=9
    )
    fig.suptitle("Perfil dos compradores por modelo (métricas normalizadas)")
    save_figure(fig, "mongo_fig_radar_perfis")


def fig_status_por_perfil(profiles: pd.DataFrame) -> None:
    counts = (
        profiles.groupby(["profile", "status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STATUS_ORDER, fill_value=0)
    )
    pct = counts.div(counts.sum(axis=1), axis=0).mul(100)
    pct.index = [PROFILE_LABELS.get(i, i) for i in pct.index]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    bottom = np.zeros(len(pct))
    for status in STATUS_ORDER:
        ax.barh(
            pct.index,
            pct[status],
            left=bottom,
            color=STATUS_COLORS[status],
            label=status,
            edgecolor="white",
        )
        bottom += pct[status].values
    ax.set_xlabel("Distribuição (%)")
    ax.set_ylabel("Perfil")
    ax.set_title("Status de satisfação por perfil de comprador")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    save_figure(fig, "mongo_fig_status_por_perfil")


def fig_violino_score_perfil(profiles: pd.DataFrame) -> None:
    df = profiles.copy()
    df["Perfil"] = df["profile"].map(PROFILE_LABELS)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    sns.violinplot(
        data=df,
        x="Perfil",
        y="score",
        hue="llm_model",
        order=[PROFILE_LABELS[p] for p in PROFILE_ORDER],
        split=True,
        inner="quartile",
        palette=MODEL_COLORS,
        ax=ax,
    )
    ax.axhline(7.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Score atribuído")
    ax.set_title("Distribuição dos scores por perfil — violino comparativo")
    ax.legend(title="Modelo", frameon=False)
    save_figure(fig, "mongo_fig_violino_score_perfil")


def fig_top_termos_barras(profiles: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, kind, title, color in [
        (axes[0], "strengths", "Pontos fortes", "#2E7D32"),
        (axes[1], "concerns", "Preocupações", "#C62828"),
    ]:
        tokens = clean_tokens(
            tokenize(" ".join(p for lst in profiles[kind] for p in lst))
        )
        top = Counter(tokens).most_common(15)
        terms, counts = zip(*top)
        ax.barh(list(reversed(terms)), list(reversed(counts)), color=color, alpha=0.85)
        ax.set_title(f"Top 15 termos — {title}")
        ax.set_xlabel("Frequência")
    fig.suptitle("Termos mais frequentes nas avaliações dos perfis")
    save_figure(fig, "mongo_fig_top_termos_barras")


def fig_correlacao_metricas(products: pd.DataFrame) -> None:
    cols = [
        "discount_pct",
        "rating",
        "mean_score",
        "coverage_score",
        "risk_score",
        "consensus_level",
        "overall_appeal",
        "conversion_potential",
        "improvement_potential",
    ]
    corr = products[cols].corr()
    labels = [
        "Desconto",
        "Rating",
        "Score Médio",
        "Cobertura",
        "Risco",
        "Consenso",
        "Apelo",
        "Conversão",
        "Melhoria",
    ]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Correlação de Pearson"},
        ax=ax,
    )
    ax.set_title("Matriz de correlação entre métricas do produto")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    save_figure(fig, "mongo_fig_correlacao_metricas")


def fig_bubble_desconto_rating(products: pd.DataFrame) -> None:
    df = products.dropna(subset=["discount_pct", "rating", "overall_appeal"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, model in zip(axes, sorted(df["llm_model"].unique())):
        sub = df[df["llm_model"] == model]
        sc = ax.scatter(
            sub["discount_pct"],
            sub["rating"],
            c=sub["overall_appeal"],
            s=sub["conversion_potential"].fillna(20) * 1.5 + 10,
            cmap="viridis",
            alpha=0.7,
            edgecolor="black",
            linewidth=0.3,
            vmin=0,
            vmax=10,
        )
        ax.set_title(f"{model}")
        ax.set_xlabel("Desconto (%)")
        ax.grid(alpha=0.3)
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("Apelo geral (0-10)")
    axes[0].set_ylabel("Rating do produto")
    fig.suptitle("Desconto × rating × apelo (tamanho = potencial de conversão)")
    save_figure(fig, "mongo_fig_bubble_desconto_rating")


def fig_scatter_intencao_score(profiles: pd.DataFrame) -> None:
    df = profiles.copy()
    df["Perfil"] = df["profile"].map(PROFILE_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, model in zip(axes, sorted(df["llm_model"].unique())):
        sub = df[df["llm_model"] == model]
        for profile in PROFILE_ORDER:
            ps = sub[sub["profile"] == profile]
            ax.scatter(
                ps["score"],
                ps["purchase_intention"],
                color=PROFILE_COLORS[profile],
                alpha=0.55,
                s=22,
                edgecolor="black",
                linewidth=0.2,
                label=PROFILE_LABELS[profile],
            )
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Score atribuído")
        ax.set_title(model)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Intenção de compra (%)")
    axes[-1].legend(
        loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, title="Perfil"
    )
    fig.suptitle("Relação entre score atribuído e intenção de compra")
    save_figure(fig, "mongo_fig_scatter_intencao_score")


def fig_recomendacao_tamanho(recs: pd.DataFrame) -> None:
    df = recs.copy()
    df["palavras"] = df["message"].fillna("").apply(lambda s: len(tokenize(s)))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    sns.histplot(
        data=df,
        x="palavras",
        hue="llm_model",
        bins=30,
        palette=MODEL_COLORS,
        kde=True,
        alpha=0.55,
        edgecolor="black",
        linewidth=0.3,
        ax=ax,
    )
    ax.set_xlabel("Número de palavras na recomendação")
    ax.set_ylabel("Frequência")
    ax.set_title("Tamanho das mensagens geradas para o vendedor")
    save_figure(fig, "mongo_fig_recomendacao_tamanho")


def fig_concordancia_perfil(profiles: pd.DataFrame) -> None:
    """Mostra, por produto, em quantos perfis cada modelo emitiu o mesmo status."""
    pivot = profiles.pivot_table(
        index=["product_id", "profile"],
        columns="llm_model",
        values="status",
        aggfunc="first",
    ).dropna()
    if pivot.shape[1] < 2:
        return
    a_col, b_col = pivot.columns[0], pivot.columns[1]
    matrix = pd.crosstab(pivot[a_col], pivot[b_col])
    matrix = matrix.reindex(index=STATUS_ORDER, columns=STATUS_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Avaliações coincidentes"},
        ax=ax,
    )
    ax.set_xlabel(b_col)
    ax.set_ylabel(a_col)
    ax.set_title("Concordância de status entre os modelos (perfil × produto)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    save_figure(fig, "mongo_fig_concordancia_status")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uri",
        default="mongodb://localhost:27017",
        help="URI de conexão com o MongoDB",
    )
    parser.add_argument(
        "--db", default="tcc_evaluations", help="Nome do banco no MongoDB"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    print(f"Lendo {args.uri}/{args.db}.evaluation_reports ...")
    data = load_dataframes(args.uri, args.db)
    print(
        f"  produtos={len(data['products'])} | perfis={len(data['profiles'])} | "
        f"consensos={len(data['consensus'])} | recomendações={len(data['recommendations'])}"
    )

    print("\nGerando tabelas:")
    table_top_termos_por_perfil = table_top_terms_por_perfil(data["profiles"])
    table_consenso_termos(data["consensus"])
    table_recomendacao_metricas(data["recommendations"])

    print("\nGerando nuvens de palavras:")
    fig_wordcloud_geral(data["profiles"])
    fig_wordcloud_por_perfil(data["profiles"], "concerns")
    fig_wordcloud_por_perfil(data["profiles"], "strengths")
    fig_wordcloud_consenso(data["consensus"])
    fig_wordcloud_recomendacao(data["recommendations"])

    print("\nGerando gráficos adicionais:")
    fig_radar_perfis(data["profiles"])
    fig_status_por_perfil(data["profiles"])
    fig_violino_score_perfil(data["profiles"])
    fig_top_termos_barras(data["profiles"])
    fig_correlacao_metricas(data["products"])
    fig_bubble_desconto_rating(data["products"])
    fig_scatter_intencao_score(data["profiles"])
    fig_recomendacao_tamanho(data["recommendations"])
    fig_concordancia_perfil(data["profiles"])

    print("\nConcluído.")


if __name__ == "__main__":
    main()
