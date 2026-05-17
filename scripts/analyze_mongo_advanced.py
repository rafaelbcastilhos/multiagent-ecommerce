"""
Análises avançadas para enriquecer a seção de Resultados da monografia.

Lê do MongoDB (`tcc_evaluations.evaluation_reports`) e produz:

1. Inferência estatística (Wilcoxon pareado + Mann-Whitney + Cohen's d / Cliff's delta)
2. Concordância entre modelos (Cohen's kappa quadrático para status, ICC para scores)
3. Performance segmentada por faixas de desconto, rating e preço
4. Validação contra rating real dos compradores (ground truth)
5. Inventário de falhas das mensagens (vazamentos, formatação, tamanho) + bigramas

Uso:
    python scripts/analyze_mongo_advanced.py
    python scripts/analyze_mongo_advanced.py --uri mongodb://localhost:27017 --db tcc_evaluations

Saídas:
    outputs/charts/mongo_adv_*.png
    outputs/tables/mongo_adv_*.csv
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
from pymongo import MongoClient
from scipy import stats
from sklearn.metrics import cohen_kappa_score

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
STATUS_ORDER = [
    "Muito Insatisfeito",
    "Insatisfeito",
    "Neutro",
    "Satisfeito",
    "Muito Satisfeito",
]
STATUS_NUM = {s: i for i, s in enumerate(STATUS_ORDER)}
MODEL_COLORS = {"llama3.2": "#1f77b4", "qwen2.5:7b": "#ff7f0e"}
MODELS = ["llama3.2", "qwen2.5:7b"]

PT_STOPWORDS = {
    "a", "à", "às", "ao", "aos", "as", "o", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos", "em", "na", "no", "nas", "nos", "por",
    "pelo", "pela", "pelos", "pelas", "para", "com", "sem", "sob", "sobre",
    "entre", "ate", "até", "e", "ou", "mas", "que", "se", "como", "quando",
    "porque", "qual", "quais", "quem", "onde", "ja", "já", "nao", "não",
    "sim", "muito", "muita", "muitos", "muitas", "ser", "é", "foi", "são",
    "era", "eram", "estar", "está", "estão", "ter", "tem", "têm", "esse",
    "essa", "esses", "essas", "este", "esta", "estes", "estas", "isso",
    "isto", "seu", "sua", "seus", "suas", "ele", "ela", "eles", "elas",
    "também", "tambem", "ainda", "mais", "menos", "bem", "tudo", "todo",
    "toda", "todos", "todas", "outro", "outra", "outros", "outras",
}

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-]+")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
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
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"  tabela: {path.relative_to(ROOT)}")


def save_figure(fig: plt.Figure, name: str) -> None:
    path = CHARTS_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  figura: {path.relative_to(ROOT)}")


def load_dataframes(uri: str, db_name: str) -> dict[str, pd.DataFrame]:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    docs = list(client[db_name]["evaluation_reports"].find({}, {"_id": 0}))

    products, profiles, recs = [], [], []
    for d in docs:
        product = d.get("product", {}) or {}
        overall = d.get("overall_analysis", {}) or {}
        promo = d.get("promotion_effectiveness", {}) or {}
        rec = d.get("seller_recommendation", {}) or {}
        model = rec.get("model")

        products.append(
            {
                "run_id": d.get("run_id"),
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
            if isinstance(payload, dict):
                profiles.append(
                    {
                        "llm_model": model,
                        "product_id": product.get("id"),
                        "profile": profile_key,
                        "score": payload.get("score"),
                        "purchase_intention": payload.get("purchase_intention"),
                        "status": payload.get("status"),
                    }
                )

        recs.append(
            {
                "llm_model": model,
                "product_id": product.get("id"),
                "message": rec.get("message", ""),
                "prompt_tokens": rec.get("prompt_tokens"),
                "completion_tokens": rec.get("completion_tokens"),
            }
        )

    return {
        "products": pd.DataFrame(products),
        "profiles": pd.DataFrame(profiles),
        "recommendations": pd.DataFrame(recs),
    }


# ---------------------------------------------------------------------------
# 1. INFERÊNCIA ESTATÍSTICA
# ---------------------------------------------------------------------------
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d para amostras pareadas."""
    diff = a - b
    return float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta — effect size não paramétrico (-1 a +1)."""
    n_a, n_b = len(a), len(b)
    greater = sum((x > y) for x in a for y in b)
    less = sum((x < y) for x in a for y in b)
    return (greater - less) / (n_a * n_b)


def interpret_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "desprezível"
    if ad < 0.5:
        return "pequeno"
    if ad < 0.8:
        return "médio"
    return "grande"


def interpret_delta(delta: float) -> str:
    ad = abs(delta)
    if ad < 0.147:
        return "desprezível"
    if ad < 0.33:
        return "pequeno"
    if ad < 0.474:
        return "médio"
    return "grande"


def table_inferencia_modelos(products: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon pareado + Mann-Whitney + Cohen's d + Cliff's delta entre modelos."""
    metrics = {
        "mean_score": "Score Médio",
        "overall_appeal": "Apelo Geral",
        "conversion_potential": "Potencial de Conversão",
        "risk_score": "Risco",
        "consensus_level": "Nível de Consenso",
        "coverage_score": "Cobertura",
        "improvement_potential": "Potencial de Melhoria",
    }
    pivot = products.pivot_table(
        index="product_id", columns="llm_model", values=list(metrics.keys())
    ).dropna()

    rows = []
    for col, label in metrics.items():
        a = pivot[(col, MODELS[0])].values
        b = pivot[(col, MODELS[1])].values

        # Wilcoxon pareado: ignora pares com diferença zero
        diffs = a - b
        if np.any(diffs != 0):
            w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox")
        else:
            w_stat, w_p = np.nan, 1.0

        # Mann-Whitney (sanity check, considerando independente)
        mw_stat, mw_p = stats.mannwhitneyu(a, b, alternative="two-sided")

        d = cohens_d(a, b)
        delta = cliffs_delta(a, b)

        rows.append(
            {
                "Métrica": label,
                f"Média {MODELS[0]}": round(float(a.mean()), 3),
                f"Média {MODELS[1]}": round(float(b.mean()), 3),
                "Δ (llama − qwen)": round(float(a.mean() - b.mean()), 3),
                "Wilcoxon p-valor": round(float(w_p), 5),
                "Mann-Whitney p-valor": round(float(mw_p), 5),
                "Cohen's d": round(d, 3),
                "Cliff's δ": round(delta, 3),
                "Interp. d": interpret_d(d),
                "Interp. δ": interpret_delta(delta),
                "Significativo (α=0,05)": "sim" if w_p < 0.05 else "não",
            }
        )
    df = pd.DataFrame(rows)
    save_table(df, "mongo_adv_inferencia_modelos")
    return df


def fig_effect_sizes(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(df))
    colors = ["#2E7D32" if d > 0 else "#C62828" for d in df["Cliff's δ"]]
    bars = ax.barh(y, df["Cliff's δ"], color=colors, alpha=0.75, edgecolor="black", linewidth=0.4)
    for i, (bar, p) in enumerate(zip(bars, df["Wilcoxon p-valor"])):
        marker = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        x_pos = bar.get_width()
        ax.text(
            x_pos + (0.015 if x_pos >= 0 else -0.015),
            i,
            marker,
            va="center",
            ha="left" if x_pos >= 0 else "right",
            fontsize=10,
        )
    ax.axvline(0, color="black", linewidth=0.6)
    for thresh in [0.147, 0.33, 0.474]:
        ax.axvline(thresh, color="gray", linestyle=":", linewidth=0.5)
        ax.axvline(-thresh, color="gray", linestyle=":", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["Métrica"])
    ax.set_xlabel("Cliff's δ  (positivo → llama3.2 maior)")
    ax.set_title("Tamanho do efeito entre os modelos com significância (Wilcoxon)")
    ax.set_xlim(-1, 1)
    ax.text(
        0.99, -0.18,
        "* p<0,05  ** p<0,01  *** p<0,001  |  linhas: limiares de Cliff's δ (peq./méd./grande)",
        ha="right", va="top", transform=ax.transAxes, fontsize=8, color="gray",
    )
    save_figure(fig, "mongo_adv_fig_effect_sizes")


# ---------------------------------------------------------------------------
# 2. CONCORDÂNCIA ENTRE MODELOS
# ---------------------------------------------------------------------------
def icc_2_1(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1) — confiabilidade absoluta entre dois avaliadores."""
    ratings = np.column_stack([a, b])
    n, k = ratings.shape
    mean_per_target = ratings.mean(axis=1)
    mean_per_rater = ratings.mean(axis=0)
    grand_mean = ratings.mean()

    ms_between = k * np.sum((mean_per_target - grand_mean) ** 2) / (n - 1)
    ms_rater = n * np.sum((mean_per_rater - grand_mean) ** 2) / (k - 1)
    ss_total = np.sum((ratings - grand_mean) ** 2)
    ss_error = ss_total - (n - 1) * ms_between - (k - 1) * ms_rater
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom = ms_between + (k - 1) * ms_error + k * (ms_rater - ms_error) / n
    return float((ms_between - ms_error) / denom) if denom > 0 else float("nan")


def interpret_kappa(k: float) -> str:
    if k < 0:
        return "discordância"
    if k < 0.20:
        return "leve"
    if k < 0.40:
        return "fraca"
    if k < 0.60:
        return "moderada"
    if k < 0.80:
        return "substancial"
    return "quase perfeita"


def table_concordancia(profiles: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Cohen's kappa quadrático no status + ICC nos scores, por perfil e geral."""
    rows = []

    # Por perfil — kappa quadrático sobre status
    for profile in PROFILE_ORDER:
        sub = profiles[profiles["profile"] == profile]
        pivot = sub.pivot_table(
            index="product_id",
            columns="llm_model",
            values="status",
            aggfunc="first",
        ).dropna()
        if pivot.shape[0] < 5 or pivot.shape[1] < 2:
            continue
        a = pivot[MODELS[0]].map(STATUS_NUM).values
        b = pivot[MODELS[1]].map(STATUS_NUM).values
        kappa_q = cohen_kappa_score(a, b, weights="quadratic")
        kappa_l = cohen_kappa_score(a, b, weights="linear")
        agreement = float((a == b).mean())

        score_pivot = sub.pivot_table(
            index="product_id", columns="llm_model", values="score"
        ).dropna()
        icc_v = icc_2_1(
            score_pivot[MODELS[0]].values, score_pivot[MODELS[1]].values
        )
        rows.append(
            {
                "Nível": PROFILE_LABELS[profile],
                "n produtos": int(pivot.shape[0]),
                "Concordância exata (%)": round(agreement * 100, 2),
                "Kappa quadrático": round(kappa_q, 3),
                "Kappa linear": round(kappa_l, 3),
                "ICC(2,1) score": round(icc_v, 3),
                "Interpretação κ": interpret_kappa(kappa_q),
            }
        )

    # Geral — todos os perfis empilhados
    pivot_all = profiles.pivot_table(
        index=["product_id", "profile"],
        columns="llm_model",
        values="status",
        aggfunc="first",
    ).dropna()
    a = pivot_all[MODELS[0]].map(STATUS_NUM).values
    b = pivot_all[MODELS[1]].map(STATUS_NUM).values
    score_pivot = profiles.pivot_table(
        index=["product_id", "profile"], columns="llm_model", values="score"
    ).dropna()
    rows.append(
        {
            "Nível": "GERAL (todos os perfis)",
            "n produtos": int(pivot_all.shape[0]),
            "Concordância exata (%)": round(float((a == b).mean()) * 100, 2),
            "Kappa quadrático": round(cohen_kappa_score(a, b, weights="quadratic"), 3),
            "Kappa linear": round(cohen_kappa_score(a, b, weights="linear"), 3),
            "ICC(2,1) score": round(
                icc_2_1(score_pivot[MODELS[0]].values, score_pivot[MODELS[1]].values), 3
            ),
            "Interpretação κ": interpret_kappa(
                cohen_kappa_score(a, b, weights="quadratic")
            ),
        }
    )

    df = pd.DataFrame(rows)
    save_table(df, "mongo_adv_concordancia_modelos")
    return df


def fig_concordancia(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    df_plot = df.copy()
    y = np.arange(len(df_plot))
    width = 0.4
    ax.barh(
        y - width / 2, df_plot["Kappa quadrático"], height=width,
        color="#1f77b4", alpha=0.8, label="Cohen's κ (quadrático)", edgecolor="black", linewidth=0.3,
    )
    ax.barh(
        y + width / 2, df_plot["ICC(2,1) score"], height=width,
        color="#ff7f0e", alpha=0.8, label="ICC(2,1) — score", edgecolor="black", linewidth=0.3,
    )
    for thresh, label in [(0.4, "fraca"), (0.6, "moderada"), (0.8, "substancial")]:
        ax.axvline(thresh, color="gray", linestyle=":", linewidth=0.5)
        ax.text(thresh, len(df_plot) - 0.3, label, fontsize=7, color="gray", rotation=90, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["Nível"])
    ax.set_xlim(-0.1, 1)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Coeficiente de concordância")
    ax.set_title("Concordância entre llama3.2 e qwen2.5:7b por perfil")
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, "mongo_adv_fig_concordancia")


# ---------------------------------------------------------------------------
# 3. PERFORMANCE SEGMENTADA
# ---------------------------------------------------------------------------
def add_tier_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["faixa_desconto"] = pd.cut(
        df["discount_pct"],
        bins=[-0.1, 20, 40, 60, 100],
        labels=["0-20%", "20-40%", "40-60%", ">60%"],
    )
    df["faixa_rating"] = pd.cut(
        df["rating"],
        bins=[-0.1, 0.01, 3.0, 4.0, 5.01],
        labels=["sem rating", "baixo (<3)", "médio (3-4)", "alto (>4)"],
    )
    df["faixa_preco"] = pd.qcut(
        df["current_price"],
        q=4,
        labels=["barato", "médio-baixo", "médio-alto", "caro"],
    )
    df["faixa_reviews"] = pd.cut(
        df["reviews"].fillna(0),
        bins=[-0.1, 0.01, 10, 100, 1e9],
        labels=["sem reviews", "1-10", "11-100", "+100"],
    )
    return df


def table_performance_segmentada(products: pd.DataFrame) -> pd.DataFrame:
    df = add_tier_columns(products)
    rows = []
    for tier_col, tier_label in [
        ("faixa_desconto", "Desconto"),
        ("faixa_rating", "Rating"),
        ("faixa_preco", "Preço"),
        ("faixa_reviews", "Reviews"),
    ]:
        agg = (
            df.groupby([tier_col, "llm_model"], observed=True)
            .agg(
                n=("product_id", "count"),
                score_medio=("mean_score", "mean"),
                apelo_medio=("overall_appeal", "mean"),
                conversao_media=("conversion_potential", "mean"),
                risco_medio=("risk_score", "mean"),
            )
            .round(2)
            .reset_index()
        )
        agg["Segmento"] = tier_label
        agg = agg.rename(columns={tier_col: "Faixa"})
        rows.append(agg[["Segmento", "Faixa", "llm_model", "n", "score_medio",
                          "apelo_medio", "conversao_media", "risco_medio"]])
    out = pd.concat(rows, ignore_index=True)
    out.columns = ["Segmento", "Faixa", "Modelo", "n", "Score Médio",
                   "Apelo Médio", "Conversão Média", "Risco Médio"]
    save_table(out, "mongo_adv_performance_segmentada")
    return out


def fig_performance_segmentada(products: pd.DataFrame) -> None:
    df = add_tier_columns(products)
    segments = [
        ("faixa_desconto", "Faixa de Desconto", ["0-20%", "20-40%", "40-60%", ">60%"]),
        ("faixa_rating", "Faixa de Rating", ["sem rating", "baixo (<3)", "médio (3-4)", "alto (>4)"]),
        ("faixa_preco", "Faixa de Preço", ["barato", "médio-baixo", "médio-alto", "caro"]),
        ("faixa_reviews", "Faixa de Reviews", ["sem reviews", "1-10", "11-100", "+100"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()
    for ax, (col, title, order) in zip(axes, segments):
        agg = (
            df.groupby([col, "llm_model"], observed=True)["mean_score"]
            .mean()
            .unstack("llm_model")
            .reindex(order)
        )
        x = np.arange(len(agg))
        width = 0.38
        for i, model in enumerate(MODELS):
            if model not in agg.columns:
                continue
            ax.bar(
                x + (i - 0.5) * width, agg[model],
                width=width, label=model, color=MODEL_COLORS[model],
                edgecolor="black", linewidth=0.4,
            )
            for j, val in enumerate(agg[model]):
                if not np.isnan(val):
                    ax.text(x[j] + (i - 0.5) * width, val + 0.1, f"{val:.1f}",
                            ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(agg.index, rotation=15, ha="right")
        ax.set_ylim(0, 10)
        ax.axhline(7.0, color="gray", linestyle="--", linewidth=0.7)
        ax.set_ylabel("Score médio")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Score médio por segmento — análise de viés do sistema", fontsize=13)
    save_figure(fig, "mongo_adv_fig_performance_segmentada")


def fig_heatmap_segmento_metrica(products: pd.DataFrame) -> None:
    df = add_tier_columns(products)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics = {
        "mean_score": ("Score Médio (0-10)", 10),
        "conversion_potential": ("Conversão (%)", 100),
    }
    for ax, model in zip(axes, MODELS):
        sub = df[df["llm_model"] == model]
        matrix = (
            sub.groupby(["faixa_rating", "faixa_desconto"], observed=True)["mean_score"]
            .mean()
            .unstack("faixa_desconto")
            .reindex(
                index=["sem rating", "baixo (<3)", "médio (3-4)", "alto (>4)"],
                columns=["0-20%", "20-40%", "40-60%", ">60%"],
            )
        )
        sns.heatmap(
            matrix, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=0, vmax=10, ax=ax, cbar_kws={"label": "Score médio"},
        )
        ax.set_title(f"{model}")
        ax.set_xlabel("Faixa de Desconto")
        ax.set_ylabel("Faixa de Rating")
    fig.suptitle("Score médio por (rating × desconto) — interação de fatores")
    save_figure(fig, "mongo_adv_fig_heatmap_rating_desconto")


# ---------------------------------------------------------------------------
# 4. VALIDAÇÃO COM GROUND TRUTH (rating real)
# ---------------------------------------------------------------------------
def table_validacao_ground_truth(products: pd.DataFrame) -> pd.DataFrame:
    df = products[products["rating"] > 0].copy()  # ignora produtos sem rating real
    rows = []
    for model in MODELS:
        sub = df[df["llm_model"] == model]
        if len(sub) < 5:
            continue
        # Normaliza score do sistema (0-10) para 0-5 para comparar com rating real (0-5)
        sub_score_norm = sub["mean_score"] / 2
        pearson_r, pearson_p = stats.pearsonr(sub_score_norm, sub["rating"])
        spearman_r, spearman_p = stats.spearmanr(sub_score_norm, sub["rating"])
        mae = float(np.abs(sub_score_norm - sub["rating"]).mean())
        rmse = float(np.sqrt(((sub_score_norm - sub["rating"]) ** 2).mean()))
        bias = float((sub_score_norm - sub["rating"]).mean())
        rows.append(
            {
                "Modelo": model,
                "n produtos com rating": len(sub),
                "Pearson r": round(pearson_r, 3),
                "Pearson p": round(pearson_p, 5),
                "Spearman ρ": round(spearman_r, 3),
                "Spearman p": round(spearman_p, 5),
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
                "Viés (sistema − humano)": round(bias, 3),
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, "mongo_adv_validacao_ground_truth")
    return out


def fig_ground_truth(products: pd.DataFrame) -> None:
    df = products[products["rating"] > 0].copy()
    df["score_norm"] = df["mean_score"] / 2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, model in zip(axes, MODELS):
        sub = df[df["llm_model"] == model]
        ax.scatter(
            sub["rating"], sub["score_norm"],
            alpha=0.5, s=28, color=MODEL_COLORS[model],
            edgecolor="black", linewidth=0.3,
        )
        # linha y=x
        ax.plot([0, 5], [0, 5], "--", color="gray", linewidth=0.8, label="y = x (concordância perfeita)")
        # regressão linear
        if len(sub) > 2:
            coef = np.polyfit(sub["rating"], sub["score_norm"], 1)
            xs = np.linspace(0, 5, 50)
            ax.plot(xs, np.polyval(coef, xs), color=MODEL_COLORS[model], linewidth=1.6,
                    label=f"y = {coef[0]:.2f}x + {coef[1]:.2f}")
        r, p = stats.pearsonr(sub["rating"], sub["score_norm"])
        ax.set_title(f"{model}  (r = {r:.3f}, n = {len(sub)})")
        ax.set_xlabel("Rating real dos compradores (0-5)")
        ax.set_ylabel("Score do sistema normalizado (0-5)")
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    fig.suptitle("Validação contra ground truth — score do sistema vs rating humano")
    save_figure(fig, "mongo_adv_fig_ground_truth")


def fig_residuos_ground_truth(products: pd.DataFrame) -> None:
    """Distribuição dos resíduos (sistema - humano) por modelo."""
    df = products[products["rating"] > 0].copy()
    df["residuo"] = (df["mean_score"] / 2) - df["rating"]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    sns.violinplot(
        data=df, x="llm_model", y="residuo", hue="llm_model",
        palette=MODEL_COLORS, inner="box", ax=ax, legend=False,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Resíduo (sistema − humano)  [escala 0-5]")
    ax.set_xlabel("Modelo")
    ax.set_title("Distribuição dos resíduos — viés sistemático na avaliação")
    save_figure(fig, "mongo_adv_fig_residuos_ground_truth")


# ---------------------------------------------------------------------------
# 5. INVENTÁRIO DE FALHAS + BIGRAMAS
# ---------------------------------------------------------------------------
LEAK_TOKENS = ["top_concern", "top_strength", "purchase_intention",
               "mean_score", "{", "}", "<", ">", "todo", "tbd"]


def detect_failures(message: str) -> dict[str, bool | int]:
    msg = message or ""
    msg_lower = msg.lower()
    words = TOKEN_RE.findall(msg)
    n_words = len(words)

    # Item duplicado — ex.: dois "3." na lista
    item_numbers = re.findall(r"^\s*(\d+)[.\)]", msg, flags=re.MULTILINE)
    duplicated_items = len(item_numbers) != len(set(item_numbers))

    return {
        "leak_token": any(tok in msg_lower for tok in LEAK_TOKENS),
        "duplicated_item": duplicated_items,
        "very_short": n_words < 50,
        "very_long": n_words > 300,
        "vazio": n_words == 0,
        "markdown_quebrado": "**" in msg and msg.count("**") % 2 != 0,
        "n_palavras": n_words,
        "n_caracteres": len(msg),
    }


def table_inventario_falhas(recs: pd.DataFrame) -> pd.DataFrame:
    df = recs.copy()
    failures = pd.DataFrame([detect_failures(m) for m in df["message"]])
    df = pd.concat([df.reset_index(drop=True), failures], axis=1)

    rows = []
    for model in MODELS:
        sub = df[df["llm_model"] == model]
        n = len(sub)
        rows.append(
            {
                "Modelo": model,
                "n total": n,
                "Vazamento de tokens (%)": round(sub["leak_token"].mean() * 100, 2),
                "Itens duplicados (%)": round(sub["duplicated_item"].mean() * 100, 2),
                "Mensagens curtas <50p (%)": round(sub["very_short"].mean() * 100, 2),
                "Mensagens longas >300p (%)": round(sub["very_long"].mean() * 100, 2),
                "Markdown malformado (%)": round(sub["markdown_quebrado"].mean() * 100, 2),
                "Vazias (%)": round(sub["vazio"].mean() * 100, 2),
                "Palavras (mediana)": int(sub["n_palavras"].median()),
                "Palavras (média)": round(float(sub["n_palavras"].mean()), 1),
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, "mongo_adv_inventario_falhas")
    return out


def fig_inventario_falhas(recs: pd.DataFrame) -> None:
    df = recs.copy()
    failures = pd.DataFrame([detect_failures(m) for m in df["message"]])
    df = pd.concat([df.reset_index(drop=True), failures], axis=1)

    failure_types = [
        ("leak_token", "Vazamento\nde tokens"),
        ("duplicated_item", "Itens\nduplicados"),
        ("very_short", "Mensagens\ncurtas (<50p)"),
        ("very_long", "Mensagens\nlongas (>300p)"),
        ("markdown_quebrado", "Markdown\nmalformado"),
    ]
    rates = pd.DataFrame(
        {
            model: [
                df[df["llm_model"] == model][col].mean() * 100
                for col, _ in failure_types
            ]
            for model in MODELS
        },
        index=[lbl for _, lbl in failure_types],
    )

    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(rates))
    width = 0.38
    for i, model in enumerate(MODELS):
        bars = ax.bar(
            x + (i - 0.5) * width, rates[model],
            width=width, label=model, color=MODEL_COLORS[model],
            edgecolor="black", linewidth=0.4,
        )
        for bar, val in zip(bars, rates[model]):
            if val > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.4,
                        f"{val:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(rates.index)
    ax.set_ylabel("Frequência (%)")
    ax.set_title("Inventário de falhas nas mensagens geradas")
    ax.legend(frameon=False)
    save_figure(fig, "mongo_adv_fig_inventario_falhas")


def tokenize_clean(text: str) -> list[str]:
    return [
        w.lower() for w in TOKEN_RE.findall(text or "")
        if w.lower() not in PT_STOPWORDS and len(w) > 2
    ]


def extract_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def table_bigramas(recs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        msgs = recs[recs["llm_model"] == model]["message"].dropna().tolist()
        all_tokens = [tokenize_clean(m) for m in msgs]
        bigrams = Counter()
        trigrams = Counter()
        for toks in all_tokens:
            bigrams.update(extract_ngrams(toks, 2))
            trigrams.update(extract_ngrams(toks, 3))
        for ng, count in bigrams.most_common(15):
            rows.append({"Modelo": model, "Tipo": "Bigrama", "Termo": " ".join(ng), "Frequência": count})
        for ng, count in trigrams.most_common(10):
            rows.append({"Modelo": model, "Tipo": "Trigrama", "Termo": " ".join(ng), "Frequência": count})
    df = pd.DataFrame(rows)
    save_table(df, "mongo_adv_bigramas_recomendacao")
    return df


def fig_bigramas(recs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, model in zip(axes, MODELS):
        msgs = recs[recs["llm_model"] == model]["message"].dropna().tolist()
        bigrams = Counter()
        for m in msgs:
            bigrams.update(extract_ngrams(tokenize_clean(m), 2))
        top = bigrams.most_common(15)
        terms = [" ".join(ng) for ng, _ in top]
        counts = [c for _, c in top]
        ax.barh(list(reversed(terms)), list(reversed(counts)),
                color=MODEL_COLORS[model], alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.set_xlabel("Frequência")
        ax.set_title(f"Top 15 bigramas — {model}")
    fig.suptitle("Padrões linguísticos recorrentes nas recomendações", fontsize=13)
    save_figure(fig, "mongo_adv_fig_bigramas")


def table_diversidade_lexical(recs: pd.DataFrame) -> pd.DataFrame:
    """Type-Token Ratio (TTR) e variantes para medir repetição."""
    rows = []
    for model in MODELS:
        msgs = recs[recs["llm_model"] == model]["message"].dropna().tolist()
        # TTR por mensagem (média)
        ttrs = []
        for m in msgs:
            toks = tokenize_clean(m)
            if toks:
                ttrs.append(len(set(toks)) / len(toks))
        # TTR global (corpus)
        all_toks = [t for m in msgs for t in tokenize_clean(m)]
        global_ttr = len(set(all_toks)) / len(all_toks) if all_toks else 0
        rows.append(
            {
                "Modelo": model,
                "n mensagens": len(msgs),
                "Tokens totais": len(all_toks),
                "Tokens únicos": len(set(all_toks)),
                "TTR global": round(global_ttr, 3),
                "TTR médio por msg": round(float(np.mean(ttrs)), 3),
                "Comprimento médio (tokens)": round(float(np.mean([len(tokenize_clean(m)) for m in msgs])), 1),
            }
        )
    df = pd.DataFrame(rows)
    save_table(df, "mongo_adv_diversidade_lexical")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="mongodb://localhost:27017")
    parser.add_argument("--db", default="tcc_evaluations")
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
        f"recomendações={len(data['recommendations'])}"
    )

    print("\n[1/5] Inferência estatística entre modelos:")
    df_inf = table_inferencia_modelos(data["products"])
    fig_effect_sizes(df_inf)

    print("\n[2/5] Concordância entre modelos:")
    df_conc = table_concordancia(data["profiles"], data["products"])
    fig_concordancia(df_conc)

    print("\n[3/5] Performance segmentada:")
    table_performance_segmentada(data["products"])
    fig_performance_segmentada(data["products"])
    fig_heatmap_segmento_metrica(data["products"])

    print("\n[4/5] Validação contra ground truth:")
    table_validacao_ground_truth(data["products"])
    fig_ground_truth(data["products"])
    fig_residuos_ground_truth(data["products"])

    print("\n[5/5] Inventário de falhas e análise textual:")
    table_inventario_falhas(data["recommendations"])
    fig_inventario_falhas(data["recommendations"])
    table_bigramas(data["recommendations"])
    fig_bigramas(data["recommendations"])
    table_diversidade_lexical(data["recommendations"])

    print("\nConcluído.")


if __name__ == "__main__":
    main()
