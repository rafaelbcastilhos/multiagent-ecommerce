"""
Análise dos resultados das execuções do sistema multiagente.

Lê o banco data/evaluations.db, gera tabelas (CSV) e gráficos (PNG)
para subsidiar o capítulo 5 (Resultados) do TCC.

Uso:
    python scripts/analyze_results.py

Saídas:
    outputs/tables/*.csv
    outputs/charts/*.png
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "evaluations.db"
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
STATUS_COLORS = {
    "Muito Insatisfeito": "#8B0000",
    "Insatisfeito": "#E57373",
    "Neutro": "#BDBDBD",
    "Satisfeito": "#81C784",
    "Muito Satisfeito": "#1B5E20",
}
MODEL_COLORS = {"llama3.2": "#1f77b4", "qwen2.5:7b": "#ff7f0e"}


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


def load_dataframes(db_path: Path) -> dict[str, pd.DataFrame]:
    with sqlite3.connect(db_path) as conn:
        runs = pd.read_sql_query("SELECT * FROM evaluation_runs", conn)
        products = pd.read_sql_query("SELECT * FROM products", conn)
        product_evals = pd.read_sql_query(
            """
            SELECT pe.*, er.llm_model, er.llm_temperature
            FROM product_evaluations pe
            JOIN evaluation_runs er ON er.run_id = pe.run_id
            """,
            conn,
        )
        profile_evals = pd.read_sql_query(
            """
            SELECT pfe.*, pe.product_id, pe.run_id, er.llm_model
            FROM profile_evaluations pfe
            JOIN product_evaluations pe ON pe.id = pfe.product_evaluation_id
            JOIN evaluation_runs er ON er.run_id = pe.run_id
            """,
            conn,
        )
        consensus = pd.read_sql_query(
            """
            SELECT ci.*, pe.product_id, er.llm_model
            FROM consensus_items ci
            JOIN product_evaluations pe ON pe.id = ci.product_evaluation_id
            JOIN evaluation_runs er ON er.run_id = pe.run_id
            """,
            conn,
        )
    return {
        "runs": runs,
        "products": products,
        "product_evals": product_evals,
        "profile_evals": profile_evals,
        "consensus": consensus,
    }


def save_table(df: pd.DataFrame, name: str) -> None:
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=False, float_format="%.3f")
    print(f"  tabela: {path.relative_to(ROOT)}")


def save_figure(fig: plt.Figure, name: str) -> None:
    path = CHARTS_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  figura: {path.relative_to(ROOT)}")


def table_resumo_execucoes(runs: pd.DataFrame) -> pd.DataFrame:
    df = runs[["llm_model", "llm_temperature", "product_count", "created_at"]].copy()
    df.columns = ["Modelo", "Temperatura", "Produtos", "Data de execução"]
    save_table(df, "tabela_resumo_execucoes")
    return df


def table_metricas_agregadas(product_evals: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "mean_score",
        "coverage_score",
        "risk_score",
        "consensus_level",
        "overall_appeal",
        "conversion_potential",
        "improvement_potential",
    ]
    grouped = product_evals.groupby("llm_model")[cols].mean().round(3)
    grouped.index.name = "Modelo"
    grouped.columns = [
        "Score Médio",
        "Cobertura (%)",
        "Risco (%)",
        "Consenso",
        "Apelo Geral",
        "Potencial Conversão",
        "Potencial Melhoria",
    ]
    out = grouped.reset_index()
    save_table(out, "tabela_metricas_agregadas")
    return out


def table_score_por_perfil(profile_evals: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        profile_evals.groupby(["llm_model", "profile"])
        .agg(
            n=("score", "size"),
            score_medio=("score", "mean"),
            intencao_compra=("purchase_intention", "mean"),
        )
        .round(3)
        .reset_index()
    )
    grouped["profile"] = grouped["profile"].map(PROFILE_LABELS).fillna(grouped["profile"])
    grouped.columns = ["Modelo", "Perfil", "n", "Score Médio", "Intenção de Compra"]
    save_table(grouped, "tabela_score_por_perfil")
    return grouped


def table_status_distribuicao(profile_evals: pd.DataFrame) -> pd.DataFrame:
    counts = (
        profile_evals.groupby(["llm_model", "status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STATUS_ORDER, fill_value=0)
    )
    pct = counts.div(counts.sum(axis=1), axis=0).mul(100).round(2)
    pct.columns = [f"{c} (%)" for c in pct.columns]
    out = pd.concat([counts, pct], axis=1).reset_index()
    save_table(out, "tabela_status_distribuicao")
    return out


def table_concordancia_modelos(product_evals: pd.DataFrame) -> pd.DataFrame:
    pivot = product_evals.pivot_table(
        index="product_id",
        columns="llm_model",
        values=[
            "mean_score",
            "overall_appeal",
            "conversion_potential",
            "consensus_level",
            "coverage_score",
        ],
    ).dropna()

    rows = []
    for metric in pivot.columns.levels[0]:
        a = pivot[(metric, "llama3.2")]
        b = pivot[(metric, "qwen2.5:7b")]
        rows.append(
            {
                "Métrica": metric,
                "Diferença Média Absoluta": float((a - b).abs().mean()),
                "Diferença Mediana": float((a - b).median()),
                "Correlação Pearson": float(a.corr(b)),
            }
        )
    df = pd.DataFrame(rows).round(3)
    save_table(df, "tabela_concordancia_modelos")
    return df


def fig_distribuicao_status(profile_evals: pd.DataFrame) -> None:
    counts = (
        profile_evals.groupby(["llm_model", "status"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STATUS_ORDER, fill_value=0)
    )
    pct = counts.div(counts.sum(axis=1), axis=0).mul(100)

    fig, ax = plt.subplots(figsize=(8, 4.2))
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

    ax.set_xlabel("Distribuição de status (%)")
    ax.set_ylabel("Modelo")
    ax.set_title("Distribuição de status de satisfação por modelo")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)
    save_figure(fig, "fig_distribuicao_status")


def fig_score_por_perfil(profile_evals: pd.DataFrame) -> None:
    pivot = (
        profile_evals.groupby(["profile", "llm_model"])["score"]
        .mean()
        .unstack("llm_model")
        .reindex(PROFILE_ORDER)
    )
    pivot.index = [PROFILE_LABELS[p] for p in pivot.index]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(pivot))
    width = 0.38
    for i, model in enumerate(pivot.columns):
        ax.bar(
            x + (i - 0.5) * width,
            pivot[model],
            width=width,
            label=model,
            color=MODEL_COLORS.get(model, None),
            edgecolor="black",
            linewidth=0.4,
        )
    ax.axhline(7.0, color="gray", linestyle="--", linewidth=0.8, label="Limiar de satisfação (7,0)")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Score médio")
    ax.set_xlabel("Perfil de comprador")
    ax.set_title("Score médio atribuído por perfil em cada modelo")
    ax.set_ylim(0, 10)
    ax.legend(frameon=False)
    save_figure(fig, "fig_score_por_perfil")


def fig_intencao_compra_perfil(profile_evals: pd.DataFrame) -> None:
    pivot = (
        profile_evals.groupby(["profile", "llm_model"])["purchase_intention"]
        .mean()
        .unstack("llm_model")
        .reindex(PROFILE_ORDER)
    )
    pivot.index = [PROFILE_LABELS[p] for p in pivot.index]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(pivot))
    width = 0.38
    for i, model in enumerate(pivot.columns):
        ax.bar(
            x + (i - 0.5) * width,
            pivot[model],
            width=width,
            label=model,
            color=MODEL_COLORS.get(model, None),
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Intenção de compra média (%)")
    ax.set_xlabel("Perfil de comprador")
    ax.set_title("Intenção de compra média por perfil e modelo")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    save_figure(fig, "fig_intencao_compra_perfil")


def fig_boxplot_scores_perfil(profile_evals: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model in zip(axes, sorted(profile_evals["llm_model"].unique())):
        subset = profile_evals[profile_evals["llm_model"] == model]
        data = [
            subset[subset["profile"] == p]["score"].dropna().values
            for p in PROFILE_ORDER
        ]
        bp = ax.boxplot(
            data,
            tick_labels=[PROFILE_LABELS[p] for p in PROFILE_ORDER],
            patch_artist=True,
            widths=0.6,
            medianprops={"color": "black"},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(MODEL_COLORS.get(model, "#cccccc"))
            patch.set_alpha(0.7)
        ax.set_title(model)
        ax.set_ylim(0, 10)
        ax.axhline(7.0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Score atribuído")
    fig.suptitle("Distribuição de scores por perfil em cada modelo")
    save_figure(fig, "fig_boxplot_scores_perfil")


def fig_correlacao_llama_qwen(product_evals: pd.DataFrame) -> None:
    pivot = product_evals.pivot_table(
        index="product_id", columns="llm_model", values="mean_score"
    ).dropna()
    if "llama3.2" not in pivot.columns or "qwen2.5:7b" not in pivot.columns:
        return

    a, b = pivot["llama3.2"], pivot["qwen2.5:7b"]
    corr = a.corr(b)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(a, b, alpha=0.45, s=22, edgecolor="black", linewidth=0.3)
    lims = [min(a.min(), b.min()) - 0.3, max(a.max(), b.max()) + 0.3]
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=0.8, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Score médio — Llama 3.2")
    ax.set_ylabel("Score médio — Qwen 2.5 7B")
    ax.set_title(f"Concordância entre modelos por produto (r = {corr:.3f})")
    ax.legend(frameon=False)
    save_figure(fig, "fig_correlacao_llama_qwen")


def fig_metricas_agregadas(product_evals: pd.DataFrame) -> None:
    metrics = {
        "mean_score": ("Score Médio", 10),
        "overall_appeal": ("Apelo Geral", 10),
        "coverage_score": ("Cobertura (%)", 100),
        "risk_score": ("Risco (%)", 100),
        "consensus_level": ("Consenso (%)", 100),
        "conversion_potential": ("Conversão (%)", 100),
    }
    grouped = product_evals.groupby("llm_model")[list(metrics.keys())].mean()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(metrics))
    width = 0.38
    for i, model in enumerate(grouped.index):
        normalized = [
            grouped.loc[model, m] / metrics[m][1] * 100 for m in metrics
        ]
        ax.bar(
            x + (i - 0.5) * width,
            normalized,
            width=width,
            label=model,
            color=MODEL_COLORS.get(model, None),
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics.values()], rotation=15, ha="right")
    ax.set_ylabel("Valor normalizado (% do máximo)")
    ax.set_title("Comparação das métricas agregadas entre os modelos")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    save_figure(fig, "fig_metricas_agregadas")


def fig_distribuicao_consensus(product_evals: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for model in sorted(product_evals["llm_model"].unique()):
        values = product_evals[product_evals["llm_model"] == model][
            "consensus_level"
        ].dropna()
        ax.hist(
            values,
            bins=20,
            alpha=0.55,
            label=model,
            color=MODEL_COLORS.get(model, None),
            edgecolor="black",
            linewidth=0.3,
        )
    ax.set_xlabel("Nível de consenso (%)")
    ax.set_ylabel("Frequência (produtos)")
    ax.set_title("Distribuição do nível de consenso entre perfis")
    ax.legend(frameon=False)
    save_figure(fig, "fig_distribuicao_consensus")


def fig_appeal_vs_desconto(
    product_evals: pd.DataFrame, products: pd.DataFrame
) -> None:
    df = product_evals.merge(
        products[["product_id", "discount_pct"]], on="product_id", how="left"
    )
    df = df.dropna(subset=["discount_pct", "overall_appeal"])

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for model in sorted(df["llm_model"].unique()):
        subset = df[df["llm_model"] == model]
        ax.scatter(
            subset["discount_pct"],
            subset["overall_appeal"],
            alpha=0.45,
            s=22,
            label=model,
            color=MODEL_COLORS.get(model, None),
            edgecolor="black",
            linewidth=0.2,
        )
        if len(subset) > 5:
            coef = np.polyfit(subset["discount_pct"], subset["overall_appeal"], 1)
            xs = np.linspace(subset["discount_pct"].min(), subset["discount_pct"].max(), 50)
            ax.plot(
                xs,
                np.polyval(coef, xs),
                color=MODEL_COLORS.get(model, "black"),
                linewidth=1.4,
            )
    ax.set_xlabel("Desconto (%)")
    ax.set_ylabel("Apelo geral (0-10)")
    ax.set_title("Relação entre desconto e apelo geral")
    ax.set_ylim(0, 10)
    ax.legend(frameon=False)
    save_figure(fig, "fig_appeal_vs_desconto")


def fig_heatmap_perfil_metrica(profile_evals: pd.DataFrame) -> None:
    matrix = (
        profile_evals.groupby(["llm_model", "profile"])
        .agg(score=("score", "mean"), intention=("purchase_intention", "mean"))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, metric, title, vmax in zip(
        axes,
        ["score", "intention"],
        ["Score médio (0-10)", "Intenção de compra (%)"],
        [10, 100],
    ):
        pivot = matrix.pivot(index="llm_model", columns="profile", values=metric)[
            PROFILE_ORDER
        ]
        pivot.columns = [PROFILE_LABELS[p] for p in pivot.columns]
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(title)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{pivot.values[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.suptitle("Mapa de calor: perfil × modelo")
    save_figure(fig, "fig_heatmap_perfil_metrica")


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"Banco não encontrado: {DB_PATH}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    print(f"Lendo {DB_PATH.relative_to(ROOT)}...")
    data = load_dataframes(DB_PATH)
    print(
        f"  runs={len(data['runs'])} | produtos={len(data['products'])} | "
        f"product_evals={len(data['product_evals'])} | profile_evals={len(data['profile_evals'])}"
    )

    print("\nGerando tabelas:")
    table_resumo_execucoes(data["runs"])
    table_metricas_agregadas(data["product_evals"])
    table_score_por_perfil(data["profile_evals"])
    table_status_distribuicao(data["profile_evals"])
    table_concordancia_modelos(data["product_evals"])

    print("\nGerando gráficos:")
    fig_distribuicao_status(data["profile_evals"])
    fig_score_por_perfil(data["profile_evals"])
    fig_intencao_compra_perfil(data["profile_evals"])
    fig_boxplot_scores_perfil(data["profile_evals"])
    fig_correlacao_llama_qwen(data["product_evals"])
    fig_metricas_agregadas(data["product_evals"])
    fig_distribuicao_consensus(data["product_evals"])
    fig_appeal_vs_desconto(data["product_evals"], data["products"])
    fig_heatmap_perfil_metrica(data["profile_evals"])

    print("\nConcluído.")


if __name__ == "__main__":
    main()
