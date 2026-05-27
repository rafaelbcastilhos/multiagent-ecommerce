"""
Análise de correlação dos Top-N itens mais bem avaliados por perfil e modelo.

Compara os rankings entre llama3.2 e qwen2.5:7b para identificar:
  - Correlação de scores (Pearson + Spearman) em todos os produtos
  - Interseção dos top-N por perfil (com índice de Jaccard)
  - Produtos com maior consenso cross-model (aparições em múltiplas combinações)
  - Produtos universais: top-N em todos os perfis dentro de cada modelo
  - Produtos com consenso total (top-N em todos os perfis e ambos os modelos)

Uso:
    python scripts/analyze_top_n_correlation.py          # padrão: TOP_N=30
    python scripts/analyze_top_n_correlation.py --top 20
    python scripts/analyze_top_n_correlation.py --top 50

Saídas:
    outputs/tables/top{N}_ranking_por_perfil_modelo.csv
    outputs/tables/top{N}_correlacao_entre_modelos.csv
    outputs/tables/top{N}_intersecao_por_perfil.csv
    outputs/tables/top{N}_consenso_cross_model.csv
    outputs/charts/top{N}_fig_correlacao_scores.png
    outputs/charts/top{N}_fig_jaccard_por_perfil.png
    outputs/charts/top{N}_fig_heatmap_presenca.png
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT       = Path(__file__).resolve().parent.parent
DB_PATH    = ROOT / "data" / "evaluations.db"
TABLES_DIR = ROOT / "outputs" / "tables"
CHARTS_DIR = ROOT / "outputs" / "charts"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

PERFIS  = ["anxious", "demanding", "economic", "impulsive", "rational"]
MODELOS = ["llama3.2", "qwen2.5:7b"]

PROFILE_LABELS = {
    "anxious":   "Ansioso",
    "demanding": "Exigente",
    "economic":  "Econômico",
    "impulsive": "Impulsivo",
    "rational":  "Racional",
}


def load_data(db_path: Path) -> pd.DataFrame:
    """Carrega todos os scores por produto × perfil × modelo do SQLite."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT
            er.llm_model              AS modelo,
            pe2.profile               AS perfil,
            p.product_id,
            p.title,
            p.brand,
            p.category,
            p.current_price,
            p.discount_pct,
            p.rating                  AS rating_dataset,
            pe2.score                 AS score,
            pe2.purchase_intention,
            pe2.status,
            pe.mean_score,
            pe.consensus_level
        FROM profile_evaluations  pe2
        JOIN product_evaluations  pe  ON pe2.product_evaluation_id = pe.id
        JOIN evaluation_runs      er  ON pe.run_id = er.run_id
        JOIN products             p   ON pe.product_id = p.product_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def add_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna 'rank' dentro de cada (modelo, perfil) por score desc."""
    df = df.copy()
    df["rank"] = (
        df.groupby(["modelo", "perfil"])["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return df


def top_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df[df["rank"] <= n].copy()


def correlacao_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Correlação Pearson + Spearman entre modelos para cada perfil (todos os produtos)."""
    pivot = df.pivot_table(
        index="product_id",
        columns=["modelo", "perfil"],
        values="score",
        aggfunc="mean",
    )
    pivot.columns = [f"{m}|{p}" for m, p in pivot.columns]

    rows = []
    for perfil in PERFIS:
        col_l = f"llama3.2|{perfil}"
        col_q = f"qwen2.5:7b|{perfil}"
        sub = pivot[[col_l, col_q]].dropna()
        pr, pp = pearsonr(sub[col_l], sub[col_q])
        sr, sp = spearmanr(sub[col_l], sub[col_q])
        rows.append(
            dict(
                perfil=perfil,
                perfil_pt=PROFILE_LABELS[perfil],
                n=len(sub),
                pearson_r=round(pr, 4),
                pearson_p=round(pp, 6),
                spearman_r=round(sr, 4),
                spearman_p=round(sp, 6),
                significancia="***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else "ns",
            )
        )
    return pd.DataFrame(rows)


def intersecao_top_n(df_topn: pd.DataFrame) -> pd.DataFrame:
    """Jaccard e cobertura da interseção dos top-N entre modelos por perfil."""
    rows = []
    for perfil in PERFIS:
        A = set(df_topn[(df_topn["modelo"] == "llama3.2")   & (df_topn["perfil"] == perfil)]["product_id"])
        B = set(df_topn[(df_topn["modelo"] == "qwen2.5:7b") & (df_topn["perfil"] == perfil)]["product_id"])
        inter = A & B
        union = A | B
        jaccard = len(inter) / len(union) if union else 0.0
        rows.append(
            dict(
                perfil=perfil,
                perfil_pt=PROFILE_LABELS[perfil],
                n_llama=len(A),
                n_qwen=len(B),
                intersecao=len(inter),
                uniao=len(union),
                jaccard=round(jaccard, 4),
                cobertura_llama_pct=round(len(inter) / len(A) * 100, 1) if A else 0,
                cobertura_qwen_pct=round(len(inter) / len(B) * 100, 1) if B else 0,
                produtos_em_comum="; ".join(sorted(inter)),
            )
        )
    return pd.DataFrame(rows)


def consenso_cross_model(df_topn: pd.DataFrame) -> pd.DataFrame:
    """Conta em quantas combinações (modelo × perfil) cada produto aparece no top-N."""
    total_combos = len(MODELOS) * len(PERFIS)  # 10

    combos = (
        df_topn.groupby("product_id")
        .apply(lambda g: g[["modelo", "perfil"]].drop_duplicates().shape[0])
        .reset_index(name="n_combos")
    )
    titles = df_topn[["product_id", "title"]].drop_duplicates("product_id")
    combos = combos.merge(titles, on="product_id").sort_values("n_combos", ascending=False)
    combos["pct_combos"] = (combos["n_combos"] / total_combos * 100).round(1)
    combos["total_possiveis"] = total_combos

    # detalha presença por modelo×perfil
    detail_rows = []
    for _, row in combos.iterrows():
        pid = row["product_id"]
        sub = df_topn[df_topn["product_id"] == pid]
        for modelo in MODELOS:
            for perfil in PERFIS:
                entry = sub[(sub["modelo"] == modelo) & (sub["perfil"] == perfil)]
                detail_rows.append(
                    dict(
                        product_id=pid,
                        title=row["title"],
                        n_combos=row["n_combos"],
                        modelo=modelo,
                        perfil=perfil,
                        presente=int(not entry.empty),
                        score=entry["score"].values[0] if not entry.empty else None,
                        rank=entry["rank"].values[0] if not entry.empty else None,
                        purchase_intention=entry["purchase_intention"].values[0] if not entry.empty else None,
                    )
                )
    return pd.DataFrame(detail_rows)


def universais_por_modelo(df_topn: pd.DataFrame) -> pd.DataFrame:
    """Produtos que estão no top-N de TODOS os perfis dentro de cada modelo."""
    rows = []
    for modelo in MODELOS:
        sub = df_topn[df_topn["modelo"] == modelo]
        counts = sub.groupby("product_id")["perfil"].nunique()
        universais = counts[counts == len(PERFIS)].index.tolist()
        for pid in universais:
            title = sub[sub["product_id"] == pid]["title"].iloc[0]
            scores = sub[sub["product_id"] == pid].set_index("perfil")["score"].to_dict()
            ranks  = sub[sub["product_id"] == pid].set_index("perfil")["rank"].to_dict()
            rows.append(
                dict(
                    modelo=modelo,
                    product_id=pid,
                    title=title,
                    score_anxious=scores.get("anxious"),
                    score_demanding=scores.get("demanding"),
                    score_economic=scores.get("economic"),
                    score_impulsive=scores.get("impulsive"),
                    score_rational=scores.get("rational"),
                    rank_anxious=ranks.get("anxious"),
                    rank_demanding=ranks.get("demanding"),
                    rank_economic=ranks.get("economic"),
                    rank_impulsive=ranks.get("impulsive"),
                    rank_rational=ranks.get("rational"),
                )
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def plot_correlacao(df_corr: pd.DataFrame, n: int, out_dir: Path) -> None:
    """Gráfico de barras horizontais — Pearson r por perfil."""
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [PROFILE_LABELS[p] for p in df_corr["perfil"]]
    colors = ["#2ecc71" if r >= 0.7 else "#f39c12" if r >= 0.5 else "#e74c3c"
              for r in df_corr["pearson_r"]]
    bars = ax.barh(labels, df_corr["pearson_r"], color=colors, edgecolor="white", height=0.55)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
    ax.axvline(0.7, ls="--", color="gray", lw=1, label="r = 0.70 (alta)")
    ax.axvline(0.5, ls=":",  color="silver", lw=1, label="r = 0.50 (moderada)")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Pearson r", fontsize=11)
    ax.set_title(f"Correlação de Scores entre Modelos por Perfil (top-{n} | todos os produtos)",
                 fontsize=12, pad=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / f"top{n}_fig_correlacao_scores.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✔ {path.relative_to(ROOT)}")


def plot_jaccard(df_inter: pd.DataFrame, n: int, out_dir: Path) -> None:
    """Gráfico de barras — Jaccard e cobertura por perfil."""
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(PERFIS))
    w = 0.28
    labels = [PROFILE_LABELS[p] for p in PERFIS]

    b1 = ax.bar(x - w, df_inter["jaccard"],               w, label="Jaccard",         color="#3498db")
    b2 = ax.bar(x,     df_inter["cobertura_llama_pct"]/100, w, label="Cobertura llama3.2",  color="#e67e22")
    b3 = ax.bar(x + w, df_inter["cobertura_qwen_pct"]/100,  w, label="Cobertura qwen2.5:7b", color="#9b59b6")

    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(f"Interseção Top-{n}: Jaccard e Cobertura por Perfil", fontsize=12, pad=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / f"top{n}_fig_jaccard_por_perfil.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✔ {path.relative_to(ROOT)}")


def plot_heatmap_presenca(df_cross: pd.DataFrame, n: int, out_dir: Path) -> None:
    """Heatmap: produtos com ≥ 6 combinações × (modelo|perfil)."""
    # filtra produtos relevantes
    top_pids = (
        df_cross[df_cross["presente"] == 1]
        .groupby("product_id")["presente"].sum()
        .sort_values(ascending=False)
        .head(25)
        .index.tolist()
    )
    if not top_pids:
        return

    sub = df_cross[df_cross["product_id"].isin(top_pids)].copy()
    sub["col"] = sub["modelo"].str.replace("qwen2.5:7b", "qwen") + "|" + sub["perfil"].str[:3].str.upper()
    col_order = [f"{m.replace('qwen2.5:7b','qwen')}|{p[:3].upper()}" for m in MODELOS for p in PERFIS]

    heat = sub.pivot_table(index="title", columns="col", values="presente", fill_value=0)
    heat = heat.reindex(columns=[c for c in col_order if c in heat.columns])
    # ordena por total de presença
    heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).index]
    # trunca títulos
    heat.index = [t[:50] + "…" if len(t) > 50 else t for t in heat.index]

    fig, ax = plt.subplots(figsize=(12, max(5, len(heat) * 0.38)))
    im = ax.imshow(heat.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=8)
    ax.set_title(f"Presença no Top-{n} — Produtos com maior consenso cross-model", fontsize=12, pad=10)
    plt.colorbar(im, ax=ax, shrink=0.5, label="Presente (1) / Ausente (0)")
    fig.tight_layout()
    path = out_dir / f"top{n}_fig_heatmap_presenca.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ {path.relative_to(ROOT)}")


def print_header(title: str) -> None:
    print(f"\n{'═'*72}\n  {title}\n{'═'*72}")


def print_top_n_table(df_topn: pd.DataFrame, n: int) -> None:
    print_header(f"TOP {n} ITENS POR PERFIL E MODELO")
    for modelo in MODELOS:
        print(f"\n{'━'*72}\n  MODELO: {modelo.upper()}\n{'━'*72}")
        for perfil in PERFIS:
            sub = df_topn[(df_topn["modelo"] == modelo) & (df_topn["perfil"] == perfil)] \
                    .sort_values("rank")
            label = PROFILE_LABELS[perfil]
            stats = f"max={sub['score'].max():.2f}  min={sub['score'].min():.2f}  mean={sub['score'].mean():.2f}"
            print(f"\n  Perfil: {label:<12} | {stats}")
            print(f"  {'Rank':<5} {'Score':>6} {'Intenção':>9} {'Status':<24} Produto")
            print(f"  {'─'*5} {'─'*6} {'─'*9} {'─'*24} {'─'*42}")
            for _, row in sub.iterrows():
                title = row["title"][:48] + "…" if len(row["title"]) > 48 else row["title"]
                intent = f"{row['purchase_intention']:.0f}%" if pd.notna(row["purchase_intention"]) else "—"
                print(f"  {int(row['rank']):<5} {row['score']:>6.2f} {intent:>9} "
                      f"{str(row['status']):<24} {title}")


def print_correlacao(df_corr: pd.DataFrame) -> None:
    print_header("CORRELAÇÃO DE SCORES ENTRE MODELOS (todos os produtos)")
    print(f"\n  {'Perfil':<12} {'Pearson r':>10} {'p-val':>10} {'Spearman ρ':>12} {'p-val':>10}  Sig.")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*12} {'─'*10}  {'─'*4}")
    for _, row in df_corr.iterrows():
        print(f"  {row['perfil']:<12} {row['pearson_r']:>10.4f} {row['pearson_p']:>10.6f} "
              f"{row['spearman_r']:>12.4f} {row['spearman_p']:>10.6f}  {row['significancia']}")


def print_intersecao(df_inter: pd.DataFrame, n: int) -> None:
    print_header(f"INTERSEÇÃO DO TOP-{n} ENTRE MODELOS POR PERFIL")
    print(f"\n  {'Perfil':<12} {'N comum':>8} {'|A∪B|':>7} {'Jaccard':>9} "
          f"{'Cob. llama':>11} {'Cob. qwen':>11}")
    print(f"  {'─'*12} {'─'*8} {'─'*7} {'─'*9} {'─'*11} {'─'*11}")
    for _, row in df_inter.iterrows():
        print(f"  {row['perfil']:<12} {row['intersecao']:>8} {row['uniao']:>7} "
              f"{row['jaccard']:>9.3f} {row['cobertura_llama_pct']:>10.1f}% "
              f"{row['cobertura_qwen_pct']:>10.1f}%")

    # detalhe dos produtos em comum por perfil
    print()
    for _, row in df_inter.iterrows():
        if not row["produtos_em_comum"]:
            continue
        print(f"\n  ► {PROFILE_LABELS[row['perfil']]} — {row['intersecao']} produto(s) em comum:")
        # recupera títulos da df_inter (apenas IDs) — reimprime do df_topn via produtos_em_comum
        for pid in row["produtos_em_comum"].split("; "):
            print(f"      • {pid}")


def print_consenso(df_cross: pd.DataFrame, n: int) -> None:
    print_header(f"PRODUTOS COM MAIOR CONSENSO CROSS-MODEL (top-{n})")
    summary = (
        df_cross[df_cross["presente"] == 1]
        .groupby(["product_id", "title", "n_combos"])
        .size()
        .reset_index(name="_")
        .drop(columns="_")
        .drop_duplicates()
        .sort_values("n_combos", ascending=False)
    )
    total = len(MODELOS) * len(PERFIS)
    print(f"\n  {'#Combos':>8}  {'%':>5}  Produto")
    print(f"  {'─'*8}  {'─'*5}  {'─'*55}")
    for _, row in summary[summary["n_combos"] >= 5].iterrows():
        pct = row["n_combos"] / total * 100
        short = row["title"][:60] + "…" if len(row["title"]) > 60 else row["title"]
        print(f"  {int(row['n_combos']):>8}  {pct:>4.0f}%  {short}")

    # detalhe dos produtos com ≥ 8 combinações
    top_pids = summary[summary["n_combos"] >= 8]["product_id"].tolist()
    if top_pids:
        print(f"\n\n  DETALHE: produtos em ≥ 8 combinações")
        print(f"  {'─'*70}")
        for pid in top_pids:
            title = summary[summary["product_id"] == pid]["title"].iloc[0]
            print(f"\n  ► {title}")
            sub = df_cross[(df_cross["product_id"] == pid) & (df_cross["presente"] == 1)] \
                    .sort_values(["modelo", "perfil"])
            for _, row in sub.iterrows():
                print(f"      {row['modelo']:<14} {row['perfil']:<12} "
                      f"score={row['score']:.1f}  rank={int(row['rank']):<4} "
                      f"intent={row['purchase_intention']:.0f}%")


def print_universais(df_univ: pd.DataFrame, n: int) -> None:
    print_header(f"PRODUTOS NO TOP-{n} DE TODOS OS 5 PERFIS — POR MODELO")
    if df_univ.empty:
        print("\n  Nenhum produto universal encontrado.")
        return
    for modelo in MODELOS:
        sub = df_univ[df_univ["modelo"] == modelo]
        print(f"\n  {modelo}  ({len(sub)} produto(s)):")
        for _, row in sub.iterrows():
            print(f"\n    • {row['title']}")
            for perfil in PERFIS:
                s = row[f"score_{perfil}"]
                r = row[f"rank_{perfil}"]
                label = PROFILE_LABELS[perfil]
                print(f"        {label:<12} score={s:.1f}  rank={int(r)}")


def main(n: int) -> None:
    print(f"\n🔍  Analisando top-{n} itens por perfil × modelo ...")
    print(f"    DB: {DB_PATH}")

    # 1. Carrega dados
    df_raw = load_data(DB_PATH)
    df_all = add_rank(df_raw)
    df_topn = top_n(df_all, n)

    # 2. Análises
    df_corr  = correlacao_scores(df_all)
    df_inter = intersecao_top_n(df_topn)
    df_cross = consenso_cross_model(df_topn)
    df_univ  = universais_por_modelo(df_topn)

    # 3. Impressão no terminal
    print_top_n_table(df_topn, n)
    print_correlacao(df_corr)
    print_intersecao(df_inter, n)
    print_consenso(df_cross, n)
    print_universais(df_univ, n)

    # 4. Salva CSVs
    print_header("EXPORTANDO ARQUIVOS")
    csv_ranking = TABLES_DIR / f"top{n}_ranking_por_perfil_modelo.csv"
    csv_corr    = TABLES_DIR / f"top{n}_correlacao_entre_modelos.csv"
    csv_inter   = TABLES_DIR / f"top{n}_intersecao_por_perfil.csv"
    csv_cross   = TABLES_DIR / f"top{n}_consenso_cross_model.csv"

    df_topn.to_csv(csv_ranking, index=False)
    df_corr.to_csv(csv_corr,    index=False)
    df_inter.drop(columns="produtos_em_comum").to_csv(csv_inter, index=False)
    df_cross.to_csv(csv_cross,  index=False)

    print(f"  ✔ {csv_ranking.relative_to(ROOT)}")
    print(f"  ✔ {csv_corr.relative_to(ROOT)}")
    print(f"  ✔ {csv_inter.relative_to(ROOT)}")
    print(f"  ✔ {csv_cross.relative_to(ROOT)}")

    # 5. Gráficos
    plot_correlacao(df_corr, n, CHARTS_DIR)
    plot_jaccard(df_inter, n, CHARTS_DIR)
    plot_heatmap_presenca(df_cross, n, CHARTS_DIR)

    print(f"\n✅  Análise top-{n} concluída.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analisa correlação dos top-N itens entre modelos LLM."
    )
    parser.add_argument(
        "--top", type=int, default=30,
        metavar="N",
        help="Número de itens top a considerar por perfil/modelo (padrão: 30)",
    )
    args = parser.parse_args()
    main(args.top)
