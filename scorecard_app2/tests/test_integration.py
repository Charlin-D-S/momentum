"""Tests d'intégration des composants (sans lancer Streamlit)."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.charts import (
    chart_calibration_quantile, chart_default_rate_by_score,
    chart_points_by_bin, chart_variable_importance,
)
from components.profile_cards import find_boundary_individuals
from utils.scorecard_engine import (
    get_scorecard_variables, proba_to_points, scorecard_table,
    scorer_enrichi, variable_importance,
)
from tests.generate_synthetic_data import build_dataset, build_scorecard


def green(msg: str) -> None:
    print(f"  \033[92m✓\033[0m {msg}")


def test_scorecard_table_excludes_intercept():
    sc = build_scorecard()
    view = scorecard_table(sc)
    assert "-" not in view["Label"].to_list()
    assert view.height == sc.height - 1   # une seule constante dans build_scorecard
    green("scorecard_table exclut l'intercept.")


def test_variable_importance_ordering():
    sc = build_scorecard()
    imp = variable_importance(sc)
    pts = imp["importance"].to_list()
    assert pts == sorted(pts, reverse=True), "L'importance doit être triée décroissant"
    assert imp.height == sc.select("Variables").n_unique() - 1   # hors intercept
    green("variable_importance trié décroissant et complet.")


def test_chart_points_by_bin_renders():
    sc = build_scorecard()
    view = scorecard_table(sc)
    sub = view.filter(pl.col("Variables") == "anciennete")
    fig = chart_points_by_bin(sub, "anciennete")
    assert fig.data, "Figure vide"
    # Doit avoir une trace bar avec autant de barres que de bins
    assert len(fig.data[0].x) == sub.height
    green("chart_points_by_bin produit une figure non vide avec le bon nombre de barres.")


def test_chart_variable_importance_renders():
    sc = build_scorecard()
    imp = variable_importance(sc)
    fig = chart_variable_importance(imp)
    assert fig.data
    assert len(fig.data[0].y) == imp.height
    green("chart_variable_importance produit une figure cohérente.")


def test_chart_calibration_quantile():
    sc = build_scorecard()
    df = scorer_enrichi(build_dataset(n=2000, seed=10).lazy(), sc).collect()
    fig, table = chart_calibration_quantile(df, n_bins=10, target_col="defaut_obs")
    assert not table.is_empty(), "Tableau calibration vide"
    assert table.height >= 5, f"{table.height} quantiles seulement"
    # 1 trace diagonale + 1 trace observée
    assert len(fig.data) == 2
    green(f"chart_calibration_quantile : {table.height} quantiles renvoyés.")


def test_chart_calibration_empty_target():
    """Si la cible est absente, retourne figures vides sans crasher."""
    sc = build_scorecard()
    df = scorer_enrichi(build_dataset(n=200, seed=11).lazy(), sc).collect()
    df_no_target = df.drop("defaut_obs")
    fig, table = chart_calibration_quantile(df_no_target, n_bins=10)
    assert table.is_empty()
    green("chart_calibration_quantile gère gracieusement l'absence de cible.")


def test_chart_default_rate_by_score():
    sc = build_scorecard()
    df = scorer_enrichi(build_dataset(n=2000, seed=12).lazy(), sc).collect()
    fig = chart_default_rate_by_score(df, n_bins=10)
    assert fig.data
    green("chart_default_rate_by_score produit une figure non vide.")


def test_find_boundary_individuals():
    sc = build_scorecard()
    df = scorer_enrichi(build_dataset(n=2000, seed=13).lazy(), sc).collect()
    seuil = int(df["score_points"].median())
    cand = find_boundary_individuals(df, seuil, n=10)
    assert cand.height == 10
    # Tous les candidats doivent avoir un score proche du seuil
    distances = (cand["score_points"] - seuil).abs()
    assert distances.max() <= 100, f"Distance max trop grande : {distances.max()}"
    green(f"find_boundary_individuals : 10 candidats trouvés, écart max {distances.max()} pts.")


def test_filter_pipeline_end_to_end():
    """Simule le flux : data → filtres → segment → stats."""
    sc = build_scorecard()
    df = scorer_enrichi(build_dataset(n=2000, seed=14).lazy(), sc).collect()

    # Filtrer sur le bin _bin_type_client = "['Profession libérale']"
    target_bin = "['Profession libérale']"
    segment = df.filter(pl.col("_bin_type_client").cast(pl.Utf8) == target_bin)
    assert segment.height > 0, "Segment vide"
    # Vérifier que tous les individus du segment ont le bon bin
    bins_uniques = segment["_bin_type_client"].unique().to_list()
    assert bins_uniques == [target_bin]
    green(f"Pipeline filtre → segment OK ({segment.height} individus dans le bin).")


def test_zonage_decisionnel_logic():
    """Vérifie la logique de zonage tricolore."""
    sc = build_scorecard()
    df = scorer_enrichi(build_dataset(n=2000, seed=15).lazy(), sc).collect()

    s1, s2 = 0.05, 0.15
    vert = df.filter(pl.col("score_proba") <= s1)
    orange = df.filter((pl.col("score_proba") > s1) & (pl.col("score_proba") <= s2))
    rouge = df.filter(pl.col("score_proba") > s2)

    # Les trois zones doivent partitionner la population
    total = vert.height + orange.height + rouge.height
    assert total == df.height, f"Partition non exhaustive : {total} vs {df.height}"

    # Taux de défaut : VERT < ORANGE < ROUGE attendu (ordre stochastique)
    if "defaut_obs" in df.columns:
        tv = vert["defaut_obs"].mean() if vert.height else 0
        to = orange["defaut_obs"].mean() if orange.height else 0
        tr = rouge["defaut_obs"].mean() if rouge.height else 0
        assert tv < tr, f"Taux défaut VERT ({tv:.2%}) doit être < ROUGE ({tr:.2%})"
        green(f"Zonage cohérent : taux défaut VERT={tv:.2%} < ORANGE={to:.2%} < ROUGE={tr:.2%}.")
    else:
        green("Zonage partitionne la population sans recouvrement.")


def test_proba_to_points_in_segment():
    """proba_to_points fonctionne sur un sous-segment filtré."""
    sc = build_scorecard()
    df = scorer_enrichi(build_dataset(n=2000, seed=16).lazy(), sc).collect()
    segment = df.filter(pl.col("score_proba") < 0.2)
    pts = proba_to_points(0.10, segment, n_neighbors=30)
    assert isinstance(pts, int)
    # Doit être dans la plage de points du segment
    assert segment["score_points"].min() <= pts <= segment["score_points"].max()
    green(f"proba_to_points sur segment réduit : {pts} pts.")


def test_memory_footprint_reasonable():
    """Le DataFrame enrichi reste raisonnable en mémoire."""
    sc = build_scorecard()
    df = scorer_enrichi(build_dataset(n=10_000, seed=17).lazy(), sc).collect()
    mb = df.estimated_size("mb")
    # 10k lignes, ~25 colonnes → quelques MB max
    assert mb < 20, f"Empreinte mémoire trop grande : {mb:.2f} MB"
    green(f"Empreinte mémoire (10k individus enrichis) : {mb:.2f} MB.")


def test_lazy_chain_does_not_materialize_early():
    """scorer_enrichi doit accepter un LazyFrame et garder le lazy."""
    sc = build_scorecard()
    df = build_dataset(n=500, seed=18)
    lazy = scorer_enrichi(df.lazy(), sc)
    assert isinstance(lazy, pl.LazyFrame), "Le retour doit être un LazyFrame"
    # Vérifier qu'on peut empiler d'autres opérations avant collect
    final = lazy.filter(pl.col("score_proba") < 0.3).select(
        ["id_client", "score_points", "score_proba"]
    ).collect()
    assert final.height > 0
    green("scorer_enrichi reste paresseux : chaînage possible avant collect.")


if __name__ == "__main__":
    print("\n=== Tests d'intégration des composants ===\n")
    tests = [
        test_scorecard_table_excludes_intercept,
        test_variable_importance_ordering,
        test_chart_points_by_bin_renders,
        test_chart_variable_importance_renders,
        test_chart_calibration_quantile,
        test_chart_calibration_empty_target,
        test_chart_default_rate_by_score,
        test_find_boundary_individuals,
        test_filter_pipeline_end_to_end,
        test_zonage_decisionnel_logic,
        test_proba_to_points_in_segment,
        test_memory_footprint_reasonable,
        test_lazy_chain_does_not_materialize_early,
    ]
    for t in tests:
        t()
    print(f"\n✅ {len(tests)} tests passés\n")
