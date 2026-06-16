"""Tests unitaires du moteur scorecard."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Permet d'importer depuis utils/ et tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.scorecard_engine import (
    parser_scorecard, scorer_enrichi, get_scorecard_variables,
    proba_to_points, decompose_individual,
)
from tests.generate_synthetic_data import build_scorecard, build_dataset


def green(msg: str) -> None:
    print(f"  \033[92m✓\033[0m {msg}")


def fail(msg: str) -> None:
    print(f"  \033[91m✗ {msg}\033[0m")
    raise AssertionError(msg)


def test_parser_returns_label_for_each_rule():
    sc = build_scorecard()
    règles = parser_scorecard(sc)
    for r in règles:
        if r["type"] != "constante":
            assert "label" in r, f"Label absent dans la règle : {r}"
    green("Le parser conserve le label dans chaque règle.")


def test_parser_handles_three_types():
    sc = build_scorecard()
    règles = parser_scorecard(sc)
    types = {r["type"] for r in règles}
    assert "constante" in types
    assert "numerique" in types
    assert "categorielle" in types
    green("Le parser reconnaît les 3 types (constante / numérique / catégorielle).")


def test_parser_inclut_missing_numerique():
    sc = build_scorecard()
    règles = parser_scorecard(sc)
    anc_missing_rule = [
        r for r in règles if r["variable"] == "anciennete" and r.get("inclut_missing")
    ]
    assert len(anc_missing_rule) == 1, "Une seule règle anciennete doit inclure MISSING"
    assert anc_missing_rule[0]["borne_sup"] == math.inf
    green("Détection du flag MISSING dans les bins numériques.")


def test_parser_inclut_missing_categorielle():
    sc = build_scorecard()
    règles = parser_scorecard(sc)
    type_missing = [
        r for r in règles if r["variable"] == "type_client" and r.get("inclut_missing")
    ]
    assert len(type_missing) == 1
    assert "Auto-entrepreneur" in type_missing[0]["valeurs"]
    assert "MISSING" not in type_missing[0]["valeurs"]
    green("Détection de MISSING dans les bins catégoriels.")


def test_scorer_columns_present():
    sc = build_scorecard()
    df = build_dataset(n=200, seed=1)
    enriched = scorer_enrichi(df.lazy(), sc).collect()

    variables = get_scorecard_variables(sc)
    for v in variables:
        assert f"_bin_{v}" in enriched.columns, f"_bin_{v} absent"
        assert f"_pts_{v}" in enriched.columns, f"_pts_{v} absent"
        assert f"_coef_{v}" not in enriched.columns, f"_coef_{v} aurait dû être dropped"

    for c in ("score_points", "score_logit", "score_proba"):
        assert c in enriched.columns, f"{c} absent"
    green("Toutes les colonnes attendues sont présentes (et _coef_* sont dropped).")


def test_scorer_pts_sum_equals_score_points():
    sc = build_scorecard()
    df = build_dataset(n=500, seed=2)
    enriched = scorer_enrichi(df.lazy(), sc).collect()

    variables = get_scorecard_variables(sc)
    pts_cols = [f"_pts_{v}" for v in variables]

    # Récupérer la constante en points
    règles = parser_scorecard(sc)
    cst_pts = next((r["points"] for r in règles if r["type"] == "constante"), 0)

    somme = enriched.select(pl.sum_horizontal(pts_cols)).to_series()
    écart = (somme + cst_pts - enriched["score_points"]).abs().max()
    assert écart < 1e-6, f"Écart maximal : {écart}"
    green("Σ _pts_* + intercept == score_points (cohérence interne).")


def test_proba_equals_sigmoid_logit():
    sc = build_scorecard()
    df = build_dataset(n=500, seed=3)
    enriched = scorer_enrichi(df.lazy(), sc).collect()

    logit = enriched["score_logit"].to_numpy()
    proba = enriched["score_proba"].to_numpy()
    expected = 1 / (1 + np.exp(-logit))
    écart = np.abs(proba - expected).max()
    assert écart < 1e-9, f"Écart sigmoid : {écart}"
    green("score_proba == sigmoid(score_logit).")


def test_no_null_pts_after_scoring():
    """Tous les individus doivent matcher exactement une règle par variable."""
    sc = build_scorecard()
    df = build_dataset(n=2000, seed=4)
    enriched = scorer_enrichi(df.lazy(), sc).collect()

    variables = get_scorecard_variables(sc)
    for v in variables:
        n_null = enriched[f"_pts_{v}"].null_count()
        assert n_null == 0, (
            f"_pts_{v} contient {n_null} nuls — couverture des bins incomplète"
        )
    green("Tous les individus matchent une règle pour chaque variable (pas de _pts nul).")


def test_missing_value_routes_correctly():
    """Une valeur manquante doit être routée vers le bin déclarant '+ MISSING'."""
    sc = build_scorecard()
    df = pl.DataFrame({
        "id_client": ["X1", "X2"],
        "anciennete": [None, 15.0],   # X1 missing, X2 dans [10, inf)
        "revenus": [3500.0, 3500.0],
        "nb_transactions": [10.0, 10.0],
        "incidents": [0.0, 0.0],
        "type_client": ["Artisan", "Artisan"],
        "region": ["IDF", "IDF"],
    })
    enriched = scorer_enrichi(df.lazy(), sc).collect()

    # Les deux doivent atterrir dans le même bin "[10, inf) + MISSING"
    bins_anc = enriched["_bin_anciennete"].to_list()
    assert bins_anc[0] == bins_anc[1] == "[10.0, inf) + MISSING", bins_anc
    green("Valeur manquante numérique routée vers le bin contenant '+ MISSING'.")


def test_categorical_missing_routes_correctly():
    sc = build_scorecard()
    df = pl.DataFrame({
        "id_client": ["X1", "X2"],
        "anciennete": [5.0, 5.0],
        "revenus": [3500.0, 3500.0],
        "nb_transactions": [10.0, 10.0],
        "incidents": [0.0, 0.0],
        "type_client": [None, "Auto-entrepreneur"],
        "region": ["IDF", "IDF"],
    })
    enriched = scorer_enrichi(df.lazy(), sc).collect()
    bins_type = enriched["_bin_type_client"].to_list()
    assert bins_type[0] == bins_type[1], bins_type
    assert "MISSING" in bins_type[0]
    green("Valeur catégorielle manquante routée vers le bin contenant MISSING.")


def test_proba_to_points_monotone():
    """Si on demande une proba plus grande, le seuil en points doit être plus petit
    (puisque plus de points = moins de proba de défaut)."""
    sc = build_scorecard()
    df = build_dataset(n=2000, seed=5)
    enriched = scorer_enrichi(df.lazy(), sc).collect()

    pts_low = proba_to_points(0.05, enriched, n_neighbors=50)
    pts_high = proba_to_points(0.30, enriched, n_neighbors=50)
    assert pts_low > pts_high, (
        f"Monotonie violée : pts({0.05})={pts_low} devrait être > pts({0.30})={pts_high}"
    )
    green(f"proba_to_points monotone : 5% → {pts_low} pts > 30% → {pts_high} pts.")


def test_decompose_individual_returns_all_variables():
    sc = build_scorecard()
    df = build_dataset(n=10, seed=6)
    enriched = scorer_enrichi(df.lazy(), sc).collect()
    variables = get_scorecard_variables(sc)
    row = enriched.row(0, named=True)
    decomp = decompose_individual(row, variables)
    assert len(decomp) == len(variables), f"{len(decomp)} vs {len(variables)}"
    for d in decomp:
        assert "variable" in d and "bin" in d and "points" in d
    # Tri par |points| décroissant
    abs_pts = [abs(d["points"]) for d in decomp]
    assert abs_pts == sorted(abs_pts, reverse=True)
    green("decompose_individual retourne toutes les variables, triées par |points| décroissant.")


def test_decomposition_sums_to_score():
    sc = build_scorecard()
    df = build_dataset(n=20, seed=7)
    enriched = scorer_enrichi(df.lazy(), sc).collect()
    variables = get_scorecard_variables(sc)

    règles = parser_scorecard(sc)
    cst_pts = next((r["points"] for r in règles if r["type"] == "constante"), 0)

    for i in range(enriched.height):
        row = enriched.row(i, named=True)
        decomp = decompose_individual(row, variables)
        somme = sum(d["points"] for d in decomp) + cst_pts
        assert somme == int(row["score_points"]), (
            f"Ligne {i} : décomposition={somme} vs score_points={row['score_points']}"
        )
    green("Σ décomposition individu + intercept == score_points pour 20 individus.")


def test_calibration_makes_sense():
    """Avec une cible cohérente, AUC > 0.6 attendu."""
    sc = build_scorecard()
    df = build_dataset(n=5000, seed=8)
    enriched = scorer_enrichi(df.lazy(), sc).collect()

    # AUC manuel via concordance
    proba = enriched["score_proba"].to_numpy()
    cible = enriched["defaut_obs"].to_numpy()
    pos = proba[cible == 1]
    neg = proba[cible == 0]

    # Échantillonnage 1000 paires pour rapidité
    rng = np.random.default_rng(0)
    idx_p = rng.integers(0, len(pos), 5000)
    idx_n = rng.integers(0, len(neg), 5000)
    concordant = np.mean(pos[idx_p] > neg[idx_n])
    assert concordant > 0.6, f"AUC empirique trop faible : {concordant:.3f}"
    green(f"Score discrimine : AUC empirique ≈ {concordant:.3f} > 0.60.")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Tests du moteur scorecard ===\n")
    tests = [
        test_parser_returns_label_for_each_rule,
        test_parser_handles_three_types,
        test_parser_inclut_missing_numerique,
        test_parser_inclut_missing_categorielle,
        test_scorer_columns_present,
        test_scorer_pts_sum_equals_score_points,
        test_proba_equals_sigmoid_logit,
        test_no_null_pts_after_scoring,
        test_missing_value_routes_correctly,
        test_categorical_missing_routes_correctly,
        test_proba_to_points_monotone,
        test_decompose_individual_returns_all_variables,
        test_decomposition_sums_to_score,
        test_calibration_makes_sense,
    ]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            fail(str(e))
            sys.exit(1)
    print(f"\n✅ {len(tests)} tests passés\n")
