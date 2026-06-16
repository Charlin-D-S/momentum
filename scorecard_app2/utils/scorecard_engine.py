"""
Moteur scorecard : parsing, scoring, décomposition.

Le LazyFrame de sortie est enrichi avec, pour chaque variable de la scorecard :
    _bin_{var}  : libellé du bin (str)        → filtres, cartes de profil
    _pts_{var}  : contribution en points       → importance, décomposition
plus les colonnes globales :
    score_points, score_logit, score_proba.
"""
from __future__ import annotations

import ast
import re

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parser_scorecard(sc: pl.DataFrame) -> list[dict]:
    """
    Parse la scorecard en une liste de règles.
    Chaque règle est un dict avec, selon le type :
        - constante  : variable, type, coef, points
        - categorielle : + label, valeurs, inclut_missing
        - numerique  : + label, borne_inf, borne_sup, inclut_missing
    """
    règles: list[dict] = []
    for row in sc.iter_rows(named=True):
        var = row["Variables"]
        label = row["Label"]
        coef = row["coef"]
        points = row["points_1000"]

        # constante (intercept)
        if label == "-":
            règles.append({
                "variable": var, "type": "constante",
                "coef": coef, "points": points,
            })
            continue

        # catégorielle : ex. "['A', 'B']" ou "['1', 'MISSING']"
        if label.startswith("[") and "'" in label:
            modalités = ast.literal_eval(label)
            inclut_missing = "MISSING" in modalités
            valeurs = [m for m in modalités if m != "MISSING"]
            règles.append({
                "variable": var, "type": "categorielle",
                "label": label,
                "coef": coef, "points": points,
                "valeurs": valeurs,
                "inclut_missing": inclut_missing,
            })
            continue

        # numérique : ex. "[a, b)" ou "[a, b) + MISSING"
        inclut_missing = "+ MISSING" in label
        label_num = label.replace("+ MISSING", "").strip()
        match = re.match(r"\[([^,]+),\s*([^)]+)\)", label_num)
        if match:
            borne_inf = match.group(1).strip()
            borne_sup = match.group(2).strip()
            borne_inf = -np.inf if borne_inf == "-inf" else float(borne_inf)
            borne_sup = np.inf if borne_sup == "inf" else float(borne_sup)
            règles.append({
                "variable": var, "type": "numerique",
                "label": label,
                "coef": coef, "points": points,
                "borne_inf": borne_inf,
                "borne_sup": borne_sup,
                "inclut_missing": inclut_missing,
            })
    return règles


def get_scorecard_variables(sc: pl.DataFrame) -> list[str]:
    """Liste ordonnée des variables (hors intercept)."""
    règles = parser_scorecard(sc)
    seen: list[str] = []
    for r in règles:
        if r["type"] != "constante" and r["variable"] not in seen:
            seen.append(r["variable"])
    return seen


def get_variable_type(sc: pl.DataFrame, var: str) -> str:
    """Retourne 'numerique' ou 'categorielle' pour une variable."""
    règles = parser_scorecard(sc)
    for r in règles:
        if r["variable"] == var and r["type"] != "constante":
            return r["type"]
    raise ValueError(f"Variable {var!r} introuvable dans la scorecard.")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def scorer_enrichi(lf: pl.LazyFrame, sc: pl.DataFrame) -> pl.LazyFrame:
    """
    Applique la scorecard et enrichit le LazyFrame avec :
        _bin_{var}, _pts_{var}, score_points, score_logit, score_proba.
    Les colonnes _coef_{var} sont calculées en interne puis supprimées.
    """
    règles = parser_scorecard(sc)

    variables: dict[str, list[dict]] = {}
    constante_coef = 0.0
    constante_points = 0.0

    for r in règles:
        if r["type"] == "constante":
            constante_coef = r["coef"]
            constante_points = r["points"]
        else:
            variables.setdefault(r["variable"], []).append(r)

    exprs_points: list[pl.Expr] = []
    exprs_coef: list[pl.Expr] = []
    exprs_bin: list[pl.Expr] = []

    for var, règles_var in variables.items():
        type_var = règles_var[0]["type"]

        expr_p = pl.lit(None, dtype=pl.Float64)
        expr_c = pl.lit(None, dtype=pl.Float64)
        expr_b = pl.lit(None, dtype=pl.Utf8)

        if type_var == "numerique":
            # On considère NaN comme missing (robustesse, certains parquets exportent NaN)
            is_missing = pl.col(var).is_null() | pl.col(var).is_nan()
            for r in règles_var:
                cond_num = (
                    (pl.col(var) >= r["borne_inf"]) &
                    (pl.col(var) < r["borne_sup"])
                )
                cond = (cond_num | is_missing) if r["inclut_missing"] else cond_num
                expr_p = pl.when(cond).then(pl.lit(r["points"])).otherwise(expr_p)
                expr_c = pl.when(cond).then(pl.lit(r["coef"])).otherwise(expr_c)
                expr_b = pl.when(cond).then(pl.lit(r["label"])).otherwise(expr_b)
        else:
            col_str = pl.col(var).cast(pl.Utf8)
            for r in règles_var:
                valeurs_str = [str(v) for v in r["valeurs"]]
                cond_val = col_str.is_in(valeurs_str)
                cond_missing = pl.col(var).is_null()

                if r["inclut_missing"] and valeurs_str:
                    cond = cond_val | cond_missing
                elif r["inclut_missing"] and not valeurs_str:
                    cond = cond_missing
                else:
                    cond = cond_val

                expr_p = pl.when(cond).then(pl.lit(r["points"])).otherwise(expr_p)
                expr_c = pl.when(cond).then(pl.lit(r["coef"])).otherwise(expr_c)
                expr_b = pl.when(cond).then(pl.lit(r["label"])).otherwise(expr_b)

        exprs_points.append(expr_p.alias(f"_pts_{var}"))
        exprs_coef.append(expr_c.alias(f"_coef_{var}"))
        exprs_bin.append(expr_b.alias(f"_bin_{var}"))

    cols_pts = [f"_pts_{v}" for v in variables]
    cols_coef = [f"_coef_{v}" for v in variables]

    return (
        lf
        .with_columns(exprs_points + exprs_coef + exprs_bin)
        .with_columns([
            (pl.sum_horizontal([pl.col(c) for c in cols_pts]) + constante_points)
                .alias("score_points"),
            (pl.sum_horizontal([pl.col(c) for c in cols_coef]) + constante_coef)
                .alias("score_logit"),
        ])
        .with_columns(
            (1 / (1 + (-pl.col("score_logit")).exp())).alias("score_proba")
        )
        .drop(cols_coef)
    )


# ---------------------------------------------------------------------------
# Utilitaires pour les pages
# ---------------------------------------------------------------------------
def scorecard_table(sc: pl.DataFrame) -> pl.DataFrame:
    """
    Vue lisible de la scorecard pour la Page 1 :
    Variables, Label, points_1000, coef, ordonnée par |points|.
    """
    return (
        sc
        .filter(pl.col("Label") != "-")
        .select(["Variables", "Label", "points_1000", "coef"])
    )


def variable_importance(sc: pl.DataFrame) -> pl.DataFrame:
    """
    Importance d'une variable = étendue (max - min) des points_1000 sur ses bins.
    Indicateur intrinsèque à la scorecard (indépendant des données).
    """
    return (
        sc
        .filter(pl.col("Label") != "-")
        .group_by("Variables")
        .agg([
            (pl.col("points_1000").max() - pl.col("points_1000").min()).alias("importance"),
            pl.col("points_1000").max().alias("pts_max"),
            pl.col("points_1000").min().alias("pts_min"),
            pl.col("Label").count().alias("n_bins"),
        ])
        .sort("importance", descending=True)
    )


def proba_to_points(proba_seuil: float, df: pl.DataFrame, n_neighbors: int = 50) -> int:
    """
    Convertit un seuil de probabilité en seuil de points_1000 en cherchant
    dans les données les individus dont la proba est la plus proche du seuil.

    On prend la médiane des points_1000 des n_neighbors plus proches voisins
    en probabilité — robuste aux quelques aberrations.
    """
    if df.is_empty():
        return 0
    proches = (
        df.with_columns((pl.col("score_proba") - proba_seuil).abs().alias("_dist"))
          .sort("_dist")
          .head(n_neighbors)
          .select("score_points")
          .to_series()
    )
    return int(round(proches.median()))


def decompose_individual(row: dict, variables: list[str]) -> list[dict]:
    """
    Décomposition du score d'un individu à partir d'une ligne du DataFrame enrichi.
    Retourne une liste de tuples (variable, bin, points) triée par |points| décroissant.
    """
    décomposition = []
    for v in variables:
        bin_val = row.get(f"_bin_{v}")
        pts_val = row.get(f"_pts_{v}")
        if pts_val is not None:
            décomposition.append({
                "variable": v,
                "bin": bin_val if bin_val is not None else "—",
                "points": int(pts_val) if not np.isnan(pts_val) else 0,
            })
    décomposition.sort(key=lambda x: abs(x["points"]), reverse=True)
    return décomposition
