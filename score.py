from sklearn.metrics import roc_auc_score, roc_curve
import polars as pl
import pandas as pd
import numpy as np


def evaluer_segment(
    lf: pl.LazyFrame,
    col_cible: str,
    col_proba: str,
    filtre: pl.Expr | None = None,
) -> pd.DataFrame:
    """
    Filtre un LazyFrame, collecte les données nécessaires et retourne
    un DataFrame de métriques de performance.

    Paramètres
    ----------
    lf        : LazyFrame scoré (contient col_cible et col_proba).
    col_cible : Nom de la colonne cible binaire (0/1).
    col_proba : Nom de la colonne de probabilité de défaut (score_proba).
    filtre    : Expression Polars de filtrage (None = pas de filtre).

    Retourne
    --------
    DataFrame pandas avec une ligne et les colonnes :
        effectif, nb_defauts, taux_defaut, pd_moyenne,
        auc, gini, ks_stat
    """
    # Application du filtre
    lf_filtre = lf.filter(filtre) if filtre is not None else lf

    # Collecte minimale — uniquement les colonnes nécessaires
    df = (
        lf_filtre
        .select([col_cible, col_proba])
        .collect()
    )

    y    = df[col_cible].to_numpy()
    prob = df[col_proba].to_numpy()

    effectif   = len(y)
    nb_defauts = int(y.sum())
    taux_defaut = nb_defauts / effectif
    pd_moyenne  = prob.mean()

    # AUC & Gini
    auc  = roc_auc_score(y, prob)
    gini = 2 * auc - 1

    # KS stat
    fpr, tpr, _ = roc_curve(y, prob)
    ks_stat = float(np.max(tpr - fpr))

    return pd.DataFrame([{
        "effectif":    effectif,
        "nb_defauts":  nb_defauts,
        "taux_defaut": round(taux_defaut, 4),
        "pd_moyenne":  round(pd_moyenne,  4),
        "auc":         round(auc,         4),
        "gini":        round(gini,        4),
        "ks_stat":     round(ks_stat,     4),
    }])
    
    segments = {
    "total":       None,
    "secteur_AB":  pl.col("secteur").is_in(["A", "B"]),
    "anciens":     pl.col("anciennete") > 24,
    "nouveaux":    pl.col("anciennete") <= 24,
}

résumé = pd.concat(
    [evaluer_segment(lf_scoré, "defaut", "score_proba", f).assign(segment=nom)
     for nom, f in segments.items()],
    ignore_index=True,
).set_index("segment")

print(résumé)
    
    
    
    import polars as pl
import numpy as np
import ast
import re


def parser_scorecard(sc: pl.DataFrame) -> list[dict]:
    """
    Parse la scorecard en une liste de règles prêtes à être appliquées.
    Chaque règle est un dict avec les clés :
        variable, type ('constante', 'numerique', 'categorielle'),
        coef, points_1000, et les paramètres de matching.
    """
    règles = []

    for row in sc.iter_rows(named=True):
        var    = row["Variables"]
        label  = row["Label"]
        coef   = row["coef"]
        points = row["points_1000"]

        # Constante
        if label == "-":
            règles.append({
                "variable": var, "type": "constante",
                "coef": coef, "points": points,
            })
            continue

        # Catégorielle : label commence par "["  ET contient des quotes
        # ex : "['A', 'B']" ou "['1', 'MISSING']"
        if label.startswith("[") and "'" in label:
            modalités = ast.literal_eval(label)  # → liste Python de strings
            inclut_missing = "MISSING" in modalités
            valeurs = [m for m in modalités if m != "MISSING"]
            règles.append({
                "variable": var, "type": "categorielle",
                "coef": coef, "points": points,
                "valeurs": valeurs,           # liste de strings (sans MISSING)
                "inclut_missing": inclut_missing,
            })
            continue

        # Numérique : label de type "[a, b)" ou "[a, b) + MISSING"
        inclut_missing = "+ MISSING" in label
        label_num = label.replace("+ MISSING", "").strip()

        # Extraction des bornes via regex
        match = re.match(r"\[([^,]+),\s*([^)]+)\)", label_num)
        if match:
            borne_inf = match.group(1).strip()
            borne_sup = match.group(2).strip()
            borne_inf = -np.inf if borne_inf == "-inf" else float(borne_inf)
            borne_sup =  np.inf if borne_sup ==  "inf" else float(borne_sup)
            règles.append({
                "variable": var, "type": "numerique",
                "coef": coef, "points": points,
                "borne_inf": borne_inf,   # incluse
                "borne_sup": borne_sup,   # exclue
                "inclut_missing": inclut_missing,
            })

    return règles


def scorer(
    lf: pl.LazyFrame,
    sc: pl.DataFrame,
    col_id: str | None = None,
) -> pl.LazyFrame:
    """
    Applique la scorecard sur un LazyFrame et retourne un LazyFrame enrichi
    avec les colonnes score_points et score_proba.

    Paramètres
    ----------
    lf     : LazyFrame à scorer (variables en colonnes).
    sc     : DataFrame scorecard avec colonnes Variables, Label, coef, points_1000.
    col_id : Colonne identifiant individu (optionnel, conservée en tête).

    Retourne
    --------
    LazyFrame avec colonnes supplémentaires :
        score_points : somme des points_1000 matchés (scorecard additive)
        score_logit  : somme des coef matchés (log-odds)
        score_proba  : probabilité de défaut = sigmoid(score_logit)
    """
    règles = parser_scorecard(sc)

    # Regroupement par variable
    variables = {}
    constante_coef   = 0.0
    constante_points = 0.0

    for r in règles:
        if r["type"] == "constante":
            constante_coef   = r["coef"]
            constante_points = r["points"]
        else:
            variables.setdefault(r["variable"], []).append(r)

    # Construction des expressions Polars — une par variable
    exprs_points = []
    exprs_coef   = []

    for var, règles_var in variables.items():
        type_var = règles_var[0]["type"]

        if type_var == "numerique":
            # Chaîne de when/then pour les tranches numériques
            expr_p = pl.lit(None, dtype=pl.Float64)
            expr_c = pl.lit(None, dtype=pl.Float64)

            for r in règles_var:
                # Condition numérique : borne_inf <= col < borne_sup
                cond_num = (
                    (pl.col(var) >= r["borne_inf"]) &
                    (pl.col(var) <  r["borne_sup"])
                )
                # Condition missing
                cond = (
                    (cond_num | pl.col(var).is_null())
                    if r["inclut_missing"]
                    else cond_num
                )
                expr_p = pl.when(cond).then(pl.lit(r["points"])).otherwise(expr_p)
                expr_c = pl.when(cond).then(pl.lit(r["coef"]  )).otherwise(expr_c)

        else:  # categorielle
            expr_p = pl.lit(None, dtype=pl.Float64)
            expr_c = pl.lit(None, dtype=pl.Float64)

            # Cast en string pour comparaison uniforme (gère les int catégoriels)
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
                expr_c = pl.when(cond).then(pl.lit(r["coef"]  )).otherwise(expr_c)

        exprs_points.append(expr_p.alias(f"_pts_{var}"))
        exprs_coef.append(  expr_c.alias(f"_coef_{var}"))

    # Colonnes intermédiaires → somme → score final
    cols_pts  = [f"_pts_{v}"  for v in variables]
    cols_coef = [f"_coef_{v}" for v in variables]

    return (
        lf
        .with_columns(exprs_points + exprs_coef)
        .with_columns([
            (pl.sum_horizontal([pl.col(c) for c in cols_pts ]) + constante_points)
              .alias("score_points"),
            (pl.sum_horizontal([pl.col(c) for c in cols_coef]) + constante_coef)
              .alias("score_logit"),
        ])
        .with_columns(
            (1 / (1 + (-pl.col("score_logit")).exp())).alias("score_proba")
        )
        .drop(cols_pts + cols_coef)
    )


résultat = scorer(lf, scorecard, col_id="id_client")

résultat.select(["id_client", "score_points", "score_proba"]).collect()
