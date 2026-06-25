"""
woe_encoder_polars.py

Version Polars du calcul de WoE. Entrée et sortie en LazyFrame partout
où c'est possible.

- compute_woe   : seule étape qui matérialise des données. Un mapping
                  Python (dict) ne peut pas rester lazy, donc un .collect()
                  est exécuté ici, une fois, sur les colonnes nécessaires.
- apply_woe     : 100% lazy. Construit l'expression de mapping et la
                  chaîne sur le LazyFrame, aucun .collect() interne.
- fit_transform_woe : enchaîne les deux sur train + un autre LazyFrame
                  (validation, production), garantit le même mapping
                  pour les deux.
- unseen_bins_report : utilitaire d'audit, reste lazy, à .collect()
                  seulement quand le volume de données le permet.

Convention : target vaut 1 pour un défaut (mauvais), 0 pour un bon.
Cette fonction ne discrétise pas les variables continues, elle suppose
que le binning a déjà été fait en amont (XGBoost depth-1, OptBinning,
ou autre).
"""

from __future__ import annotations

import polars as pl


def compute_woe(
    lf: pl.LazyFrame,
    target: str,
    features: list[str],
    bad_label: int = 1,
    epsilon: float = 0.5,
) -> tuple[pl.DataFrame, dict[str, dict]]:
    """
    Calcule le WoE et la contribution à l'IV par bin pour chaque variable.

    Une requête group_by + agg par variable est lancée, puis exécutée
    en une seule fois via pl.collect_all (un passage sur les données
    par variable, pas un aller-retour répété).

    Parameters
    ----------
    lf : LazyFrame contenant les colonnes déjà binnées et la cible.
    target : nom de la colonne cible (0/1).
    features : colonnes déjà discrétisées (bins ou catégories).
    bad_label : valeur de la cible correspondant à un défaut.
    epsilon : lissage de Laplace, évite les WoE infinis sur un bin pur
              (0 défaut ou 0 bon dans un bin).

    Returns
    -------
    detail : DataFrame variable / bin / n_obs / n_good / n_bad / woe / iv_contrib.
             IV total d'une variable :
             detail.group_by("variable").agg(iv=pl.col("iv_contrib").sum())
    mapping : dict {variable: {bin_value: woe}}, prêt pour apply_woe().
              Les clés conservent le type d'origine de la colonne (pas
              de cast en string), pour matcher exactement à l'application.
    """
    is_bad = (pl.col(target) == bad_label).cast(pl.Int64)

    totals = lf.select(total_obs=pl.len(), total_bad=is_bad.sum()).collect()
    total_obs = totals["total_obs"][0]
    total_bad = totals["total_bad"][0]
    total_good = total_obs - total_bad

    if total_good == 0 or total_bad == 0:
        raise ValueError("Il faut au moins un bon et un mauvais dans le LazyFrame.")

    queries = [
        lf.with_columns(is_bad=is_bad)
        .group_by(col)
        .agg(n_obs=pl.len(), n_bad=pl.col("is_bad").sum())
        .with_columns(n_good=pl.col("n_obs") - pl.col("n_bad"))
        for col in features
    ]
    grouped = pl.collect_all(queries)

    detail_frames = []
    mapping: dict[str, dict] = {}

    for col, gdf in zip(features, grouped):
        n_bins = gdf.height
        pct_good = (gdf["n_good"] + epsilon) / (total_good + epsilon * n_bins)
        pct_bad = (gdf["n_bad"] + epsilon) / (total_bad + epsilon * n_bins)
        woe = (pct_good / pct_bad).log()
        iv_contrib = (pct_good - pct_bad) * woe

        # mapping : clés dans le type d'origine de la colonne (string,
        # catégorie, entier...), pas la version castée du tableau detail.
        mapping[col] = dict(zip(gdf[col].to_list(), woe.to_list()))

        bin_table = gdf.select(
            variable=pl.lit(col),
            bin=pl.col(col).cast(pl.Utf8),
            n_obs=pl.col("n_obs"),
            n_good=pl.col("n_good"),
            n_bad=pl.col("n_bad"),
        ).with_columns(woe=woe, iv_contrib=iv_contrib)
        detail_frames.append(bin_table)

    detail = pl.concat(detail_frames)
    return detail, mapping


def apply_woe(
    lf: pl.LazyFrame,
    mapping: dict[str, dict],
    missing_value: float = 0.0,
    suffix: str = "_woe",
    inplace_columns: bool = False,
) -> pl.LazyFrame:
    """
    Applique un mapping WoE (issu de compute_woe) à un LazyFrame.
    Reste lazy de bout en bout, aucun .collect() interne.

    Les bins absents du mapping (catégorie jamais vue à l'apprentissage,
    typiquement un signe de dérive de population) reçoivent
    `missing_value`, silencieusement. Pour auditer leur volume avant de
    les laisser passer, voir unseen_bins_report().

    Parameters
    ----------
    lf : LazyFrame à transformer, potentiellement différent de celui
         utilisé pour compute_woe (cas train -> validation/production).
    mapping : dict {variable: {bin_value: woe}}.
    missing_value : valeur attribuée aux bins non vus à l'apprentissage.
    suffix : suffixe des colonnes WoE créées.
    inplace_columns : si True, remplace les colonnes d'origine au lieu
                      d'ajouter des colonnes <var><suffix>.

    Returns
    -------
    LazyFrame avec les colonnes WoE ajoutées (ou remplacées).
    """
    exprs = []
    for col, woe_dict in mapping.items():
        target_col = col if inplace_columns else f"{col}{suffix}"
        exprs.append(
            pl.col(col)
            .replace_strict(woe_dict, default=missing_value, return_dtype=pl.Float64)
            .alias(target_col)
        )
    return lf.with_columns(exprs)


def unseen_bins_report(lf: pl.LazyFrame, mapping: dict[str, dict]) -> pl.LazyFrame:
    """
    Compte, pour chaque variable du mapping, le nombre de lignes dont le
    bin n'a pas été vu à l'apprentissage. Reste lazy : à .collect()
    quand le volume de données le permet (utile avant de lancer ça sur
    toute la production).
    """
    exprs = [
        (~pl.col(col).is_in(list(woe_dict.keys()))).sum().alias(col)
        for col, woe_dict in mapping.items()
    ]
    return lf.select(exprs)


def fit_transform_woe(
    lf_train: pl.LazyFrame,
    lf_other: pl.LazyFrame,
    target: str,
    features: list[str],
    bad_label: int = 1,
    epsilon: float = 0.5,
    **apply_kwargs,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.DataFrame, dict]:
    """
    Calcule le WoE sur lf_train (seul point de matérialisation) et
    applique le même mapping à lf_train et lf_other (validation,
    production...), lazy de bout en bout pour les deux sorties.

    Returns
    -------
    lf_train_woe, lf_other_woe, detail, mapping
    """
    detail, mapping = compute_woe(
        lf_train, target=target, features=features, bad_label=bad_label, epsilon=epsilon
    )
    lf_train_woe = apply_woe(lf_train, mapping, **apply_kwargs)
    lf_other_woe = apply_woe(lf_other, mapping, **apply_kwargs)
    return lf_train_woe, lf_other_woe, detail, mapping


if __name__ == "__main__":
    # Exemple minimal
    train = pl.DataFrame(
        {
            "age_bin": ["18-25", "26-40", "26-40", "41-60", "18-25", "41-60"],
            "income_bin": ["low", "high", "low", "high", "low", "high"],
            "default": [1, 0, 0, 0, 1, 1],
        }
    ).lazy()

    prod = pl.DataFrame(
        {
            "age_bin": ["26-40", "41-60", "60+"],  # "60+" jamais vu en train
            "income_bin": ["low", "high", "low"],
        }
    ).lazy()

    train_woe, prod_woe, detail, mapping = fit_transform_woe(
        train, prod, target="default", features=["age_bin", "income_bin"]
    )

    print(detail)
    print(detail.group_by("variable").agg(iv=pl.col("iv_contrib").sum()))
    print(mapping)

    # train_woe / prod_woe sont encore des LazyFrame ici, rien n'a été collecté
    print(type(prod_woe))
    print(prod_woe.collect())

    # audit avant de pousser le mapping sur un gros volume de production
    print(unseen_bins_report(prod, mapping).collect())
