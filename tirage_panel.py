"""
Plan de sondage annuel sur un panel mensuel (une ligne = un individu × un mois).

Règle
-----
Chaque année, l'échantillon tiré est réparti sur 6 ancres (janvier à juin).
Un individu ancré en m est observé en m et en m+6 : les 12 mois reçoivent donc
le même effectif. Les individus dont la présence ne permet aucune paire
(m, m+6) sont conservés avec une visite unique. Le tirage est refait chaque
année, indépendamment des précédents.

API
---
    plan = tirage_annuel(lf, seed=42)      # LazyFrame -> LazyFrame
    stats = stats_tirage(plan)             # dict de DataFrames (déjà collectés)
    afficher_stats(stats)                  # impression lisible

Les entrées/sorties sont paresseuses : seule la table d'éligibilité
(une ligne par individu × année) est matérialisée, car l'affectation aux ancres
n'est pas exprimable en expressions Polars.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["tirage_annuel", "stats_tirage", "afficher_stats",
           "diagnostic_eligibilite"]


# --------------------------------------------------------------------- outils
def _prepare(lf: pl.LazyFrame | pl.DataFrame, date_col: str | None,
             annee_col: str, mois_col: str) -> pl.LazyFrame:
    lf = lf.lazy() if isinstance(lf, pl.DataFrame) else lf
    if date_col and date_col in lf.collect_schema().names():
        lf = lf.with_columns(
            pl.col(date_col).dt.year().alias(annee_col),
            pl.col(date_col).dt.month().alias(mois_col),
        )
    return lf


def _eligibilite(base: pl.LazyFrame, id_col: str, annee_col: str, mois_col: str,
                 ecart: int, n_ancres: int) -> pl.DataFrame:
    """
    Une ligne par (individu, année) :
      masque       : bits des ancres utilisables (bit a <=> ancre a+1)
      masque_mois  : bits des mois réellement présents (bit m-1 <=> mois m)
      complet      : présent les 12 mois
    Collecté ici : c'est le seul objet manipulé par numpy.
    """
    presence = (
        base.select(id_col, annee_col, mois_col)
        .unique()
        .group_by([id_col, annee_col])
        .agg([(pl.col(mois_col) == m).any().alias(f"m{m}") for m in range(1, 13)])
        .sort([id_col, annee_col])      # ordre stable => seed reproductible
    )

    masque = pl.lit(0, dtype=pl.Int64)
    for a in range(n_ancres):
        m = a + 1
        masque = masque + (
            pl.when(pl.col(f"m{m}") & pl.col(f"m{m + ecart}"))
            .then(pl.lit(1 << a, dtype=pl.Int64)).otherwise(0)
        )

    masque_mois = pl.lit(0, dtype=pl.Int64)
    for m in range(1, 13):
        masque_mois = masque_mois + (
            pl.when(pl.col(f"m{m}"))
            .then(pl.lit(1 << (m - 1), dtype=pl.Int64)).otherwise(0)
        )

    return (
        presence
        .with_columns(
            masque=masque,
            masque_mois=masque_mois,
            complet=pl.sum_horizontal([pl.col(f"m{m}") for m in range(1, 13)]) == 12,
        )
        .select(id_col, annee_col, "masque", "masque_mois", "complet")
        .collect(engine="streaming")
    )


def _repartir(masques: np.ndarray, n_slots: int, rng: np.random.Generator,
              charge_initiale: np.ndarray | None = None) -> np.ndarray:
    """
    Affecte chaque individu à l'un des `n_slots` créneaux ouverts pour lui
    (bit à 1 dans son masque), en égalisant au mieux les effectifs.

    Les individus de même masque étant interchangeables, on raisonne par classe
    de masque, de la plus contrainte à la moins contrainte, et on remplit dans
    chaque classe les créneaux les moins garnis (remplissage par niveaux).

    charge_initiale : niveau de départ de chaque créneau, pour rattraper un
    déséquilibre déjà installé.
    """
    n = masques.shape[0]
    affectation = np.full(n, -1, dtype=np.int64)
    compte = (np.zeros(n_slots, dtype=np.int64) if charge_initiale is None
              else charge_initiale.astype(np.int64).copy())

    classes, inverse = np.unique(masques, return_inverse=True)
    ordre = sorted(range(len(classes)),
                   key=lambda k: bin(int(classes[k])).count("1"))

    for k in ordre:
        masque = int(classes[k])
        membres = np.flatnonzero(inverse == k)
        if masque == 0 or membres.size == 0:
            continue
        rng.shuffle(membres)                       # qui va où : tiré au sort
        possibles = np.array([s for s in range(n_slots) if masque >> s & 1])

        reste = membres.size
        quota = np.zeros(possibles.size, dtype=np.int64)
        while reste > 0:
            niveaux = compte[possibles] + quota
            candidats = np.flatnonzero(niveaux == niveaux.min())
            prise = min(reste, candidats.size)
            quota[rng.permutation(candidats)[:prise]] += 1
            reste -= prise

        debut = 0
        for pos, s in enumerate(possibles):
            fin = debut + quota[pos]
            affectation[membres[debut:fin]] = s
            debut = fin
        compte[possibles] += quota

    return affectation


# -------------------------------------------------------------------- tirage
def tirage_annuel(
    lf: pl.LazyFrame | pl.DataFrame,
    id_col: str = "id",
    date_col: str | None = "date",
    annee_col: str = "annee",
    mois_col: str = "mois",
    ecart: int = 6,
    taux_individus: float = 1.0,
    n_par_an: int | None = None,
    tolerer_une_visite: bool = True,
    seed: int | None = None,
) -> pl.LazyFrame:
    """
    Renvoie un LazyFrame : les lignes retenues du panel, enrichies de
    `mois_ancrage`, `groupe`, `rang_tirage` (1 ou 2) et `n_visites` (2 = paire
    complète, 1 = visite unique).

    Étapes, année par année :
      1. tirage aléatoire des individus (taux_individus ou n_par_an) ;
      2. les individus sans paire possible sont placés une seule fois, dans
         celui de leurs mois présents qui est le moins garni ;
      3. les partiels disposant d'une paire sont affectés aux ancres, en
         partant du déséquilibre laissé par l'étape 2 ;
      4. les individus complets passent en dernier : sans contrainte, ils
         comblent les ancres restées creuses et lissent les 12 mois.

    tolerer_une_visite=False exclut les individus sans paire (aucune visite).
    """
    rng = np.random.default_rng(seed)
    n_ancres = 12 - ecart

    base = _prepare(lf, date_col, annee_col, mois_col)
    type_mois = base.collect_schema()[mois_col]

    elig = _eligibilite(base, id_col, annee_col, mois_col, ecart, n_ancres)
    elig = elig.filter(pl.col("masque") > 0 if not tolerer_une_visite
                       else (pl.col("masque") > 0) | (pl.col("masque_mois") > 0))

    plans = []
    for (annee,), bloc in elig.group_by([annee_col], maintain_order=True):
        bloc = bloc.sort(id_col)

        # --- 1. tirage des individus -----------------------------------
        n_dispo = bloc.height
        cible = (n_par_an if n_par_an is not None
                 else int(round(n_dispo * taux_individus)))
        cible = min(cible, n_dispo)
        if cible == 0:
            continue
        bloc = bloc[np.sort(rng.permutation(n_dispo)[:cible])]

        masque = bloc["masque"].to_numpy()
        mois_dispo = bloc["masque_mois"].to_numpy()
        complet = bloc["complet"].to_numpy()

        ancre = np.full(bloc.height, -1, dtype=np.int64)
        mois_seul = np.full(bloc.height, -1, dtype=np.int64)
        charge = np.zeros(12, dtype=np.int64)          # effectif par mois

        # --- 2. visites uniques : le mois présent le moins garni --------
        seuls = np.flatnonzero(masque == 0)
        if seuls.size:
            mois_seul[seuls] = _repartir(mois_dispo[seuls], 12, rng, charge)
            charge += np.bincount(mois_seul[seuls], minlength=12)

        # --- 3. partiels avec paire, les plus contraints d'abord --------
        # handicap d'une ancre = charge moyenne de ses deux mois (a, a+ecart)
        handicap = (charge[:n_ancres] + charge[ecart:ecart + n_ancres]) // 2
        partiels = np.flatnonzero((masque > 0) & ~complet)
        if partiels.size:
            ancre[partiels] = _repartir(masque[partiels], n_ancres, rng, handicap)
            handicap = handicap + np.bincount(ancre[partiels], minlength=n_ancres)

        # --- 4. complets en dernier : ils comblent les creux -------------
        complets = np.flatnonzero(complet)
        if complets.size:
            ancre[complets] = _repartir(masque[complets], n_ancres, rng, handicap)

        plans.append(bloc.with_columns(
            mois_ancrage=pl.Series(np.where(ancre >= 0, ancre + 1, 0), dtype=pl.Int32),
            mois_seul=pl.Series(np.where(mois_seul >= 0, mois_seul + 1, 0),
                                dtype=pl.Int32),
        ))

    if not plans:
        return base.clear()

    plan = pl.concat(plans).drop("masque", "masque_mois", "complet")
    paire = plan.filter(pl.col("mois_ancrage") > 0)
    seule = plan.filter(pl.col("mois_ancrage") == 0)

    morceaux = [
        paire.with_columns(pl.col("mois_ancrage").alias(mois_col),
                           rang_tirage=pl.lit(1, pl.Int8),
                           n_visites=pl.lit(2, pl.Int8)),
        paire.with_columns((pl.col("mois_ancrage") + ecart).alias(mois_col),
                           rang_tirage=pl.lit(2, pl.Int8),
                           n_visites=pl.lit(2, pl.Int8)),
        seule.with_columns(pl.col("mois_seul").alias(mois_col),
                           rang_tirage=pl.lit(1, pl.Int8),
                           n_visites=pl.lit(1, pl.Int8)),
    ]

    long = (
        pl.concat(morceaux)
        .with_columns(
            groupe=pl.when(pl.col("mois_ancrage") > 0)
                     .then(pl.format("G{}", pl.col("mois_ancrage")))
                     .otherwise(pl.lit("visite_unique")),
            **{mois_col: pl.col(mois_col).cast(type_mois)},
        )
        .drop("mois_seul")
    )

    return base.join(long.lazy(), on=[id_col, annee_col, mois_col], how="inner")


# ------------------------------------------------------------------- contrôle
def diagnostic_eligibilite(
    lf: pl.LazyFrame | pl.DataFrame,
    id_col: str = "id",
    date_col: str | None = "date",
    annee_col: str = "annee",
    mois_col: str = "mois",
    ecart: int = 6,
) -> pl.DataFrame:
    """À lancer AVANT le tirage : qui est tirable, et avec quelle marge."""
    n_ancres = 12 - ecart
    base = _prepare(lf, date_col, annee_col, mois_col)
    elig = _eligibilite(base, id_col, annee_col, mois_col, ecart, n_ancres)
    n_possibles = pl.sum_horizontal(
        [(pl.col("masque") // (1 << a)) % 2 for a in range(n_ancres)])
    return (
        elig.with_columns(n_ancres_possibles=n_possibles)
        .group_by(annee_col)
        .agg(
            individus=pl.len(),
            complets=pl.col("complet").sum(),
            sans_paire=(pl.col("n_ancres_possibles") == 0).sum(),
            part_sans_paire=(pl.col("n_ancres_possibles") == 0).mean(),
            ancres_moy=pl.col("n_ancres_possibles")
                         .filter(pl.col("n_ancres_possibles") > 0).mean(),
        )
        .sort(annee_col)
    )


def stats_tirage(plan: pl.LazyFrame | pl.DataFrame, id_col="id",
                 annee_col="annee", mois_col="mois",
                 ecart: int = 6) -> dict[str, pl.DataFrame]:
    """
    Statistiques de contrôle, calculées en un seul passage sur le plan.

    resume    : par année — individus, visites, part à 2 visites, effectif
                mensuel min/moyen/max et écart relatif (indicateur d'uniformité)
    mensuel   : effectif par mois et par année, dont visites uniques
    groupes   : effectif par groupe (ancre)
    anomalies : violations éventuelles des règles (doit être vide)
    """
    p = plan.lazy() if isinstance(plan, pl.DataFrame) else plan

    par_mois = (
        p.group_by([annee_col, mois_col])
        .agg(n=pl.len(),
             visites_uniques=(pl.col("n_visites") == 1).sum(),
             premieres_visites=(pl.col("rang_tirage") == 1).sum())
        .sort([annee_col, mois_col])
        .collect()
    )

    par_individu = (
        p.group_by([id_col, annee_col])
        .agg(n=pl.len(), etendue=pl.col(mois_col).max() - pl.col(mois_col).min())
        .collect()
    )

    resume = (
        par_mois.group_by(annee_col)
        .agg(
            mois_min=pl.col("n").min(),
            mois_moyen=pl.col("n").mean().round(1),
            mois_max=pl.col("n").max(),
            visites=pl.col("n").sum(),
            visites_uniques=pl.col("visites_uniques").sum(),
        )
        .join(
            par_individu.group_by(annee_col).agg(
                individus=pl.len(),
                part_2_visites=(pl.col("n") == 2).mean().round(4),
            ),
            on=annee_col,
        )
        .with_columns(
            ecart_relatif=((pl.col("mois_max") - pl.col("mois_min"))
                           / pl.col("mois_moyen")).round(4)
        )
        .select(annee_col, "individus", "visites", "part_2_visites",
                "visites_uniques", "mois_min", "mois_moyen", "mois_max",
                "ecart_relatif")
        .sort(annee_col)
    )

    groupes = (
        p.group_by([annee_col, "groupe"])
        .agg(individus=pl.col(id_col).n_unique())
        .sort([annee_col, "groupe"]).collect()
    )

    anomalies = par_individu.filter(
        (pl.col("n") > 2)
        | ((pl.col("n") == 2) & (pl.col("etendue") != ecart))
    )

    return {"resume": resume, "mensuel": par_mois, "groupes": groupes,
            "anomalies": anomalies}


def afficher_stats(stats: dict[str, pl.DataFrame]) -> None:
    """Impression lisible des statistiques de contrôle."""
    print("=== Résumé par année ===")
    print(stats["resume"])
    print("\n=== Effectifs par mois ===")
    print(stats["mensuel"].pivot(on="mois", index="annee", values="n"))
    print("\n=== Taille des groupes ===")
    print(stats["groupes"].pivot(on="groupe", index="annee", values="individus"))
    n_anomalies = stats["anomalies"].height
    print(f"\n=== Anomalies : {n_anomalies} ===")
    if n_anomalies:
        print(stats["anomalies"].head(10))
    else:
        print("aucune : au plus 2 visites par individu et par an, "
              "espacées de l'écart demandé.")


# ------------------------------------------------------------------- exemple
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_ind, annees = 6_000, [2021, 2022, 2023]

    lignes = []
    for a in annees:
        for i in range(n_ind):
            if i % 10 < 6:                       # 60 % de complets
                mois = range(1, 13)
            elif i % 10 < 9:                     # 30 % de partiels
                mois = rng.choice(range(1, 13), rng.integers(4, 12), replace=False)
            else:                                # 10 % très peu présents
                mois = rng.choice(range(1, 13), rng.integers(1, 4), replace=False)
            lignes += [(i, a, int(m)) for m in mois]

    df = pl.DataFrame(lignes, schema=["id", "annee", "mois"], orient="row")
    df = df.with_columns(montant=pl.Series(rng.gamma(2, 1000, df.height)))
    print("panel :", f"{df.height:,}".replace(",", " "), "lignes\n")

    print(diagnostic_eligibilite(df, date_col=None), "\n")

    plan = tirage_annuel(df.lazy(), date_col=None, taux_individus=0.5, seed=42)
    print("sortie :", type(plan).__name__)
    afficher_stats(stats_tirage(plan))
