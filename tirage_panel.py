"""
Plan de sondage annuel sur un panel mensuel (une ligne = un individu × un mois).

Règle : chaque année, l'échantillon est réparti en 6 groupes de taille égale,
ancrés sur les mois 1 à 6 ; chaque groupe est observé deux fois, à `ecart` mois
d'intervalle (janvier → juillet, février → août, ...). Les 12 mois reçoivent
donc le même effectif, N/6. Le tirage est refait de zéro chaque année.

Difficulté traitée : un individu n'est pas forcément présent tous les mois. Une
ancre m n'est utilisable pour lui que si les mois m ET m+ecart existent dans ses
données. L'affectation aux ancres est donc un problème d'équilibrage sous
contrainte d'éligibilité, résolu ici exactement (remplissage par niveaux, en
traitant les individus les plus contraints en premier).

    plan = tirage_annuel(df, seed=42)          # sous-ensemble tiré
    controler(plan)                            # vérifie les règles
    repartition_mensuelle(plan)                # effectifs par mois
"""

from __future__ import annotations

import numpy as np
import polars as pl


# --------------------------------------------------------------------- outils
def _eligibilite(base: pl.DataFrame, id_col: str, annee_col: str, mois_col: str,
                 ecart: int, n_ancres: int) -> pl.DataFrame:
    """Pour chaque (individu, année) : masque binaire des ancres utilisables."""
    presence = (
        base.select(id_col, annee_col, mois_col)
        .unique()
        .group_by([id_col, annee_col])
        .agg([(pl.col(mois_col) == m).any().alias(f"m{m}") for m in range(1, 13)])
        .sort([id_col, annee_col])          # ordre stable => seed reproductible
    )
    masque = pl.lit(0, dtype=pl.Int64)
    for a in range(n_ancres):               # bit a <=> ancre (a+1)
        m = a + 1
        masque = masque + (
            pl.when(pl.col(f"m{m}") & pl.col(f"m{m + ecart}"))
            .then(pl.lit(1 << a, dtype=pl.Int64)).otherwise(0)
        )
    return (presence.with_columns(masque=masque)
                    .select(id_col, annee_col, "masque"))


def _repartir(masques: np.ndarray, n_ancres: int,
              rng: np.random.Generator) -> np.ndarray:
    """
    Affecte chaque individu à une ancre éligible, en égalisant au maximum les
    effectifs par ancre.

    Les individus partageant le même masque d'éligibilité sont interchangeables.
    On traite les classes de masque de la plus contrainte à la moins contrainte
    et, dans chaque classe, on remplit d'abord les ancres les moins garnies
    (remplissage par niveaux) : l'écart final entre ancres est minimal.
    """
    n = masques.shape[0]
    ancres = np.full(n, -1, dtype=np.int64)
    compte = np.zeros(n_ancres, dtype=np.int64)

    classes, inverse = np.unique(masques, return_inverse=True)
    ordre = sorted(range(len(classes)),
                   key=lambda k: bin(int(classes[k])).count("1"))

    for k in ordre:
        masque = int(classes[k])
        membres = np.flatnonzero(inverse == k)
        if masque == 0 or membres.size == 0:
            continue
        rng.shuffle(membres)
        possibles = np.array([a for a in range(n_ancres) if masque >> a & 1])

        # remplissage par niveaux : combien donner à chaque ancre possible
        reste = membres.size
        quota = np.zeros(possibles.size, dtype=np.int64)
        while reste > 0:
            niveaux = compte[possibles] + quota
            mini = niveaux.min()
            candidats = np.flatnonzero(niveaux == mini)
            prise = min(reste, candidats.size)
            # si plusieurs ancres à égalité, on n'en sert qu'une partie au hasard
            choix = rng.permutation(candidats)[:prise]
            quota[choix] += 1
            reste -= prise

        debut = 0
        for pos, a in enumerate(possibles):
            fin = debut + quota[pos]
            ancres[membres[debut:fin]] = a
            debut = fin
        compte[possibles] += quota

    return ancres


# ------------------------------------------------------------------- tirage
def tirage_annuel(
    df: pl.DataFrame,
    id_col: str = "id",
    date_col: str | None = "date",
    annee_col: str = "annee",
    mois_col: str = "mois",
    ecart: int = 6,
    taux_individus: float = 1.0,
    n_par_an: int | None = None,
    seed: int | None = None,
) -> pl.DataFrame:
    """
    Renvoie les lignes de `df` retenues par le plan, enrichies de
    `mois_ancrage`, `rang_tirage` (1 = 1re visite, 2 = 2e) et `groupe`.

    taux_individus : part des individus éligibles tirés chaque année.
    n_par_an       : effectif exact par année (prioritaire sur taux_individus) ;
                     idéalement un multiple de 12/ecart pour des groupes égaux.
    """
    rng = np.random.default_rng(seed)
    n_ancres = 12 - ecart                       # 6 ancres si ecart = 6

    base = df
    if date_col and date_col in df.columns:
        base = df.with_columns(
            pl.col(date_col).dt.year().alias(annee_col),
            pl.col(date_col).dt.month().alias(mois_col),
        )

    elig = _eligibilite(base, id_col, annee_col, mois_col, ecart, n_ancres)

    plans = []
    for (annee,), bloc in elig.filter(pl.col("masque") > 0).group_by(
            [annee_col], maintain_order=True):
        bloc = bloc.sort(id_col)
        n_dispo = bloc.height
        cible = n_par_an if n_par_an is not None else int(round(
            n_dispo * taux_individus))
        cible = min(cible, n_dispo)
        if cible == 0:
            continue

        choisis = rng.permutation(n_dispo)[:cible]
        bloc = bloc[np.sort(choisis)]

        ancres = _repartir(bloc["masque"].to_numpy(), n_ancres, rng)
        plans.append(bloc.with_columns(
            mois_ancrage=pl.Series(ancres + 1, dtype=pl.Int32)))

    if not plans:
        return base.clear()

    plan = pl.concat(plans).drop("masque")

    # ------------------------------------------- deux visites -> format long
    visite1 = plan.with_columns(
        pl.col("mois_ancrage").alias(mois_col), rang_tirage=pl.lit(1, pl.Int8))
    visite2 = plan.with_columns(
        (pl.col("mois_ancrage") + ecart).alias(mois_col),
        rang_tirage=pl.lit(2, pl.Int8))

    long = (
        pl.concat([visite1, visite2])
        .with_columns(
            groupe=pl.format("G{}", pl.col("mois_ancrage")),
            **{mois_col: pl.col(mois_col).cast(base.schema[mois_col])},
        )
        .sort([id_col, annee_col, mois_col])
    )

    return base.join(long, on=[id_col, annee_col, mois_col], how="inner")


# ---------------------------------------------------------------- contrôles
def controler(ech: pl.DataFrame, id_col="id", annee_col="annee", mois_col="mois",
              ecart: int = 6) -> pl.DataFrame:
    """Au plus 2 observations par individu et par an, espacées de `ecart`."""
    return (
        ech.group_by([id_col, annee_col])
        .agg(n=pl.len(), etendue=pl.col(mois_col).max() - pl.col(mois_col).min())
        .select(
            individus_annees=pl.len(),
            max_par_an=pl.col("n").max(),
            part_2_visites=(pl.col("n") == 2).mean(),
            ecarts_non_conformes=((pl.col("n") == 2)
                                  & (pl.col("etendue") != ecart)).sum(),
        )
    )


def repartition_mensuelle(ech: pl.DataFrame, annee_col="annee", mois_col="mois"):
    """Effectifs par mois : c'est la colonne à regarder pour l'uniformité."""
    return (
        ech.group_by([annee_col, mois_col]).agg(n=pl.len())
        .sort([annee_col, mois_col])
        .pivot(on=mois_col, index=annee_col, values="n")
    )


# ------------------------------------------------------------------- exemple
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_ind, annees = 6_000, [2021, 2022, 2023]

    # panel réaliste : la moitié des individus ont des mois manquants
    lignes = []
    for a in annees:
        for i in range(n_ind):
            if i % 2 == 0:
                mois = range(1, 13)
            else:
                mois = rng.choice(range(1, 13), rng.integers(5, 13), replace=False)
            lignes += [(i, a, int(m)) for m in mois]
    df = pl.DataFrame(lignes, schema=["id", "annee", "mois"], orient="row")
    df = df.with_columns(montant=pl.Series(rng.gamma(2, 1000, df.height)))
    print("panel :", f"{df.height:,}".replace(",", " "), "lignes\n")

    ech = tirage_annuel(df, date_col=None, taux_individus=0.5, seed=42)
    print(controler(ech))
    print("\nEffectifs par mois :")
    print(repartition_mensuelle(ech))
    print("\nGroupes (année 2021) :")
    print(ech.filter(pl.col("annee") == 2021)
             .group_by("groupe").agg(individus=pl.col("id").n_unique())
             .sort("groupe"))
    print("\nUn individu suivi sur 3 ans :")
    print(ech.filter(pl.col("id") == 0).select("annee", "mois", "groupe",
                                               "rang_tirage").sort("annee", "mois"))
