"""
Choix du seuil d'octroi à partir des probabilités de défaut.

Convention : sont ACCEPTÉS les dossiers dont la proba est STRICTEMENT INFÉRIEURE
au seuil.  Le seuil retenu est le PREMIER niveau de proba qui, s'il était
accepté, ferait dépasser la contrainte (taux de défaut ou volume).  Comme il est
exclu du périmètre accepté, la population acceptée respecte donc la contrainte.

    table_seuils(df, ...)        -> courbe complète (un point par niveau de proba)
    seuil_pour_defaut(...)       -> seuil pour un taux de défaut cible
    seuil_pour_acceptation(...)  -> seuil pour un taux d'acceptation cible
"""

from dataclasses import dataclass

import polars as pl


@dataclass
class Seuil:
    seuil: float          # proba plancher : accepté <=> proba < seuil
    n_acceptes: int
    taux_acceptation: float
    taux_defaut: float    # taux de défaut de la population acceptée
    n_defauts: int
    atteint: bool         # False = contrainte jamais atteinte, tout est accepté
    contrainte: str

    def __str__(self):
        s = "aucun (tout accepté)" if not self.atteint else f"{self.seuil:.6f}"
        return (f"seuil={s} | acceptés={self.n_acceptes:,} "
                f"({self.taux_acceptation:.2%}) | défaut acceptés="
                f"{self.taux_defaut:.2%} | contrainte: {self.contrainte}")


# --------------------------------------------------------------------- courbe
def table_seuils(
    df: pl.DataFrame,
    proba: str = "proba",
    cible: str = "defaut",
    poids: str | None = None,
) -> pl.DataFrame:
    """
    Un point par niveau de proba observé, avec les cumuls sur `proba <= niveau`.

    Colonnes : proba, n, n_defaut, n_cum, defaut_cum, taux_acceptation,
               taux_defaut (= taux de défaut si l'on acceptait jusqu'à ce niveau
               inclus).
    """
    y = pl.col(cible).cast(pl.Float64)
    if poids is None:
        n_expr, d_expr = pl.len().cast(pl.Float64), y.sum()
    else:
        w = pl.col(poids).cast(pl.Float64)
        n_expr, d_expr = w.sum(), (w * y).sum()

    t = (
        df.select([proba, cible] + ([poids] if poids else []))
        .drop_nulls([proba, cible])
        .group_by(proba)
        .agg(n=n_expr, n_defaut=d_expr)
        .sort(proba)
        .with_columns(
            n_cum=pl.col("n").cum_sum(),
            defaut_cum=pl.col("n_defaut").cum_sum(),
        )
    )
    total = t["n"].sum()
    return t.with_columns(
        taux_acceptation=pl.col("n_cum") / total,
        taux_defaut=pl.col("defaut_cum") / pl.col("n_cum"),
    )


# --------------------------------------------------------------- recherche
def _premier_depassement(t: pl.DataFrame, colonne: str, cible_val: float,
                         n_min: float, contrainte: str) -> Seuil:
    """Premier niveau de proba où `colonne` dépasse `cible_val`."""
    candidats = t.filter(
        (pl.col(colonne) > cible_val) & (pl.col("n_cum") >= n_min)
    )

    if candidats.height == 0:                    # jamais dépassé : on prend tout
        derniere = t.tail(1)
        return Seuil(
            seuil=float("inf"),
            n_acceptes=int(derniere["n_cum"][0]),
            taux_acceptation=float(derniere["taux_acceptation"][0]),
            taux_defaut=float(derniere["taux_defaut"][0]),
            n_defauts=int(derniere["defaut_cum"][0]),
            atteint=False,
            contrainte=contrainte,
        )

    seuil = float(candidats["proba"][0])
    # population réellement acceptée : proba < seuil, donc la ligne précédente
    avant = t.filter(pl.col("proba") < seuil)
    if avant.height == 0:                        # dépassement dès le 1er niveau
        return Seuil(seuil, 0, 0.0, float("nan"), 0, True, contrainte)

    d = avant.tail(1)
    return Seuil(
        seuil=seuil,
        n_acceptes=int(d["n_cum"][0]),
        taux_acceptation=float(d["taux_acceptation"][0]),
        taux_defaut=float(d["taux_defaut"][0]),
        n_defauts=int(d["defaut_cum"][0]),
        atteint=True,
        contrainte=contrainte,
    )


def seuil_pour_defaut(
    df: pl.DataFrame | None,
    taux_defaut_cible: float,
    proba: str = "proba",
    cible: str = "defaut",
    poids: str | None = None,
    n_min: float = 0,
    monotone: bool = True,
    table: pl.DataFrame | None = None,
) -> Seuil:
    """
    Seuil tel que le taux de défaut des acceptés reste <= `taux_defaut_cible`.

    n_min    : nombre minimal d'acceptés avant d'autoriser un dépassement
               (évite un seuil fixé sur une poignée d'observations bruitées).
    monotone : lisse le taux de défaut cumulé par un maximum courant, pour que
               le premier franchissement soit définitif et non un accident de
               petit effectif. Mettre False pour une lecture brute.
    table    : courbe déjà calculée par table_seuils (évite de refaire le
               balayage des données à chaque appel).
    """
    t = table if table is not None else table_seuils(df, proba, cible, poids)
    col = "taux_defaut"
    if monotone:
        t = t.with_columns(taux_defaut_lisse=pl.col("taux_defaut").cum_max())
        col = "taux_defaut_lisse"
    return _premier_depassement(
        t, col, taux_defaut_cible, n_min,
        f"taux de défaut des acceptés <= {taux_defaut_cible:.2%}",
    )


def seuil_pour_acceptation(
    df: pl.DataFrame | None,
    taux_acceptation_cible: float,
    proba: str = "proba",
    cible: str = "defaut",
    poids: str | None = None,
    table: pl.DataFrame | None = None,
) -> Seuil:
    """Seuil tel que la part d'acceptés reste <= `taux_acceptation_cible`."""
    t = table if table is not None else table_seuils(df, proba, cible, poids)
    return _premier_depassement(
        t, "taux_acceptation", taux_acceptation_cible, 0,
        f"taux d'acceptation <= {taux_acceptation_cible:.2%}",
    )


def appliquer(df: pl.DataFrame, s: Seuil, proba: str = "proba") -> pl.DataFrame:
    """Ajoute la décision d'octroi (1 = accepté)."""
    return df.with_columns(
        accepte=(pl.col(proba) < s.seuil).cast(pl.Int8)
    )


# ------------------------------------------------------------------- exemple
if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)
    n = 20_000
    p = np.clip(rng.beta(2, 18, n), 0, 1).round(4)      # probas 0-1, ~10 %
    y = rng.binomial(1, p)                              # défaut cohérent
    df = pl.DataFrame({"proba": p, "defaut": y})

    print("Base :", n, "dossiers, défaut global",
          f"{df['defaut'].mean():.2%}\n")

    for cible_defaut in (0.02, 0.05, 0.08, 0.50):
        print(seuil_pour_defaut(df, cible_defaut, n_min=100))
    print()
    for cible_acc in (0.10, 0.50, 0.90):
        print(seuil_pour_acceptation(df, cible_acc))

    print("\nExtrait de la courbe :")
    print(table_seuils(df).select(
        "proba", "n", "n_cum", "taux_acceptation", "taux_defaut"
    ).head(8))
