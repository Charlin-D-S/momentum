"""
Matrices de transition flag1 x flag2 en Polars (+ rendu coloré via pandas Styler).

4 matrices :
  1. effectifs (n)
  2. pourcentages d'effectifs
  3. montants sommés
  4. pourcentages de montants

Normalisation des % : "all" (sur le total), "row" (par ligne = vraie matrice de
transition), "col" (par colonne).
"""

import polars as pl
import pandas as pd

ORDRE = ["vert", "orange", "rouge"]  # ordre d'affichage des modalités


# ---------------------------------------------------------------- coeur Polars
def croise(
    df: pl.DataFrame,
    f1: str = "flag1",
    f2: str = "flag2",
    valeur: str | None = None,
    ordre: list[str] = ORDRE,
) -> pl.DataFrame:
    """Tableau croisé : effectifs si valeur is None, sinon somme de `valeur`."""
    agg = pl.len().alias("v") if valeur is None else pl.col(valeur).sum().alias("v")

    long = (
        df.group_by([f1, f2])
        .agg(agg)
        .with_columns(pl.col("v").cast(pl.Float64))
    )

    wide = long.pivot(on=f2, index=f1, values="v", aggregate_function=None)

    # colonnes/lignes manquantes -> 0, puis remise dans l'ordre voulu
    for c in ordre:
        if c not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(c))
    wide = wide.select([f1] + ordre).fill_null(0.0)

    manquantes = [m for m in ordre if m not in wide[f1].to_list()]
    if manquantes:
        vide = pl.DataFrame(
            {f1: manquantes, **{c: [0.0] * len(manquantes) for c in ordre}}
        )
        wide = pl.concat([wide, vide], how="vertical")

    return (
        wide.with_columns(pl.col(f1).cast(pl.Enum(ordre)))
        .sort(f1)
        .with_columns(pl.col(f1).cast(pl.Utf8))
    )


def en_pourcentage(m: pl.DataFrame, f1: str = "flag1", how: str = "row",
                   ordre: list[str] = ORDRE) -> pl.DataFrame:
    """Convertit une matrice de comptes/montants en % (row / col / all)."""
    if how == "row":
        total = pl.sum_horizontal(ordre)
        return m.with_columns([
            pl.when(total > 0).then(pl.col(c) / total * 100).otherwise(0.0).alias(c)
            for c in ordre
        ])
    if how == "col":
        return m.with_columns([
            pl.when(pl.col(c).sum() > 0)
            .then(pl.col(c) / pl.col(c).sum() * 100).otherwise(0.0).alias(c)
            for c in ordre
        ])
    if how == "all":
        tot = sum(m[c].sum() for c in ordre)
        return m.with_columns([(pl.col(c) / tot * 100).alias(c) for c in ordre])
    raise ValueError("how ∈ {'row','col','all'}")


def avec_marges(m: pl.DataFrame, f1: str = "flag1",
                ordre: list[str] = ORDRE) -> pl.DataFrame:
    """Ajoute une colonne Total et une ligne Total."""
    m = m.with_columns(pl.sum_horizontal(ordre).alias("Total"))
    ligne = pl.DataFrame({f1: ["Total"],
                          **{c: [m[c].sum()] for c in ordre + ["Total"]}})
    return pl.concat([m, ligne], how="vertical")


# ------------------------------------------------------------------ affichage
def colorer(m: pl.DataFrame, titre: str, f1: str = "flag1",
            fmt: str = "{:,.0f}", cmap: str = "Blues"):
    """Rend la matrice colorée (dégradé par intensité). À afficher en notebook."""
    pdf = pd.DataFrame(m.to_dict(as_series=False)).set_index(f1)  # pas besoin de pyarrow
    cols = [c for c in pdf.columns if c != "Total"]
    sty = (
        pdf.style
        .background_gradient(cmap=cmap, subset=cols, axis=None)
        .format(fmt)
        .set_caption(titre)
        .set_table_styles([
            {"selector": "caption",
             "props": "caption-side:top; font-size:1.05em; font-weight:600; padding:6px;"},
            {"selector": "th", "props": "background-color:#f2f2f2;"},
        ])
    )
    if "Total" in pdf.columns:
        sty = sty.set_properties(subset=["Total"], **{"font-weight": "bold",
                                                      "background-color": "#eeeeee"})
    return sty


def colorer_feu(m: pl.DataFrame, titre: str, f1: str = "flag1",
                fmt: str = "{:,.1f}"):
    """Variante : couleur = dégradation (rouge) / amélioration (vert) du flag."""
    rang = {c: i for i, c in enumerate(ORDRE)}

    def style_cell(val, i, j):
        d = rang[j] - rang[i]                       # >0 = dégradation
        if d == 0:
            return "background-color:#e8e8e8;"
        base = "#c00000" if d > 0 else "#1a7f37"
        alpha = 0.18 + 0.22 * abs(d)
        return f"background-color:{base}{int(alpha*255):02x};"

    pdf = pd.DataFrame(m.to_dict(as_series=False)).set_index(f1)  # pas besoin de pyarrow
    cols = [c for c in pdf.columns if c != "Total"]

    def appliquer(df):
        out = pd.DataFrame("", index=df.index, columns=df.columns)
        for i in df.index:
            for j in cols:
                if i in rang and j in rang:
                    out.loc[i, j] = style_cell(df.loc[i, j], i, j)
        return out

    return (pdf.style.apply(appliquer, axis=None).format(fmt)
            .set_caption(titre)
            .set_table_styles([{"selector": "caption",
                                "props": "caption-side:top;font-weight:600;padding:6px;"}]))


def exporter_html(styles, chemin="matrices.html"):
    """Écrit plusieurs Styler dans un seul fichier HTML."""
    html = "<meta charset='utf-8'><style>table{border-collapse:collapse;margin:18px 0;}"\
           "td,th{border:1px solid #ddd;padding:6px 12px;text-align:right;}</style>"
    html += "".join(s.to_html() for s in styles)
    with open(chemin, "w", encoding="utf-8") as f:
        f.write(html)
    return chemin


# --------------------------------------------------------------------- exemple
if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    n = 5000
    df = pl.DataFrame({
        "flag1": rng.choice(ORDRE, n, p=[.6, .25, .15]),
        "flag2": rng.choice(ORDRE, n, p=[.5, .3, .2]),
        "montant": rng.gamma(2, 5000, n).round(2),
    })

    n_mat = avec_marges(croise(df))
    n_pct = en_pourcentage(croise(df), how="row")
    m_mat = avec_marges(croise(df, valeur="montant"))
    m_pct = en_pourcentage(croise(df, valeur="montant"), how="row")

    print(n_mat, n_pct, m_mat, m_pct, sep="\n\n")

    exporter_html([
        colorer(n_mat, "1. Effectifs (n)"),
        colorer(n_pct, "2. % par ligne (transition)", fmt="{:.1f}%", cmap="Purples"),
        colorer(m_mat, "3. Montants sommés (€)", cmap="Greens"),
        colorer(m_pct, "4. % du montant par ligne", fmt="{:.1f}%", cmap="Oranges"),
        colorer_feu(n_pct, "5. Lecture feu tricolore (% ligne)", fmt="{:.1f}%"),
    ])
