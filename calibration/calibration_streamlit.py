"""
calibration_streamlit.py
========================

Affichage Streamlit du protocole de contrôle de la calibration
(section 5.6 du mémoire). S'appuie sur `calibration_tests.py`.

Usage — une seule fonction, un DataFrame, les noms de colonnes
--------------------------------------------------------------

    import polars as pl
    from calibration_streamlit import afficher_calibration

    df = pl.read_parquet("scores_production_2025.parquet")

    afficher_calibration(
        df,
        y="defaut_12m",             # obligatoire — défaut observé 0/1
        p="pd_predite",             # obligatoire — PD prédite dans [0, 1]
        classe="classe_risque",     # facultatif  — active le backtesting par classe
        emprunteur="id_client",     # facultatif  — active le bootstrap par grappes
        segment="perimetre",        # facultatif  — ajoute un filtre interactif
    )

Tout le reste s'affiche : synthèse, courbe de calibration, pente et intercept,
tests globaux, tableau réglementaire par classe, décomposition de Brier, exports.

`p` accepte aussi une liste de colonnes, par exemple
`p=["grille_logistique", "xgboost_challenger"]` : un sélecteur apparaît et un
onglet de comparaison est ajouté.

Le DataFrame peut être un `polars.DataFrame`, un `polars.LazyFrame`, un
`pandas.DataFrame` ou un dictionnaire de tableaux. La conversion est interne.

Dépendances : streamlit, altair (fourni avec Streamlit), pandas, numpy, scipy,
statsmodels. Polars est facultatif.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from calibration_tests import (
    brier_decomposition,
    calibration_curve,
    calibration_intercept_slope,
    cluster_bootstrap_ci,
    cox_calibration_test,
    ece_mce,
    expit,
    grade_report,
    hosmer_lemeshow_test,
    logit,
    spiegelhalter_test,
)

__all__ = ["afficher_calibration", "resultats_calibration"]

# --------------------------------------------------------------------------- #
# Présentation
# --------------------------------------------------------------------------- #

def _pleine_largeur() -> dict:
    """`width="stretch"` sur Streamlit >= 1.49, `use_container_width=True` avant.

    Évite à la fois l'avertissement de dépréciation sur les versions récentes et
    l'erreur de paramètre inconnu sur les versions plus anciennes, fréquentes en
    environnement bancaire.
    """
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return {"use_container_width": True}
    return {"width": "stretch"} if (major, minor) >= (1, 49) else {"use_container_width": True}


LARGE = _pleine_largeur()

COULEURS_FEU = {
    "VERT": "#1a7f37",
    "JAUNE": "#bf8700",
    "ORANGE": "#d97706",
    "ROUGE": "#c00000",
    "NA": "#6b7280",
}
BLEU, ROUGE, GRIS = "#1f4e79", "#c00000", "#9ca3af"


# --------------------------------------------------------------------------- #
# Conversion des entrées
# --------------------------------------------------------------------------- #

def _vers_pandas(df) -> pd.DataFrame:
    """Accepte polars.DataFrame / LazyFrame, pandas.DataFrame, dict, list de dicts."""
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, "collect") and hasattr(df, "lazy"):          # polars.LazyFrame
        df = df.collect()
    if hasattr(df, "to_pandas"):                                 # polars.DataFrame, pyarrow
        return df.to_pandas()
    if isinstance(df, dict):
        return pd.DataFrame(df)
    if isinstance(df, (list, tuple)):
        return pd.DataFrame(list(df))
    raise TypeError(
        "df doit être un polars.DataFrame, un polars.LazyFrame, un pandas.DataFrame, "
        f"un dictionnaire ou une liste de dictionnaires (reçu : {type(df).__name__})."
    )


def _liste(x: Union[str, Sequence[str], None]) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


# --------------------------------------------------------------------------- #
# Calculs mis en cache
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def _globaux(y: np.ndarray, p: np.ndarray, n_bins: int) -> dict:
    return {
        "ece": ece_mce(y, p, n_bins=n_bins),
        "reg": calibration_intercept_slope(y, p).to_dict(),
        "cox": cox_calibration_test(y, p),
        "spiegelhalter": spiegelhalter_test(y, p),
        "hosmer_lemeshow": hosmer_lemeshow_test(y, p, g=10),
        "brier": brier_decomposition(y, p, n_bins=n_bins),
        "courbe": calibration_curve(y, p, n_bins=n_bins),
    }


@st.cache_data(show_spinner=False)
def _par_classe(y: np.ndarray, p: np.ndarray, classe: np.ndarray, rho: float) -> pd.DataFrame:
    return grade_report(y, p, classe, rho=rho)


@st.cache_data(show_spinner=False)
def _bootstrap_pente(y: np.ndarray, p: np.ndarray, grappe: np.ndarray, n_boot: int) -> dict:
    return cluster_bootstrap_ci(
        y, p, grappe,
        lambda yy, pp: calibration_intercept_slope(yy, pp).slope,
        n_boot=n_boot, seed=0,
    )


@st.cache_data(show_spinner=False)
def _loess(y: np.ndarray, p: np.ndarray, frac: float) -> pd.DataFrame:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    # it=0 est impératif : les itérations robustes de lowess, calibrées pour des
    # résidus continus, écrasent la courbe vers zéro sur une cible binaire rare.
    res = lowess(y, logit(p), frac=frac, it=0, return_sorted=True)
    pas = max(1, len(res) // 400)
    return pd.DataFrame(
        {"pd_predite": expit(res[::pas, 0]), "taux_lisse": np.clip(res[::pas, 1], 0, 1)}
    )


def resultats_calibration(
    df,
    y: str,
    p: str,
    classe: Optional[str] = None,
    rho: float = 0.08,
    n_bins: int = 20,
) -> dict:
    """Version sans interface : renvoie le dictionnaire de résultats.

    Utile pour produire les tableaux du mémoire hors Streamlit, ou pour
    alimenter un autre composant.
    """
    pdf = _vers_pandas(df)
    yy = pdf[y].to_numpy(dtype=float)
    pp = pdf[p].to_numpy(dtype=float)
    res = _globaux.__wrapped__(yy, pp, n_bins)
    if classe:
        res["par_classe"] = grade_report(yy, pp, pdf[classe].astype(str).to_numpy(), rho=rho)
    return res


# --------------------------------------------------------------------------- #
# Composants graphiques
# --------------------------------------------------------------------------- #

def _badge(texte: str, couleur: str) -> str:
    return (
        f'<span style="background:{couleur};color:white;padding:2px 10px;'
        f'border-radius:10px;font-size:0.80rem;font-weight:600;">{texte}</span>'
    )


def _verdict_pente(pente: float, ic: tuple) -> tuple[str, str]:
    if ic[0] <= 1.0 <= ic[1]:
        return "Pente conforme", COULEURS_FEU["VERT"]
    if pente < 1:
        return "Prédictions trop dispersées", COULEURS_FEU["ORANGE"]
    return "Prédictions trop plates — risque sous-estimé", COULEURS_FEU["ROUGE"]


def _verdict_intercept(alpha_: float, seuil: float = 0.10) -> tuple[str, str]:
    if abs(alpha_) < seuil:
        return "Niveau correct", COULEURS_FEU["VERT"]
    return (
        f"Niveau global {'surestimé' if alpha_ < 0 else 'sous-estimé'}",
        COULEURS_FEU["ORANGE"],
    )


def _graphe_calibration(courbe: pd.DataFrame, lisse: Optional[pd.DataFrame], titre: str):
    lim = max(float(max(courbe["pd_moyenne"].max(), courbe["taux_observe"].max())) * 1.12, 1e-4)
    ech = alt.Scale(domain=[0, lim], nice=False)

    diag = alt.Chart(pd.DataFrame({"x": [0, lim], "y": [0, lim]})).mark_line(
        strokeDash=[5, 4], color=GRIS, size=1.5
    ).encode(x=alt.X("x:Q", scale=ech), y=alt.Y("y:Q", scale=ech))

    base = alt.Chart(courbe)
    barres = base.mark_rule(color=ROUGE, opacity=0.55).encode(
        x=alt.X("pd_moyenne:Q", scale=ech), y=alt.Y("ic_bas:Q", scale=ech), y2="ic_haut:Q"
    )
    points = base.mark_circle(color=ROUGE, size=70).encode(
        x=alt.X("pd_moyenne:Q", scale=ech,
                axis=alt.Axis(title="Probabilité de défaut prédite", format="%")),
        y=alt.Y("taux_observe:Q", scale=ech,
                axis=alt.Axis(title="Taux de défaut observé à 12 mois", format="%")),
        tooltip=[
            alt.Tooltip("n:Q", title="Effectif", format=","),
            alt.Tooltip("n_defauts:Q", title="Défauts", format=","),
            alt.Tooltip("pd_moyenne:Q", title="PD prédite", format=".3%"),
            alt.Tooltip("taux_observe:Q", title="Taux observé", format=".3%"),
        ],
    )

    couches = [diag, barres, points]
    if lisse is not None and len(lisse):
        # borné au domaine tracé : le lissage s'étend jusqu'à la PD maximale,
        # bien au-delà de la zone où les points sont représentés
        lisse = lisse[lisse["pd_predite"] <= lim]
    if lisse is not None and len(lisse):
        couches.insert(1, alt.Chart(lisse).mark_line(color=BLEU, size=2.5).encode(
            x=alt.X("pd_predite:Q", scale=ech), y=alt.Y("taux_lisse:Q", scale=ech)
        ))

    return (
        alt.layer(*couches)
        .properties(height=430, title=titre)
        .configure_view(strokeWidth=0)
        .configure_axis(gridOpacity=0.25)
    )


def _graphe_brier(br: dict):
    df = pd.DataFrame({
        "composante": ["Fiabilité (calibration)", "Résolution (discrimination)", "Incertitude"],
        "valeur": [br["fiabilite_calibration"], br["resolution_discrimination"], br["incertitude"]],
        "sens": ["à minimiser", "à maximiser", "hors modèle"],
    })
    return alt.Chart(df).mark_bar(size=34).encode(
        y=alt.Y("composante:N", sort=None, title=None),
        x=alt.X("valeur:Q", title="Contribution au score de Brier"),
        color=alt.Color("sens:N",
                        scale=alt.Scale(domain=["à minimiser", "à maximiser", "hors modèle"],
                                        range=[ROUGE, BLEU, GRIS]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("valeur:Q", format=".6f")],
    ).properties(height=190)


def _style_classes(tab: pd.DataFrame, alpha_: float):
    def feu(v):
        return f"background-color:{COULEURS_FEU.get(v, '#6b7280')};color:white;font-weight:600;"

    def pval(v):
        if pd.isna(v):
            return ""
        if v < alpha_ / 5:
            return f"background-color:{COULEURS_FEU['ROUGE']};color:white;"
        if v < alpha_:
            return f"background-color:{COULEURS_FEU['ORANGE']};color:white;"
        return ""

    sty = tab.style.format({
        "pd_predite": "{:.3%}", "taux_observe": "{:.3%}",
        "ic_bas": "{:.3%}", "ic_haut": "{:.3%}",
        "dr_critique_95": "{:.3%}", "dr_critique_999": "{:.3%}",
        "p_jeffreys": "{:.4f}", "p_jeffreys_ajustee": "{:.4f}", "p_binomial": "{:.4f}",
        "n": "{:,.0f}", "n_defauts": "{:,.0f}",
    })
    if "feu" in tab.columns:
        sty = sty.map(feu, subset=["feu"])
    for c in ("p_jeffreys", "p_jeffreys_ajustee"):
        if c in tab.columns:
            sty = sty.map(pval, subset=[c])
    return sty


# --------------------------------------------------------------------------- #
# Fonction principale
# --------------------------------------------------------------------------- #

def afficher_calibration(
    df,
    y: str,
    p: Union[str, Sequence[str]],
    classe: Optional[str] = None,
    emprunteur: Optional[str] = None,
    segment: Optional[str] = None,
    *,
    rho: float = 0.08,
    n_bins: int = 20,
    n_boot: int = 300,
    alpha: float = 0.05,
    lissage: float = 0.6,
    titre: Optional[str] = "Contrôle de la calibration du score d'octroi",
    panneau_parametres: bool = True,
    cle: str = "calib",
) -> None:
    """Affiche l'intégralité du protocole de calibration dans la page Streamlit.

    Paramètres obligatoires
    -----------------------
    df          DataFrame polars ou pandas.
    y           Nom de la colonne de défaut observé, binaire 0/1.
    p           Nom de la colonne de PD prédite, dans [0, 1].
                Une liste de noms active le mode comparaison de modèles.

    Paramètres facultatifs
    ----------------------
    classe      Colonne de classe de risque -> backtesting réglementaire.
    emprunteur  Colonne d'identifiant client -> bootstrap par grappes.
    segment     Colonne de segmentation -> filtre interactif (sous-périmètre,
                échantillon stock/production…).
    rho         Corrélation d'actifs des seuils de Vasicek.
    n_bins      Nombre de groupes de la courbe de calibration.
    n_boot      Réplications bootstrap (0 pour désactiver).
    alpha       Seuil de significativité.
    panneau_parametres  Affiche les curseurs de réglage dans la barre latérale.
    cle         Préfixe des clés de widgets, à changer si la fonction est
                appelée plusieurs fois dans la même page.
    """
    pdf = _vers_pandas(df)
    colonnes_p = _liste(p)

    manquantes = [c for c in [y, *colonnes_p, classe, emprunteur, segment] if c and c not in pdf.columns]
    if manquantes:
        st.error("Colonnes absentes du DataFrame : " + ", ".join(f"« {c} »" for c in manquantes))
        st.caption("Colonnes disponibles : " + ", ".join(map(str, pdf.columns)))
        return

    if titre:
        st.title(titre)
        st.caption(
            "Protocole en six étapes — diagnostic graphique, régression de calibration, "
            "tests globaux, backtesting réglementaire par classe, décomposition de Brier."
        )

    # ------------------------------------------------------------- paramétrage
    if panneau_parametres:
        with st.sidebar:
            st.header("Paramètres de calibration")
            n_bins = st.slider("Groupes de la courbe", 5, 50, n_bins, 5, key=f"{cle}_bins")
            rho = st.slider(
                "Corrélation d'actifs ρ", 0.0, 0.30, rho, 0.01, key=f"{cle}_rho",
                help="Seuils de Vasicek. 0 revient au test binomial sous indépendance. "
                     "La formule CRR donne typiquement 0,04 à 0,12 sur du retail / professionnels.",
            )
            alpha = st.select_slider(
                "Seuil de significativité", [0.01, 0.05, 0.10], value=alpha, key=f"{cle}_alpha"
            )
            if emprunteur:
                n_boot = st.slider(
                    "Réplications bootstrap", 0, 2000, n_boot, 100, key=f"{cle}_boot",
                    help=f"Bootstrap par grappes sur « {emprunteur} ». 0 pour désactiver.",
                )
            lissage = st.slider("Fenêtre du lissage local", 0.2, 1.0, lissage, 0.1, key=f"{cle}_frac")

    # ------------------------------------------------- sélection modèle/segment
    barre = st.columns([2, 2, 3])
    col_p = colonnes_p[0]
    if len(colonnes_p) > 1:
        col_p = barre[0].selectbox("Modèle", colonnes_p, key=f"{cle}_modele")

    travail = pdf
    if segment:
        valeurs = ["Tout"] + sorted(pdf[segment].dropna().astype(str).unique().tolist())
        choix = barre[1].selectbox(segment, valeurs, key=f"{cle}_segment")
        if choix != "Tout":
            travail = travail[travail[segment].astype(str) == choix]

    besoins = [c for c in {y, col_p, classe, emprunteur} if c]
    travail = travail[besoins].dropna()

    # -------------------------------------------------------------- validation
    if travail.empty:
        st.error("Aucune observation après filtrage et suppression des valeurs manquantes.")
        return

    yv = travail[y].to_numpy(dtype=float)
    pv = travail[col_p].to_numpy(dtype=float)

    if not np.all(np.isin(yv, (0.0, 1.0))):
        apercu = ", ".join(f"{v:g}" for v in sorted(pd.unique(yv))[:6])
        st.error(f"La colonne « {y} » doit être binaire (0/1). "
                 f"Valeurs trouvées : {apercu}…")
        return
    if pv.min() < 0 or pv.max() > 1:
        st.error(f"La colonne « {col_p} » doit contenir des probabilités dans [0, 1] "
                 f"(min {pv.min():.4g}, max {pv.max():.4g}).")
        return
    if yv.sum() == 0:
        st.error("Aucun défaut observé sur le périmètre sélectionné.")
        return

    res = _globaux(yv, pv, n_bins)
    reg, cox = res["reg"], res["cox"]

    ic_pente = tuple(cox["pente_ic95"])
    source_ic = "IC 95 % sous indépendance"
    if emprunteur and n_boot > 0:
        with st.spinner("Bootstrap par grappes…"):
            bs = _bootstrap_pente(yv, pv, travail[emprunteur].to_numpy(), n_boot)
        if np.isfinite(bs.get("ic_bas", np.nan)):
            ic_pente = (bs["ic_bas"], bs["ic_haut"])
            source_ic = f"IC 95 % bootstrap par emprunteur ({n_boot} réplications)"

    # ---------------------------------------------------------------- synthèse
    st.subheader("Synthèse")
    k = st.columns(5)
    k[0].metric("Observations", f"{len(yv):,}".replace(",", " "))
    k[1].metric("Taux de défaut observé", f"{yv.mean():.2%}")
    k[2].metric("PD moyenne prédite", f"{pv.mean():.2%}",
                delta=f"{pv.mean() - yv.mean():+.2%}", delta_color="off")
    k[3].metric("ECE", f"{res['ece']['ECE']:.4f}")
    k[4].metric("Pente de calibration", f"{cox['pente']:.3f}")

    v_pente, c_pente = _verdict_pente(cox["pente"], ic_pente)
    v_alpha, c_alpha = _verdict_intercept(reg["intercept_large"])
    rejet_sp = res["spiegelhalter"]["p_value"] < alpha
    st.markdown(
        f"{_badge(v_alpha, c_alpha)} &nbsp; {_badge(v_pente, c_pente)} &nbsp; "
        + _badge(
            "Spiegelhalter : " + ("rejet de la calibration" if rejet_sp else "pas de rejet"),
            COULEURS_FEU["ROUGE"] if rejet_sp else COULEURS_FEU["VERT"],
        ),
        unsafe_allow_html=True,
    )
    st.divider()

    # ----------------------------------------------------------------- onglets
    noms = ["1. Courbe de calibration", "2. Pente et intercept", "3. Tests globaux",
            "4-5. Backtesting par classe", "6. Décomposition de Brier", "Export"]
    if len(colonnes_p) > 1:
        noms.insert(5, "Comparaison des modèles")
    ong = st.tabs(noms)
    idx = {n: t for n, t in zip(noms, ong)}

    # --- 1
    with idx["1. Courbe de calibration"]:
        lisse = _loess(yv, pv, lissage) if len(yv) <= 200_000 else None
        st.altair_chart(_graphe_calibration(res["courbe"], lisse, "Diagramme de fiabilité"),
                        **LARGE)
        a, b = st.columns(2)
        a.metric("ECE", f"{res['ece']['ECE']:.5f}")
        b.metric("MCE", f"{res['ece']['MCE']:.5f}")
        st.caption(
            "Points rouges : groupes d'effectif égal avec intervalle de Wilson à 95 %. "
            "Ligne bleue : lissage local estimé sur l'échelle logit. Un écart concentré sur "
            "les scores les plus risqués, là où se situe le seuil d'acceptation, pèse "
            "davantage qu'un écart uniforme."
        )
        with st.expander("Table des groupes"):
            st.dataframe(
                res["courbe"].style.format(
                    {c: "{:.3%}" for c in ("pd_moyenne", "taux_observe", "ic_bas", "ic_haut")}),
                **LARGE,
            )

    # --- 2
    with idx["2. Pente et intercept"]:
        st.markdown(r"Régression de calibration : $\;\text{logit}\,\mathbb{P}(Y=1)"
                    r" = \alpha + \beta\,\text{logit}(\hat{p})$")
        g, d = st.columns(2)
        with g:
            st.metric("Intercept in-the-large (α)", f"{reg['intercept_large']:+.4f}")
            st.markdown(_badge(v_alpha, c_alpha), unsafe_allow_html=True)
            st.caption(
                "Estimé en contraignant β = 1. Négatif : le score surestime le risque en "
                "niveau ; positif : il le sous-estime. Corrigeable par un simple recalage "
                "de l'ordonnée à l'origine de la grille, sans toucher au classement."
            )
        with d:
            st.metric("Pente de calibration (β)", f"{cox['pente']:.4f}")
            st.markdown(_badge(v_pente, c_pente), unsafe_allow_html=True)
            st.caption(
                f"{source_ic} : [{ic_pente[0]:.3f} ; {ic_pente[1]:.3f}]. "
                "β < 1 : prédictions trop dispersées (surajustement). β > 1 : prédictions "
                "trop plates, le risque des mauvaises classes est sous-estimé."
            )

        st.markdown(r"**Test de Cox** — $H_0 : (\alpha, \beta) = (0, 1)$, 2 degrés de liberté")
        st.dataframe(pd.DataFrame([{
            "Statistique LR": round(cox["statistique_LR"], 3),
            "Degrés de liberté": 2,
            "p-valeur": f"{cox['p_value']:.3e}",
            "Conclusion": "Rejet" if cox["p_value"] < alpha else "Pas de rejet",
        }]), hide_index=True, **LARGE)

        if emprunteur and n_boot > 0:
            st.info(
                f"L'intervalle de la pente est obtenu par bootstrap par grappes sur "
                f"« {emprunteur} ». Les observations empilées à deux dates d'observation ne "
                "sont pas indépendantes : les intervalles usuels sous-estimeraient la variance."
            )
        elif not emprunteur:
            st.warning(
                "Aucun identifiant emprunteur fourni : les intervalles supposent "
                "l'indépendance des observations, hypothèse fausse avec un empilement à deux "
                "dates et des fenêtres de performance chevauchantes. Passez l'argument "
                "`emprunteur=` pour activer le bootstrap par grappes."
            )

    # --- 3
    with idx["3. Tests globaux"]:
        sp, hl = res["spiegelhalter"], res["hosmer_lemeshow"]
        st.dataframe(pd.DataFrame([
            {"Test": "Spiegelhalter (1986)", "Statistique": f"Z = {sp['Z']:+.3f}",
             "p-valeur": f"{sp['p_value']:.3e}",
             "Conclusion": "Rejet" if sp["p_value"] < alpha else "Pas de rejet",
             "Rôle": "Test global de référence — aucun découpage arbitraire"},
            {"Test": "Cox (1958)", "Statistique": f"LR = {cox['statistique_LR']:.2f}",
             "p-valeur": f"{cox['p_value']:.3e}",
             "Conclusion": "Rejet" if cox["p_value"] < alpha else "Pas de rejet",
             "Rôle": "Test conjoint sur (α, β) — indique le sens de l'écart"},
            {"Test": "Hosmer-Lemeshow (1980)",
             "Statistique": f"χ² = {hl['chi2']:.2f} ({hl['ddl']} ddl)",
             "p-valeur": f"{hl['p_value']:.3e}",
             "Conclusion": "Rejet" if hl["p_value"] < alpha else "Pas de rejet",
             "Rôle": "Reporté par convention — à ne pas utiliser comme critère"},
        ]), hide_index=True, **LARGE)
        st.caption(
            "Hosmer-Lemeshow dépend du nombre de groupes retenu (Bertolini et al., 2000) et "
            "rejette pour des écarts négligeables dès que l'effectif est grand (Nattino, "
            "Pennell & Lemeshow, 2020). Il est affiché pour mémoire ; la conclusion repose "
            "sur Spiegelhalter et sur la régression de calibration."
        )
        st.metric("Score de Brier", f"{sp['brier']:.6f}")

    # --- 4/5
    with idx["4-5. Backtesting par classe"]:
        if not classe:
            st.info(
                "Passez l'argument `classe=\"nom_de_colonne\"` pour activer le backtesting "
                "réglementaire par classe de risque (test de Jeffreys et feux tricolores)."
            )
        else:
            tab = _par_classe(yv, pv, travail[classe].astype(str).to_numpy(), rho)
            n_signal = int((tab["p_jeffreys_ajustee"] < alpha).sum())
            n_vert = int((tab["feu"] == "VERT").sum())

            m = st.columns(3)
            m[0].metric("Classes testées", len(tab))
            m[1].metric("Signalées par Jeffreys", n_signal,
                        help=f"p-valeur corrigée de Holm < {alpha}")
            m[2].metric("Classes en zone verte", f"{n_vert} / {len(tab)}",
                        help=f"Seuils de Vasicek, ρ = {rho:.2f}")

            cols = ["classe", "n", "n_defauts", "pd_predite", "taux_observe", "ic_bas",
                    "ic_haut", "p_jeffreys", "p_jeffreys_ajustee", "dr_critique_95", "feu"]
            st.dataframe(_style_classes(tab[cols], alpha),
                         hide_index=True, **LARGE)
            st.caption(
                "Test de Jeffreys : a priori Beta(½, ½), a posteriori Beta(D + ½, N − D + ½), "
                "p-valeur = fonction de répartition évaluée en la PD prédite. Hypothèse nulle "
                "unilatérale — une p-valeur faible signale une sous-estimation du risque. "
                "Correction de multiplicité par la procédure de Holm. Feux tricolores calculés "
                "sous modèle à un facteur de Vasicek."
            )
            if n_signal > 0 and n_vert == len(tab):
                st.warning(
                    f"{n_signal} classe(s) sont signalées par le test de Jeffreys alors que "
                    f"toutes restent en zone verte sous ρ = {rho:.2f}. Ce n'est pas une "
                    "incohérence : les seuils corrigés de la corrélation sont très larges. Le "
                    "Comité de Bâle (2005) note que ces tests ne permettent de détecter que "
                    "les cas de mauvaise calibration relativement évidents. Faites varier ρ "
                    "pour mesurer la sensibilité du verdict."
                )

            st.altair_chart(
                alt.Chart(tab)
                .transform_fold(["pd_predite", "taux_observe"], as_=["Série", "Valeur"])
                .mark_bar()
                .encode(
                    x=alt.X("classe:N", sort=None, title="Classe de risque"),
                    y=alt.Y("Valeur:Q", axis=alt.Axis(format="%", title="Taux")),
                    color=alt.Color("Série:N",
                                    scale=alt.Scale(domain=["pd_predite", "taux_observe"],
                                                    range=[BLEU, ROUGE]),
                                    legend=alt.Legend(title=None, orient="bottom")),
                    xOffset="Série:N",
                    tooltip=[alt.Tooltip("Valeur:Q", format=".3%")],
                )
                .properties(height=320,
                            title="PD prédite et taux de défaut observé par classe"),
                **LARGE,
            )

    # --- 6
    with idx["6. Décomposition de Brier"]:
        br = res["brier"]
        c = st.columns(4)
        c[0].metric("Brier", f"{br['brier']:.6f}")
        c[1].metric("Fiabilité", f"{br['fiabilite_calibration']:.6f}")
        c[2].metric("Résolution", f"{br['resolution_discrimination']:.6f}")
        c[3].metric("Incertitude", f"{br['incertitude']:.6f}")
        st.altair_chart(_graphe_brier(br), **LARGE)
        st.caption(
            "Décomposition de Murphy (1973) : Brier = fiabilité − résolution + incertitude. "
            "La fiabilité est la composante de calibration, la résolution celle de "
            "discrimination. Elle permet de dire si l'écart entre deux modèles vient d'un "
            "déficit de pouvoir discriminant ou d'un simple décalage de niveau, ce que le "
            "Gini ne distingue pas."
        )

    # --- comparaison multi-modèles
    if len(colonnes_p) > 1:
        with idx["Comparaison des modèles"]:
            lignes = []
            for nom in colonnes_p:
                base = pdf[[y, nom] + ([segment] if segment else [])].dropna()
                if segment and choix != "Tout":
                    base = base[base[segment].astype(str) == choix]
                yy = base[y].to_numpy(dtype=float)
                pp = base[nom].to_numpy(dtype=float)
                if len(yy) == 0 or yy.sum() == 0:
                    continue
                r = _globaux(yy, pp, n_bins)
                lignes.append({
                    "Modèle": nom,
                    "PD moyenne": pp.mean(),
                    "ECE": r["ece"]["ECE"],
                    "MCE": r["ece"]["MCE"],
                    "Intercept α": r["reg"]["intercept_large"],
                    "Pente β": r["cox"]["pente"],
                    "Spiegelhalter Z": r["spiegelhalter"]["Z"],
                    "Brier": r["brier"]["brier"],
                    "Fiabilité": r["brier"]["fiabilite_calibration"],
                    "Résolution": r["brier"]["resolution_discrimination"],
                })
            comp = pd.DataFrame(lignes)
            st.dataframe(
                comp.style.format({
                    "PD moyenne": "{:.3%}", "ECE": "{:.5f}", "MCE": "{:.5f}",
                    "Intercept α": "{:+.4f}", "Pente β": "{:.4f}",
                    "Spiegelhalter Z": "{:+.2f}", "Brier": "{:.6f}",
                    "Fiabilité": "{:.6f}", "Résolution": "{:.6f}",
                }),
                hide_index=True, **LARGE,
            )
            st.caption(
                "Lecture recommandée : un modèle parcimonieux qui perd peu en résolution tout "
                "en gagnant en fiabilité est préférable à un modèle plus complexe dont "
                "l'avantage de Brier tient à un simple décalage de niveau."
            )

    # --- export
    with idx["Export"]:
        synthese = pd.DataFrame([
            {"indicateur": "Modèle", "valeur": col_p},
            {"indicateur": "Observations", "valeur": len(yv)},
            {"indicateur": "Taux de défaut observé", "valeur": float(yv.mean())},
            {"indicateur": "PD moyenne prédite", "valeur": float(pv.mean())},
            {"indicateur": "ECE", "valeur": res["ece"]["ECE"]},
            {"indicateur": "MCE", "valeur": res["ece"]["MCE"]},
            {"indicateur": "Intercept in-the-large", "valeur": reg["intercept_large"]},
            {"indicateur": "Pente de calibration", "valeur": cox["pente"]},
            {"indicateur": "Pente — IC bas", "valeur": ic_pente[0]},
            {"indicateur": "Pente — IC haut", "valeur": ic_pente[1]},
            {"indicateur": "Cox — LR", "valeur": cox["statistique_LR"]},
            {"indicateur": "Cox — p-valeur", "valeur": cox["p_value"]},
            {"indicateur": "Spiegelhalter — Z", "valeur": res["spiegelhalter"]["Z"]},
            {"indicateur": "Spiegelhalter — p-valeur", "valeur": res["spiegelhalter"]["p_value"]},
            {"indicateur": "Hosmer-Lemeshow — p-valeur", "valeur": res["hosmer_lemeshow"]["p_value"]},
            {"indicateur": "Brier — fiabilité", "valeur": res["brier"]["fiabilite_calibration"]},
            {"indicateur": "Brier — résolution", "valeur": res["brier"]["resolution_discrimination"]},
        ])
        # colonne homogène : Arrow n'accepte pas un mélange texte / nombres
        synthese["valeur"] = synthese["valeur"].map(
            lambda v: v if isinstance(v, str) else f"{v:.6g}"
        )
        st.dataframe(synthese, hide_index=True, **LARGE)

        e = st.columns(3)
        e[0].download_button("Synthèse (CSV)",
                             synthese.to_csv(index=False, sep=";").encode("utf-8-sig"),
                             "calibration_synthese.csv", "text/csv",
                             key=f"{cle}_dl1", **LARGE)
        e[1].download_button("Courbe de calibration (CSV)",
                             res["courbe"].to_csv(index=False, sep=";").encode("utf-8-sig"),
                             "calibration_courbe.csv", "text/csv",
                             key=f"{cle}_dl2", **LARGE)
        if classe:
            tab = _par_classe(yv, pv, travail[classe].astype(str).to_numpy(), rho)
            e[2].download_button("Tableau par classe (CSV)",
                                 tab.to_csv(index=False, sep=";").encode("utf-8-sig"),
                                 "calibration_par_classe.csv", "text/csv",
                                 key=f"{cle}_dl3", **LARGE)

        st.markdown("**Phrase de résultat, prête à coller dans la section 5.6**")
        conforme = ic_pente[0] <= 1.0 <= ic_pente[1]
        phrase = (
            f"Sur les {len(yv)} observations retenues, l'écart de calibration espéré s'établit "
            f"à {res['ece']['ECE']:.4f}. L'intercept in-the-large vaut "
            f"{reg['intercept_large']:+.3f}, ce qui indique que le score "
            f"{'surestime' if reg['intercept_large'] < 0 else 'sous-estime'} le risque en "
            f"niveau. La pente de calibration vaut {cox['pente']:.3f} "
            f"(IC 95 % : [{ic_pente[0]:.3f} ; {ic_pente[1]:.3f}]), "
            + ("compatible avec la valeur 1 attendue sous calibration. " if conforme else
               f"significativement différente de 1 : les prédictions sont "
               f"{'trop dispersées' if cox['pente'] < 1 else 'trop plates'}. ")
            + f"Le test de Spiegelhalter conclut à "
            f"{'un rejet' if rejet_sp else 'une absence de rejet'} de l'hypothèse de "
            f"calibration (Z = {res['spiegelhalter']['Z']:+.2f}, "
            f"p = {res['spiegelhalter']['p_value']:.2e})."
        )
        st.code(phrase, language=None)
