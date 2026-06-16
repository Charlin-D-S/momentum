"""
Chargement des données avec stratégie de cache Streamlit.

Stratégie mémoire :
    - st.cache_resource pour les artefacts immuables (scorecard, dataset enrichi)
      → un seul exemplaire partagé entre toutes les sessions
    - st.cache_data pour les sous-ensembles dérivés (filtres) keyés par les valeurs
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st

from utils.scorecard_engine import scorer_enrichi, get_scorecard_variables

# ============================================================================
# ZONE À AJUSTER — chemins et schéma des données
# ============================================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_PATH = DATA_DIR / "dataset_predit.parquet"     # données prédites
SCORECARD_PATH = DATA_DIR / "scorecard.parquet"        # règles scorecard

# Colonne identifiant individu
ID_COL = "id_client"

# Colonne cible 0/1 (mettre None si pas disponible — désactive calibration)
TARGET_COL = "defaut_obs"

# Variables exposées comme filtres sur la page Segments.
# Elles doivent être dans la scorecard ; leurs bins servent de catégories.
# Si la liste est vide, on prend automatiquement toutes les variables.
FILTER_VARS: list[str] = []
# ============================================================================


@st.cache_resource(show_spinner="Chargement de la scorecard...")
def load_scorecard() -> pl.DataFrame:
    """Charge le parquet scorecard (immutable, partagé)."""
    return pl.read_parquet(SCORECARD_PATH)


@st.cache_resource(show_spinner="Chargement et scoring du portefeuille...")
def load_enriched_dataset() -> pl.DataFrame:
    """
    Charge le dataset, applique la scorecard, et matérialise le DataFrame enrichi
    avec uniquement les colonnes utiles. Mémorisé via cache_resource.
    """
    sc = load_scorecard()
    lf = pl.scan_parquet(DATASET_PATH)

    enriched_lf = scorer_enrichi(lf, sc)

    variables = get_scorecard_variables(sc)
    bin_cols = [f"_bin_{v}" for v in variables]
    pts_cols = [f"_pts_{v}" for v in variables]
    score_cols = ["score_points", "score_proba", "score_logit"]

    meta = [c for c in [ID_COL, TARGET_COL] if c is not None]

    # Récupérer aussi les valeurs brutes des variables (utiles pour debug
    # éventuel et pour l'affichage d'individus à la frontière)
    raw_vars = variables

    schema_names = set(enriched_lf.collect_schema().names())
    keep = list(dict.fromkeys(meta + raw_vars + bin_cols + pts_cols + score_cols))
    keep = [c for c in keep if c in schema_names]

    return enriched_lf.select(keep).collect()


@st.cache_data(show_spinner=False)
def get_filter_options(_df: pl.DataFrame, var: str) -> list[str]:
    """
    Modalités disponibles pour un filtre catégoriel (bins ou valeurs brutes).
    Le préfixe `_` du paramètre indique à Streamlit de ne pas le hasher.
    On hashe via le nom de variable seulement (le DataFrame est partagé).
    """
    bin_col = f"_bin_{var}"
    if bin_col in _df.columns:
        col = bin_col
    elif var in _df.columns:
        col = var
    else:
        return []

    vals = (
        _df.select(pl.col(col).cast(pl.Utf8))
           .drop_nulls()
           .unique()
           .sort(col)
           .to_series()
           .to_list()
    )
    return vals


def apply_filters(df: pl.DataFrame, filters: dict[str, list[str]]) -> pl.DataFrame:
    """
    Applique une série de filtres {var: [modalités sélectionnées]} sur le DataFrame.
    Filtre = AND entre variables, OR entre modalités d'une même variable.
    Une liste vide pour une variable signifie "pas de filtre actif".
    """
    out = df.lazy()
    for var, selected in filters.items():
        if not selected:
            continue
        bin_col = f"_bin_{var}"
        col = bin_col if bin_col in df.columns else var
        if col not in df.columns:
            continue
        out = out.filter(pl.col(col).cast(pl.Utf8).is_in(selected))
    return out.collect()


def get_active_filter_vars(sc: pl.DataFrame) -> list[str]:
    """Liste des variables exposées comme filtres."""
    all_vars = get_scorecard_variables(sc)
    if FILTER_VARS:
        return [v for v in FILTER_VARS if v in all_vars]
    return all_vars
