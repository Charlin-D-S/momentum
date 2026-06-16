"""
Chargement des données avec stratégie de cache Streamlit.

Toute configuration vient de config.yaml via utils.config.get_config().
Plus aucune constante hardcodée ici.

Stratégie mémoire :
    - st.cache_resource : scorecard et dataset enrichi (un exemplaire par session)
    - st.cache_data     : options de filtres (clé = nom de variable)
"""
from __future__ import annotations

import polars as pl
import streamlit as st

from utils.config import get_config
from utils.scorecard_engine import get_scorecard_variables, scorer_enrichi


@st.cache_resource(show_spinner="Chargement de la scorecard...")
def load_scorecard() -> pl.DataFrame:
    """Charge le parquet scorecard (immutable, partagé entre sessions)."""
    cfg = get_config()
    return pl.read_parquet(cfg.data.scorecard_path)


@st.cache_resource(show_spinner="Chargement et scoring du portefeuille...")
def load_enriched_dataset() -> pl.DataFrame:
    """
    Charge le dataset brut, applique la scorecard (scoring complet),
    et matérialise un DataFrame enrichi avec uniquement les colonnes utiles.

    Colonnes conservées :
        meta      : id, target (si présente)
        raw       : valeurs brutes des variables scorecard
        _bin_{v}  : bin scorecard → filtres + cartes de profil
        _pts_{v}  : contribution en points → importance + décomposition
        scores    : score_points, score_proba, score_logit
    """
    cfg = get_config()
    sc = load_scorecard()
    lf = pl.scan_parquet(cfg.data.dataset_path)

    enriched_lf = scorer_enrichi(lf, sc)

    variables = get_scorecard_variables(sc)
    bin_cols = [f"_bin_{v}" for v in variables]
    pts_cols = [f"_pts_{v}" for v in variables]
    score_cols = ["score_points", "score_proba", "score_logit"]

    meta = [cfg.columns.id]
    if cfg.columns.target:
        meta.append(cfg.columns.target)

    schema_names = set(enriched_lf.collect_schema().names())
    keep = list(dict.fromkeys(meta + variables + bin_cols + pts_cols + score_cols))
    keep = [c for c in keep if c in schema_names]

    return enriched_lf.select(keep).collect()


@st.cache_data(show_spinner=False)
def get_filter_options(_df: pl.DataFrame, var: str) -> list[str]:
    """
    Modalités disponibles pour un filtre catégoriel.
    Préfixe `_` → Streamlit ne hashe pas le DataFrame (partagé).
    """
    bin_col = f"_bin_{var}"
    col = bin_col if bin_col in _df.columns else (var if var in _df.columns else None)
    if col is None:
        return []

    return (
        _df.select(pl.col(col).cast(pl.Utf8))
           .drop_nulls()
           .unique()
           .sort(col)
           .to_series()
           .to_list()
    )


def apply_filters(df: pl.DataFrame, filters: dict[str, list[str]]) -> pl.DataFrame:
    """
    Applique les filtres {var: [modalités]} sur le DataFrame.
    AND entre variables, OR entre modalités d'une même variable.
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
    """Variables exposées comme filtres (config ou toutes par défaut)."""
    cfg = get_config()
    all_vars = get_scorecard_variables(sc)
    if cfg.filters.vars:
        return [v for v in cfg.filters.vars if v in all_vars]
    return all_vars
