"""Widgets de filtres dynamiques pour la page Segments."""
from __future__ import annotations

import streamlit as st
import polars as pl

from utils.config import col_label, get_config
from utils.data_loader import get_filter_options


def render_filters(
    df: pl.DataFrame,
    filter_vars: list[str],
) -> dict[str, list[str]]:
    """
    Affiche une grille de multiselect pour chaque variable de filtre.
    Le label affiché utilise l'alias si défini dans config.yaml.
    Retourne un dict {var: [modalités sélectionnées]}.
    """
    cfg = get_config()
    cols_per_row = cfg.filters.cols_per_row
    filters: dict[str, list[str]] = {}
    if not filter_vars:
        return filters

    rows = [filter_vars[i:i + cols_per_row] for i in range(0, len(filter_vars), cols_per_row)]
    for row in rows:
        cols = st.columns(cols_per_row)
        for i, var in enumerate(row):
            with cols[i]:
                options = get_filter_options(df, var)
                selected = st.multiselect(
                    label=col_label(var),      # alias si dispo
                    options=options,
                    default=[],
                    key=f"filter_{var}",
                    placeholder="Toutes",
                )
                filters[var] = selected
    return filters


def render_reset_button() -> bool:
    """Bouton de réinitialisation. Retourne True si cliqué."""
    if st.button("Réinitialiser les filtres", type="secondary"):
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and key.startswith("filter_"):
                del st.session_state[key]
        return True
    return False
