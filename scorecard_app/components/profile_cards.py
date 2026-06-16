"""
Cartes de profil des individus à la frontière d'un seuil.

Approche : individus réels du segment dont le score_points est le plus proche
du seuil cible. Pour chaque, on affiche la décomposition (variable, bin, points).
"""
from __future__ import annotations

import html

import polars as pl
import streamlit as st

from utils.scorecard_engine import decompose_individual
from utils.theme import DECISION_GREEN, DECISION_RED, TEXT_SECONDARY


def find_boundary_individuals(
    df: pl.DataFrame, threshold_points: int, n: int = 10,
) -> pl.DataFrame:
    """
    Retourne les n individus dont score_points est le plus proche du seuil.
    """
    if df.is_empty():
        return df

    return (
        df
        .with_columns(
            (pl.col("score_points") - threshold_points).abs().alias("_dist_seuil")
        )
        .sort("_dist_seuil")
        .head(n)
        .drop("_dist_seuil")
    )


def _format_points(p: int) -> str:
    if p > 0:
        cls = "pts-pos"
    elif p < 0:
        cls = "pts-neg"
    else:
        cls = "pts-zero"
    return f'<span class="{cls}">{p:+d}</span>'


def render_profile_card(row: dict, variables: list[str], id_col: str, idx: int) -> None:
    """
    Affiche une carte dépliable pour un individu.
    Le résumé contient l'id et le score, le contenu contient la décomposition.
    """
    score = int(row.get("score_points", 0))
    proba = row.get("score_proba", 0.0)
    id_val = row.get(id_col, f"#{idx}")

    décomposition = decompose_individual(row, variables)

    title = f"Profil {id_val}  ·  Score : {score} pts  ·  Proba : {proba:.2%}"
    with st.expander(title, expanded=False):
        html_rows = []
        for d in décomposition:
            html_rows.append(
                f'<div class="profile-row">'
                f'  <span class="var">{html.escape(d["variable"])}</span>'
                f'  <span class="bin">{html.escape(str(d["bin"]))}</span>'
                f'  {_format_points(d["points"])}'
                f'</div>'
            )
        # Ligne totale
        html_rows.append(
            f'<div class="profile-row" style="border-top: 2px solid #000; margin-top: 6px;">'
            f'  <span class="var" style="font-weight:600">TOTAL</span>'
            f'  <span class="bin"></span>'
            f'  <span class="pts-zero" style="font-weight:700;color:#1A1A1A">{score:+d}</span>'
            f'</div>'
        )
        st.markdown("".join(html_rows), unsafe_allow_html=True)


def render_boundary_section(
    segment_df: pl.DataFrame,
    threshold_points: int,
    variables: list[str],
    id_col: str,
    n: int = 8,
    title: str = "Profils à la frontière",
) -> None:
    """Section complète : titre + n cartes."""
    st.markdown(f"### {title}")
    st.caption(
        f"Les {n} individus du segment dont le score est le plus proche "
        f"de **{threshold_points} pts**. Cliquer pour déplier la décomposition."
    )
    candidates = find_boundary_individuals(segment_df, threshold_points, n=n)
    if candidates.is_empty():
        st.info("Aucun individu dans ce segment.")
        return

    rows = candidates.to_dicts()
    for i, row in enumerate(rows):
        render_profile_card(row, variables, id_col, idx=i)
