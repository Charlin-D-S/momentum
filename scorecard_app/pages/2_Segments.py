"""Page 2 — Analyse par segment."""
from __future__ import annotations

import polars as pl
import streamlit as st

from components.charts import (
    chart_calibration_quantile, chart_default_rate_by_score,
)
from components.filters import render_filters, render_reset_button
from components.profile_cards import render_boundary_section
from utils.data_loader import (
    ID_COL, TARGET_COL, apply_filters, get_active_filter_vars,
    load_enriched_dataset, load_scorecard,
)
from utils.scorecard_engine import get_scorecard_variables, proba_to_points
from utils.theme import (
    DECISION_GREEN, DECISION_ORANGE, DECISION_RED, inject_css,
)

st.set_page_config(page_title="Segments", page_icon="📈", layout="wide")
st.markdown(inject_css(), unsafe_allow_html=True)

st.markdown("# Analyse par segment")
st.caption("Filtrer le portefeuille, mesurer la performance, définir des seuils de décision.")

# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
sc = load_scorecard()
df = load_enriched_dataset()
variables = get_scorecard_variables(sc)
filter_vars = get_active_filter_vars(sc)
has_target = TARGET_COL is not None and TARGET_COL in df.columns

# ---------------------------------------------------------------------------
# Filtres
# ---------------------------------------------------------------------------
st.markdown("### Filtres")
filters = render_filters(df, filter_vars, cols_per_row=3)
render_reset_button()

segment = apply_filters(df, filters)

if segment.is_empty():
    st.warning("Aucun individu ne correspond aux filtres sélectionnés.")
    st.stop()

# ---------------------------------------------------------------------------
# Statistiques du segment
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Statistiques du segment")

n = segment.height
share = n / df.height
score_mean = segment["score_points"].mean()
proba_mean = segment["score_proba"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Effectif", f"{n:,}".replace(",", " "))
c2.metric("% du portefeuille", f"{share:.1%}")
c3.metric("Score moyen", f"{int(score_mean)} pts")
c4.metric("Proba moyenne", f"{proba_mean:.2%}")

if has_target:
    taux_def = segment[TARGET_COL].mean()
    n_def = int(segment[TARGET_COL].sum())
    c5, c6 = st.columns(2)
    c5.metric("Défauts observés", f"{n_def:,}".replace(",", " "))
    c6.metric("Taux de défaut observé", f"{taux_def:.2%}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Diagnostics (cliquables)
# ---------------------------------------------------------------------------
st.markdown("### Diagnostics")

if not has_target:
    st.info(f"Colonne cible `{TARGET_COL}` absente : calibration et taux de défaut désactivés.")

else:
    with st.expander("Courbe de calibration (par quantile)", expanded=False):
        n_bins_calib = st.slider(
            "Nombre de quantiles", min_value=5, max_value=50, value=10, step=1,
            key="n_bins_calib",
        )
        fig_cal, table_cal = chart_calibration_quantile(
            segment, n_bins=n_bins_calib, target_col=TARGET_COL,
        )
        if not table_cal.is_empty():
            st.plotly_chart(fig_cal, width='stretch')
            st.markdown("**Tableau lié**")
            st.dataframe(
                table_cal.select([
                    pl.col("_bin").alias("Quantile"),
                    pl.col("proba_moyenne").round(4).alias("Proba moy."),
                    pl.col("taux_defaut_obs").round(4).alias("Taux défaut obs."),
                    "effectif", "n_defauts",
                ]).to_pandas(),
                width='stretch', hide_index=True,
            )
        else:
            st.warning("Données insuffisantes pour la calibration.")

    with st.expander("Taux de défaut par score", expanded=False):
        n_bins_def = st.slider(
            "Nombre de tranches", min_value=5, max_value=20, value=10, step=1,
            key="n_bins_def",
        )
        fig_def = chart_default_rate_by_score(
            segment, n_bins=n_bins_def, target_col=TARGET_COL,
        )
        st.plotly_chart(fig_def, width='stretch')

st.markdown("---")

# ---------------------------------------------------------------------------
# Zonage décisionnel
# ---------------------------------------------------------------------------
st.markdown("### Zonage décisionnel")
st.caption(
    "Définir un ou deux seuils en probabilité de défaut. "
    "Les seuils en points 1000 sont déduits du segment courant (médiane des voisins en proba)."
)

mode = st.radio(
    "Mode", options=["Bicolore (vert / rouge)", "Tricolore (vert / orange / rouge)"],
    horizontal=True, key="mode_zonage",
)
tricolore = mode.startswith("Tricolore")

c1, c2 = st.columns(2)
with c1:
    seuil1_proba = st.number_input(
        "Seuil 1 (proba défaut)", min_value=0.0, max_value=1.0,
        value=0.05, step=0.01, format="%.4f",
    )
with c2:
    seuil2_proba = None
    if tricolore:
        seuil2_proba = st.number_input(
            "Seuil 2 (proba défaut)", min_value=0.0, max_value=1.0,
            value=0.15, step=0.01, format="%.4f",
        )

# Conversion proba → points (cherchée dans le segment courant)
seuil1_pts = proba_to_points(seuil1_proba, segment)
seuil2_pts = proba_to_points(seuil2_proba, segment) if seuil2_proba is not None else None

if tricolore and seuil2_proba is not None and seuil1_proba > seuil2_proba:
    st.warning("Le seuil 1 doit être inférieur au seuil 2 en probabilité.")

# Note : un score_points élevé = faible proba ; on classe sur la proba.
def _bucket_label(p_max: float | None, p_min: float | None) -> str:
    if p_max is None:
        return f"proba > {p_min:.2%}"
    if p_min is None:
        return f"proba ≤ {p_max:.2%}"
    return f"{p_min:.2%} < proba ≤ {p_max:.2%}"


def _block_html(label: str, color: str, sub_top: str, n_seg: int, n_def: int, taux: float) -> str:
    return (
        f'<div class="decision-block" style="background:{color};">'
        f'  <div class="label">{label}</div>'
        f'  <div class="value">{n_seg:,}</div>'
        f'  <div class="sub">{sub_top}</div>'
        f'  <div class="sub">défauts: {n_def}{" — taux: " + f"{taux:.2%}" if has_target else ""}</div>'
        f'</div>'
    ).replace(",", " ")


def _stats(zone_df: pl.DataFrame) -> tuple[int, int, float]:
    eff = zone_df.height
    if has_target and eff > 0:
        n_def = int(zone_df[TARGET_COL].sum())
        taux = n_def / eff
    else:
        n_def, taux = 0, 0.0
    return eff, n_def, taux


total = segment.height
if not tricolore:
    vert = segment.filter(pl.col("score_proba") <= seuil1_proba)
    rouge = segment.filter(pl.col("score_proba") > seuil1_proba)
    e_v, d_v, t_v = _stats(vert)
    e_r, d_r, t_r = _stats(rouge)

    cols = st.columns(2)
    cols[0].markdown(_block_html(
        f"VERT — {_bucket_label(seuil1_proba, None)}",
        DECISION_GREEN, f"{e_v / total:.1%} du segment", e_v, d_v, t_v,
    ), unsafe_allow_html=True)
    cols[1].markdown(_block_html(
        f"ROUGE — {_bucket_label(None, seuil1_proba)}",
        DECISION_RED, f"{e_r / total:.1%} du segment", e_r, d_r, t_r,
    ), unsafe_allow_html=True)

else:
    s1, s2 = sorted([seuil1_proba, seuil2_proba])
    vert = segment.filter(pl.col("score_proba") <= s1)
    orange = segment.filter((pl.col("score_proba") > s1) & (pl.col("score_proba") <= s2))
    rouge = segment.filter(pl.col("score_proba") > s2)
    e_v, d_v, t_v = _stats(vert)
    e_o, d_o, t_o = _stats(orange)
    e_r, d_r, t_r = _stats(rouge)

    cols = st.columns(3)
    cols[0].markdown(_block_html(
        f"VERT — {_bucket_label(s1, None)}",
        DECISION_GREEN, f"{e_v / total:.1%}", e_v, d_v, t_v,
    ), unsafe_allow_html=True)
    cols[1].markdown(_block_html(
        f"ORANGE — {_bucket_label(s2, s1)}",
        DECISION_ORANGE, f"{e_o / total:.1%}", e_o, d_o, t_o,
    ), unsafe_allow_html=True)
    cols[2].markdown(_block_html(
        f"ROUGE — {_bucket_label(None, s2)}",
        DECISION_RED, f"{e_r / total:.1%}", e_r, d_r, t_r,
    ), unsafe_allow_html=True)

st.markdown(" ")
st.info(
    f"Conversion proba → points 1000 (médiane des plus proches voisins dans le segment) : "
    f"**seuil 1 ≈ {seuil1_pts} pts**"
    + (f", **seuil 2 ≈ {seuil2_pts} pts**" if seuil2_pts is not None else "")
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Profils à la frontière
# ---------------------------------------------------------------------------
n_profiles = st.slider("Nombre de profils à afficher par seuil",
                       min_value=3, max_value=20, value=8, step=1)

render_boundary_section(
    segment_df=segment,
    threshold_points=seuil1_pts,
    variables=variables,
    id_col=ID_COL,
    n=n_profiles,
    title=f"Profils à la frontière du seuil 1 ({seuil1_pts} pts)",
)

if seuil2_pts is not None:
    render_boundary_section(
        segment_df=segment,
        threshold_points=seuil2_pts,
        variables=variables,
        id_col=ID_COL,
        n=n_profiles,
        title=f"Profils à la frontière du seuil 2 ({seuil2_pts} pts)",
    )
