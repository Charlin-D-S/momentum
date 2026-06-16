"""Page 2 — Analyse par segment."""
from __future__ import annotations

import polars as pl
import streamlit as st

from components.charts import (
    chart_calibration_quantile, chart_default_rate_by_score,
)
from components.filters import render_filters, render_reset_button
from components.profile_cards import render_boundary_section
from utils.config import col_label, get_config
from utils.data_loader import (
    apply_filters, get_active_filter_vars,
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
# Config + chargement
# ---------------------------------------------------------------------------
cfg = get_config()
sc = load_scorecard()
df = load_enriched_dataset()
variables = get_scorecard_variables(sc)
filter_vars = get_active_filter_vars(sc)

id_col = cfg.columns.id
target_col = cfg.columns.target
has_target = target_col is not None and target_col in df.columns

# ---------------------------------------------------------------------------
# Filtres
# ---------------------------------------------------------------------------
st.markdown("### Filtres")
filters = render_filters(df, filter_vars)
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
    taux_def = segment[target_col].mean()
    n_def = int(segment[target_col].sum())
    c5, c6 = st.columns(2)
    c5.metric("Défauts observés", f"{n_def:,}".replace(",", " "))
    c6.metric("Taux de défaut observé", f"{taux_def:.2%}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
st.markdown("### Diagnostics")

if not has_target:
    st.info(f"Colonne cible `{target_col}` absente — calibration et taux de défaut désactivés.")
else:
    # ── Courbe de calibration ────────────────────────────────────────────────
    with st.expander("Courbe de calibration (par quantile)", expanded=False):
        n_bins_calib = st.slider(
            "Nombre de quantiles",
            min_value=5, max_value=50,
            value=cfg.display.default_n_bins_calibration,
            step=1, key="n_bins_calib",
        )
        fig_cal, table_cal = chart_calibration_quantile(
            segment, n_bins=n_bins_calib, target_col=target_col,
        )
        if not table_cal.is_empty():
            st.plotly_chart(fig_cal, width="stretch")

            # ── Tableau lié : stats des bins ─────────────────────────────
            st.markdown("**Statistiques des bins de calibration**")
            st.dataframe(
                table_cal
                .select([
                    pl.col("_bin").alias("Quantile"),
                    pl.col("proba_moyenne").round(4).alias("Proba prédite moy."),
                    pl.col("taux_defaut_obs").round(4).alias("Taux défaut observé"),
                    pl.col("effectif").alias("Effectif"),
                    pl.col("n_defauts").alias("Défauts"),
                    # Écart calibration (observé − prédit)
                    (pl.col("taux_defaut_obs") - pl.col("proba_moyenne"))
                        .round(4).alias("Écart (obs − prédit)"),
                ])
                .to_pandas(),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("Données insuffisantes pour la calibration.")

    # ── Taux de défaut par score ─────────────────────────────────────────────
    with st.expander("Taux de défaut par score", expanded=False):
        n_bins_def = st.slider(
            "Nombre de tranches",
            min_value=5, max_value=20,
            value=cfg.display.default_n_bins_default_rate,
            step=1, key="n_bins_def",
        )
        fig_def = chart_default_rate_by_score(
            segment, n_bins=n_bins_def, target_col=target_col,
        )
        st.plotly_chart(fig_def, width="stretch")

st.markdown("---")

# ---------------------------------------------------------------------------
# Zonage décisionnel
# ---------------------------------------------------------------------------
st.markdown("### Zonage décisionnel")
st.caption(
    "Définir un ou deux seuils en probabilité de défaut. "
    "La conversion en points 1000 est calculée sur le segment courant."
)

mode = st.radio(
    "Mode",
    options=["Bicolore (vert / rouge)", "Tricolore (vert / orange / rouge)"],
    horizontal=True, key="mode_zonage",
)
tricolore = mode.startswith("Tricolore")

c1, c2 = st.columns(2)
with c1:
    seuil1_proba = st.number_input(
        "Seuil 1 (proba défaut)",
        min_value=0.0, max_value=1.0,
        value=cfg.thresholds.default_seuil1,
        step=0.01, format="%.4f",
    )
with c2:
    seuil2_proba = None
    if tricolore:
        seuil2_proba = st.number_input(
            "Seuil 2 (proba défaut)",
            min_value=0.0, max_value=1.0,
            value=cfg.thresholds.default_seuil2,
            step=0.01, format="%.4f",
        )

if tricolore and seuil2_proba is not None and seuil1_proba >= seuil2_proba:
    st.warning("Seuil 1 doit être strictement inférieur au seuil 2.")

# Conversion proba → points sur le segment courant
n_neighbors = cfg.display.proba_to_points_neighbors
seuil1_pts = proba_to_points(seuil1_proba, segment, n_neighbors=n_neighbors)
seuil2_pts = (
    proba_to_points(seuil2_proba, segment, n_neighbors=n_neighbors)
    if seuil2_proba is not None else None
)

# Info conversions
info_txt = f"**Seuil 1 ≈ {seuil1_pts} pts** (proba {seuil1_proba:.2%})"
if seuil2_pts is not None:
    info_txt += f"  ·  **Seuil 2 ≈ {seuil2_pts} pts** (proba {seuil2_proba:.2%})"
st.info(info_txt)


# ── Blocs décisionnels ──────────────────────────────────────────────────────
def _stats(zone_df: pl.DataFrame) -> tuple[int, int, float]:
    eff = zone_df.height
    if has_target and eff > 0:
        n_def = int(zone_df[target_col].sum())
        return eff, n_def, n_def / eff
    return eff, 0, 0.0


def _block_html(label: str, color: str, pct_pop: float,
                n_seg: int, n_def: int, taux: float) -> str:
    def_line = f"défauts : {n_def} — taux : {taux:.2%}" if has_target else "défauts : —"
    n_str = f"{n_seg:,}".replace(",", "\u202f")
    return (
        f'<div class="decision-block" style="background:{color};">'        f'<div class="label">{label}</div>'        f'<div class="value">{n_str}</div>'        f'<div class="sub">{pct_pop:.1%} du segment</div>'        f'<div class="sub">{def_line}</div>'        f'</div>'    )

total = segment.height

if not tricolore:
    vert = segment.filter(pl.col("score_proba") <= seuil1_proba)
    rouge = segment.filter(pl.col("score_proba") > seuil1_proba)
    e_v, d_v, t_v = _stats(vert)
    e_r, d_r, t_r = _stats(rouge)
    cols = st.columns(2)
    cols[0].markdown(_block_html(
        f"VERT  ≤ {seuil1_proba:.2%}", DECISION_GREEN, e_v / total, e_v, d_v, t_v,
    ), unsafe_allow_html=True)
    cols[1].markdown(_block_html(
        f"ROUGE  > {seuil1_proba:.2%}", DECISION_RED, e_r / total, e_r, d_r, t_r,
    ), unsafe_allow_html=True)
else:
    s1, s2 = sorted([seuil1_proba, seuil2_proba])
    vert   = segment.filter(pl.col("score_proba") <= s1)
    orange = segment.filter((pl.col("score_proba") > s1) & (pl.col("score_proba") <= s2))
    rouge  = segment.filter(pl.col("score_proba") > s2)
    e_v, d_v, t_v = _stats(vert)
    e_o, d_o, t_o = _stats(orange)
    e_r, d_r, t_r = _stats(rouge)
    cols = st.columns(3)
    cols[0].markdown(_block_html(
        f"VERT  ≤ {s1:.2%}", DECISION_GREEN, e_v / total, e_v, d_v, t_v,
    ), unsafe_allow_html=True)
    cols[1].markdown(_block_html(
        f"ORANGE  {s1:.2%} – {s2:.2%}", DECISION_ORANGE, e_o / total, e_o, d_o, t_o,
    ), unsafe_allow_html=True)
    cols[2].markdown(_block_html(
        f"ROUGE  > {s2:.2%}", DECISION_RED, e_r / total, e_r, d_r, t_r,
    ), unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Profils à la frontière — un bloc par seuil actif
# ---------------------------------------------------------------------------
n_profiles = st.slider(
    "Nombre de profils par seuil",
    min_value=3, max_value=20,
    value=cfg.display.n_boundary_profiles,
    step=1,
)

render_boundary_section(
    segment_df=segment,
    threshold_points=seuil1_pts,
    variables=variables,
    id_col=id_col,
    n=n_profiles,
    title=f"Profils à la frontière — Seuil 1 ({seuil1_pts} pts · {seuil1_proba:.2%})",
)

if seuil2_pts is not None:
    st.markdown(" ")
    render_boundary_section(
        segment_df=segment,
        threshold_points=seuil2_pts,
        variables=variables,
        id_col=id_col,
        n=n_profiles,
        title=f"Profils à la frontière — Seuil 2 ({seuil2_pts} pts · {seuil2_proba:.2%})",
    )
