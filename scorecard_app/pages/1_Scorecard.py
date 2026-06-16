"""Page 1 — Scorecard interactive."""
from __future__ import annotations

import polars as pl
import streamlit as st

from components.charts import chart_points_by_bin, chart_variable_importance
from utils.data_loader import load_scorecard
from utils.scorecard_engine import (
    get_scorecard_variables, scorecard_table, variable_importance,
)
from utils.theme import inject_css

st.set_page_config(page_title="Scorecard", page_icon="📊", layout="wide")
st.markdown(inject_css(), unsafe_allow_html=True)

st.markdown("# Scorecard")
st.caption("Grille de score additive : pour chaque variable, contribution en points par bin.")

# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
try:
    sc = load_scorecard()
except FileNotFoundError:
    st.error("Scorecard introuvable.")
    st.stop()

variables = get_scorecard_variables(sc)
sc_view = scorecard_table(sc)
importance = variable_importance(sc)

# ---------------------------------------------------------------------------
# Filtres haut de page
# ---------------------------------------------------------------------------
c1, c2 = st.columns([2, 3])
with c1:
    var_filter = st.multiselect(
        "Filtrer les variables",
        options=variables,
        default=[],
        placeholder="Toutes",
    )
with c2:
    search = st.text_input("Recherche libre (label ou variable)", value="")

displayed_vars = var_filter if var_filter else variables

# Filtrage du tableau
filtered_sc = sc_view.filter(pl.col("Variables").is_in(displayed_vars))
if search:
    filtered_sc = filtered_sc.filter(
        pl.col("Label").str.contains(search, literal=False)
        | pl.col("Variables").str.contains(search, literal=False)
    )

# ---------------------------------------------------------------------------
# Tableau scorecard
# ---------------------------------------------------------------------------
st.markdown("### Grille de score")
st.dataframe(
    filtered_sc.to_pandas(),
    width='stretch',
    hide_index=True,
    height=min(420, 60 + 36 * filtered_sc.height),
    column_config={
        "Variables": st.column_config.TextColumn("Variable", width="medium"),
        "Label": st.column_config.TextColumn("Bin", width="large"),
        "points_1000": st.column_config.NumberColumn("Points", format="%+d"),
        "coef": st.column_config.NumberColumn("Coefficient", format="%.4f"),
    },
)

# Export
csv_bytes = filtered_sc.write_csv().encode("utf-8")
st.download_button(
    "Exporter la scorecard filtrée (CSV)",
    data=csv_bytes,
    file_name="scorecard_filtree.csv",
    mime="text/csv",
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Contribution par variable (barres par bin)
# ---------------------------------------------------------------------------
st.markdown("### Contribution par bin")
st.caption("Vert = points positifs (réduit la proba de défaut). Rouge = négatifs.")

selected_var = st.selectbox(
    "Variable à visualiser",
    options=displayed_vars,
    index=0 if displayed_vars else None,
)

if selected_var:
    sc_var = sc_view.filter(pl.col("Variables") == selected_var)
    fig = chart_points_by_bin(sc_var, selected_var)
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# ---------------------------------------------------------------------------
# Importance globale
# ---------------------------------------------------------------------------
st.markdown("### Importance des variables")
st.caption(
    "Étendue des points (max − min) sur les bins de chaque variable. "
    "Mesure l'écart de score que la variable peut produire à elle seule."
)

importance_filtered = importance.filter(pl.col("Variables").is_in(displayed_vars))
fig_imp = chart_variable_importance(importance_filtered)
st.plotly_chart(fig_imp, width='stretch')
