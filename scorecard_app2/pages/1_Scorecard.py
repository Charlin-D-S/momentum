"""Page 1 — Scorecard interactive."""
from __future__ import annotations

import polars as pl
import streamlit as st

from components.charts import chart_points_by_bin, chart_variable_importance
from utils.config import col_label, get_config
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
    st.error("Scorecard introuvable. Vérifier `scorecard_path` dans config.yaml.")
    st.stop()

cfg = get_config()
variables = get_scorecard_variables(sc)
sc_view = scorecard_table(sc)
importance = variable_importance(sc)

# ---------------------------------------------------------------------------
# Filtres haut de page
# ---------------------------------------------------------------------------
c1, c2 = st.columns([2, 3])
with c1:
    # Labels des variables dans le multiselect
    var_options = {col_label(v): v for v in variables}
    selected_labels = st.multiselect(
        "Filtrer les variables",
        options=list(var_options.keys()),
        default=[],
        placeholder="Toutes",
    )
    selected_vars = [var_options[l] for l in selected_labels] if selected_labels else variables

with c2:
    search = st.text_input("Recherche libre (label ou variable)", value="")

# ---------------------------------------------------------------------------
# Tableau scorecard — renomme la colonne Variables avec les alias
# ---------------------------------------------------------------------------
filtered_sc = sc_view.filter(pl.col("Variables").is_in(selected_vars))
if search:
    filtered_sc = filtered_sc.filter(
        pl.col("Label").str.contains(search, literal=False)
        | pl.col("Variables").str.contains(search, literal=False)
    )

# Substituer les alias dans le tableau
filtered_sc_display = filtered_sc.with_columns(
    pl.col("Variables").replace(
        {v: col_label(v) for v in variables}
    ).alias("Variables")
)

st.markdown("### Grille de score")
st.dataframe(
    filtered_sc_display.to_pandas(),
    use_container_width=True,
    hide_index=True,
    height=min(420, 60 + 36 * filtered_sc_display.height),
    column_config={
        "Variables": st.column_config.TextColumn("Variable", width="medium"),
        "Label": st.column_config.TextColumn("Bin", width="large"),
        "points_1000": st.column_config.NumberColumn("Points", format="%+d"),
        "coef": st.column_config.NumberColumn("Coefficient", format="%.4f"),
    },
)

csv_bytes = filtered_sc.write_csv().encode("utf-8")
st.download_button(
    "Exporter la scorecard filtrée (CSV)",
    data=csv_bytes,
    file_name="scorecard_filtree.csv",
    mime="text/csv",
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Contribution par variable
# ---------------------------------------------------------------------------
st.markdown("### Contribution par bin")
st.caption("Vert = points positifs (réduit la proba de défaut). Rouge = négatifs.")

# Selectbox avec alias mais valeur interne = nom de variable brut
var_label_map = {col_label(v): v for v in selected_vars}
selected_label = st.selectbox(
    "Variable à visualiser",
    options=list(var_label_map.keys()),
    index=0 if var_label_map else None,
)

if selected_label:
    selected_var = var_label_map[selected_label]
    sc_var = sc_view.filter(pl.col("Variables") == selected_var)
    fig = chart_points_by_bin(sc_var, col_label(selected_var))
    st.plotly_chart(fig, width="stretch")

st.markdown("---")

# ---------------------------------------------------------------------------
# Importance des variables
# ---------------------------------------------------------------------------
st.markdown("### Importance des variables")
st.caption(
    "Étendue des points (max − min) sur les bins de chaque variable. "
    "Mesure l'écart de score que la variable peut produire à elle seule."
)

importance_filtered = importance.filter(pl.col("Variables").is_in(selected_vars))
# Remplacer les noms bruts par les alias pour le graphique
importance_display = importance_filtered.with_columns(
    pl.col("Variables").replace({v: col_label(v) for v in variables}).alias("Variables")
)
fig_imp = chart_variable_importance(importance_display)
st.plotly_chart(fig_imp, width="stretch")
