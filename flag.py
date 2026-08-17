import streamlit as st
import pandas as pd
import polars as pl
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — à adapter
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Situation actuelle — Diagnostic V/R",
    page_icon="🏦",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 { font-size: 1.4rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem; }

    /* Seuils */
    .seuil-box {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px 6px;
        color: white;
    }
    .seg-coeur  { background-color: #c0392b; }
    .seg-pro    { background-color: #2471a3; }
    .seg-er     { background-color: #6c3483; }

    /* Tableau */
    .report-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 1rem; }
    .report-table th {
        background-color: #1a1a2e;
        color: white;
        padding: 8px 10px;
        text-align: center;
        font-weight: 600;
        border: 1px solid #ddd;
    }
    .report-table td {
        padding: 7px 10px;
        border: 1px solid #e0e0e0;
        text-align: right;
        background-color: white;
    }
    .td-label    { text-align: left !important; font-weight: 600; color: #333; padding-left: 14px !important; }
    .td-seg      { text-align: left !important; font-weight: 700; color: #1a1a2e; background-color: #f0f0f5 !important; }
    .td-total    { font-weight: 700; background-color: #eaf0fb !important; }

    /* Feux */
    .feu-vert   { background-color: #d5f5e3 !important; color: #1e8449; font-weight: 700; }
    .feu-orange { background-color: #fdebd0 !important; color: #d35400; font-weight: 700; }
    .feu-rouge  { background-color: #fadbd8 !important; color: #c0392b; font-weight: 700; }
    .feu-total  { background-color: #eaf0fb !important; font-weight: 700; }

    /* KPI cards */
    .kpi-grid { display: flex; gap: 14px; flex-wrap: wrap; margin: 1rem 0; }
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 14px 20px;
        flex: 1;
        min-width: 150px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        border-top: 4px solid #1a1a2e;
    }
    .kpi-card.vert   { border-top-color: #27ae60; }
    .kpi-card.orange { border-top-color: #e67e22; }
    .kpi-card.rouge  { border-top-color: #c0392b; }
    .kpi-label { font-size: 0.72rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; margin-top: 2px; }
    .kpi-sub   { font-size: 0.78rem; color: #555; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DONNÉES — remplacer par ton LazyFrame réel
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def charger_données():
    """
    Remplacer par :
        lf = pl.scan_parquet("...")
        lf = lf.with_columns(diag_V_R(...))   # ta fonction feux tricolores
        return lf.collect().to_pandas()
    """
    données = {
        "segment":    ["Pro Cœur de cible"] * 4 + ["Pro autres"] * 4 + ["ER"] * 4 + ["Total"] * 4,
        "feu":        ["Vert", "Orange", "Rouge", "Total"] * 4,
        "nb_dossiers":[8092, 733, 1291, 10116, 12926, 2342, 11497, 26765, 10658, 2275, 2581, 15514, 31676, 5350, 15369, 52395],
        "repartition":[80.0, 7.2, 12.8, 100.0, 48.3, 8.8, 43.0, 100.0, 68.7, 14.7, 16.6, 100.0, 60.5, 10.2, 29.3, 100.0],
        "nb_defaut":  [76, 12, 127, 215, 157, 34, 1132, 1323, 108, 31, 295, 434, 341, 77, 1554, 1972],
        "tx_defaut":  [0.939, 1.637, 9.837, 2.125, 1.215, 1.452, 9.846, 4.943, 1.013, 1.363, 11.430, 2.797, 1.077, 1.439, 10.111, 3.764],
        "montant":    [262, 163, 33, 458, 446, 298, 416, 1160, 808, 472, 210, 1490, 1516, 933, 659, 3108],
        "mtn_defaut": [2.0, 0.3, 2.7, 4.9, 4.7, 1.3, 24.3, 30.3, 7.4, 2.5, 20.4, 30.4, 14.1, 4.0, 47.4, 65.5],
        "mtn_defaut_pct": [0.745, 0.160, 8.153, 1.072, 1.048, 0.437, 5.840, 2.609, 0.920, 0.524, 9.730, 2.037, 0.928, 0.433, 7.197, 2.108],
    }
    return pd.DataFrame(données)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — paramètres seuils
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    st.markdown("**Seuils Vert / Rouge (%)**")

    seuil_coeur = st.number_input("Pro Cœur de cible", value=6.5, step=0.1, format="%.1f")
    seuil_pro   = st.number_input("Pro autres",        value=3.1, step=0.1, format="%.1f")
    seuil_er    = st.number_input("ER",                value=7.0, step=0.1, format="%.1f")

    st.divider()
    filtre_montant = st.selectbox("Filtre par montant", ["Total", "> 500k€", "> 1M€"])


# ─────────────────────────────────────────────────────────────────────────────
# EN-TÊTE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🏦 Situation actuelle — Diagnostic Vert / Rouge")

# Seuils affichés
st.markdown(
    f"""
    <div style="margin-bottom: 1rem;">
        <span class="seuil-box seg-coeur">Pro Cœur de cible — Seuil V/R : {seuil_coeur}%</span>
        <span class="seuil-box seg-pro">Pro autres — Seuil V/R : {seuil_pro}%</span>
        <span class="seuil-box seg-er">ER — Seuil V/R : {seuil_er}%</span>
        <span style="margin-left:12px; font-size:0.82rem; color:#888;">
            Filtre montant : <b>{filtre_montant}</b>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

df = charger_données()


# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS — ligne de totaux
# ─────────────────────────────────────────────────────────────────────────────
totaux = df[df["segment"] == "Total"]

def kpi(feu: str) -> dict:
    row = totaux[totaux["feu"] == feu].iloc[0]
    return row

v, o, r, t = kpi("Vert"), kpi("Orange"), kpi("Rouge"), kpi("Total")

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Total dossiers</div>
        <div class="kpi-value">{t['nb_dossiers']:,}</div>
        <div class="kpi-sub">Taux défaut : {t['tx_defaut']:.3f}%</div>
    </div>
    <div class="kpi-card vert">
        <div class="kpi-label">🟢 Vert</div>
        <div class="kpi-value">{v['nb_dossiers']:,}</div>
        <div class="kpi-sub">{v['repartition']:.1f}% — TD : {v['tx_defaut']:.3f}%</div>
    </div>
    <div class="kpi-card orange">
        <div class="kpi-label">🟠 Orange</div>
        <div class="kpi-value">{o['nb_dossiers']:,}</div>
        <div class="kpi-sub">{o['repartition']:.1f}% — TD : {o['tx_defaut']:.3f}%</div>
    </div>
    <div class="kpi-card rouge">
        <div class="kpi-label">🔴 Rouge</div>
        <div class="kpi-value">{r['nb_dossiers']:,}</div>
        <div class="kpi-sub">{r['repartition']:.1f}% — TD : {r['tx_defaut']:.3f}%</div>
    </div>
    <div class="kpi-card" style="border-top-color:#8e44ad;">
        <div class="kpi-label">Montant total (m€)</div>
        <div class="kpi-value">{t['montant']:,}</div>
        <div class="kpi-sub">Mtn défaut : {t['mtn_defaut']:.1f} m€ ({t['mtn_defaut_pct']:.3f}%)</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABLEAU PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def classe_feu(feu: str) -> str:
    if feu == "Vert":   return "feu-vert"
    if feu == "Orange": return "feu-orange"
    if feu == "Rouge":  return "feu-rouge"
    return "feu-total"

def emoji_feu(feu: str) -> str:
    if feu == "Vert":   return "🟢 Vert"
    if feu == "Orange": return "🟠 Orange"
    if feu == "Rouge":  return "🔴 Rouge"
    return "⬜ Total"

def build_table(df: pd.DataFrame) -> str:
    html = """
    <table class="report-table">
    <thead>
        <tr>
            <th rowspan="2" style="width:160px;">Segment</th>
            <th rowspan="2" style="width:90px;">Feu</th>
            <th>Nb dossiers</th>
            <th>Répartition (%)</th>
            <th>Nb défaut</th>
            <th>Taux défaut (%)</th>
            <th>Montant (m€)</th>
            <th>Mtn défaut (m€)</th>
            <th>Mtn défaut (%)</th>
        </tr>
    </thead>
    <tbody>
    """

    segments = ["Pro Cœur de cible", "Pro autres", "ER", "Total"]
    feux     = ["Vert", "Orange", "Rouge", "Total"]

    for seg in segments:
        df_seg    = df[df["segment"] == seg]
        n_rows    = len(df_seg)
        first_row = True

        for feu in feux:
            row = df_seg[df_seg["feu"] == feu]
            if row.empty:
                continue
            row = row.iloc[0]
            cls = classe_feu(feu)

            seg_cell = ""
            if first_row:
                seg_cell = f'<td class="td-seg" rowspan="{n_rows}">{seg}</td>'
                first_row = False

            html += f"""
            <tr>
                {seg_cell}
                <td class="{cls} td-label">{emoji_feu(feu)}</td>
                <td class="{cls}">{row['nb_dossiers']:,}</td>
                <td class="{cls}">{row['repartition']:.1f}%</td>
                <td class="{cls}">{row['nb_defaut']:,}</td>
                <td class="{cls}">{row['tx_defaut']:.3f}%</td>
                <td class="{cls}">{row['montant']:,}</td>
                <td class="{cls}">{row['mtn_defaut']:.1f}</td>
                <td class="{cls}">{row['mtn_defaut_pct']:.3f}%</td>
            </tr>
            """

    html += "</tbody></table>"
    return html

st.markdown(build_table(df), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHIQUES
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📊 Répartition par segment")

import plotly.graph_objects as go

col1, col2 = st.columns(2)
couleurs = {"Vert": "#27ae60", "Orange": "#e67e22", "Rouge": "#c0392b"}
segments_graphe = ["Pro Cœur de cible", "Pro autres", "ER"]

with col1:
    # Répartition des dossiers
    fig1 = go.Figure()
    for feu in ["Vert", "Orange", "Rouge"]:
        vals = [
            df[(df["segment"] == s) & (df["feu"] == feu)]["nb_dossiers"].values[0]
            for s in segments_graphe
        ]
        fig1.add_trace(go.Bar(
            name=feu, x=segments_graphe, y=vals,
            marker_color=couleurs[feu], text=vals,
            textposition="inside", textfont=dict(color="white", size=11),
        ))
    fig1.update_layout(
        barmode="stack", title="Répartition des dossiers",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Taux de défaut par feu et segment
    fig2 = go.Figure()
    for feu in ["Vert", "Orange", "Rouge"]:
        vals = [
            df[(df["segment"] == s) & (df["feu"] == feu)]["tx_defaut"].values[0]
            for s in segments_graphe
        ]
        fig2.add_trace(go.Bar(
            name=feu, x=segments_graphe, y=vals,
            marker_color=couleurs[feu],
            text=[f"{v:.2f}%" for v in vals],
            textposition="outside",
        ))
    fig2.update_layout(
        barmode="group", title="Taux de défaut par feu (%)",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=40, b=20),
        yaxis=dict(ticksuffix="%"),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.caption("Données à date — généré automatiquement")
