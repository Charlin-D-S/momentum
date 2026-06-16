"""Point d'entrée Streamlit — page d'accueil."""
from __future__ import annotations

import streamlit as st

from utils.config import get_config
from utils.data_loader import load_enriched_dataset, load_scorecard
from utils.theme import BNP_GREEN, inject_css


def main() -> None:
    st.set_page_config(
        page_title="BNP Paribas — Scorecard Explorer",
        page_icon="🟢",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(inject_css(), unsafe_allow_html=True)

    st.markdown(
        f"""
        <h1 style="display:flex;align-items:center;gap:12px;">
            <span style="display:inline-block;width:12px;height:12px;
                         background:{BNP_GREEN};border-radius:2px;"></span>
            Scorecard Explorer
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.caption("RISK BCEF Architecture · Model Design")

    # Chargement (déclenche le cache à la première visite)
    try:
        sc = load_scorecard()
        df = load_enriched_dataset()
    except FileNotFoundError as e:
        st.error(f"Fichier introuvable : {e}")
        st.info(
            "Placer `dataset_predit.parquet` et `scorecard.parquet` dans le dossier `data/`."
        )
        st.stop()

    n_variables = sc.filter(sc["Label"] != "-").select("Variables").n_unique()

    c1, c2, c3 = st.columns(3)
    c1.metric("Individus scorés", f"{df.height:,}".replace(",", " "))
    c2.metric("Variables dans la scorecard", n_variables)
    c3.metric("Score moyen", f"{int(df['score_points'].mean())} pts")

    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown(
        """
        - **Scorecard** — grille de score interactive, contributions par variable, importance
        - **Segments** — analyse de performance par segment, calibration, zonage décisionnel, profils à la frontière
        """
    )


if __name__ == "__main__":
    main()
