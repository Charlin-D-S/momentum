"""
app_demo_calibration.py
=======================

Exemple minimal d'utilisation, à copier dans votre application Streamlit.

    streamlit run app_demo_calibration.py

Dans votre code réel, remplacez `_donnees_demo()` par la lecture de vos scores :

    df = pl.read_parquet("scores_production_2025.parquet")
"""

import numpy as np
import polars as pl
import streamlit as st

from calibration_streamlit import afficher_calibration

st.set_page_config(page_title="Calibration du score d'octroi", layout="wide")


@st.cache_data
def _donnees_demo(n_emp: int = 4000) -> pl.DataFrame:
    """Jeu simulé : 4 000 emprunteurs observés à 2 dates, deux modèles."""
    rng = np.random.default_rng(42)
    emprunteur = np.repeat(np.arange(n_emp), 2)
    effet = rng.normal(0, 0.6, n_emp)[emprunteur]
    score = rng.normal(0, 1, 2 * n_emp) + effet
    pd_vraie = 1 / (1 + np.exp(-(-4.2 + score)))
    y = rng.binomial(1, pd_vraie)
    return pl.DataFrame(
        {
            "id_client": emprunteur,
            "defaut_12m": y,
            "grille_logistique": pd_vraie,
            "challenger_xgboost": 1 / (1 + np.exp(-(-4.6 + 0.8 * score))),
            "classe_risque": [f"C{i}" for i in np.digitize(
                pd_vraie, np.quantile(pd_vraie, np.linspace(0, 1, 9)[1:-1]))],
            "echantillon": np.where(rng.random(2 * n_emp) < 0.5,
                                    "Stock 2024", "Production 2025"),
        }
    )


df = _donnees_demo()

# ----------------------------------------------------------------------------
# L'appel : un DataFrame, les noms de colonnes. Tout le reste s'affiche.
# ----------------------------------------------------------------------------
afficher_calibration(
    df,
    y="defaut_12m",
    p=["grille_logistique", "challenger_xgboost"],   # une str suffit s'il n'y a qu'un modèle
    classe="classe_risque",
    emprunteur="id_client",
    segment="echantillon",
)
