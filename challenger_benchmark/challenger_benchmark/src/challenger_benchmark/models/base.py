"""Interface commune des challengers.

Chaque famille implemente la meme interface pour que `pipeline.py` reste
identique d'un modele a l'autre. La preparation des donnees propre a chaque
famille (gestion native ou non des manquants et des qualitatives) est
encapsulee dans `prepare`, qui ne mute jamais l'entree partagee.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import optuna

MISSING_TOKEN = "MISSING"


class ChallengerModel(ABC):
    name: str

    def __init__(self, num_features: list[str], cat_features: list[str], seed: int = 42):
        self.num_features = num_features
        self.cat_features = cat_features
        self.seed = seed

    @abstractmethod
    def search_space(self, trial: optuna.Trial) -> dict:
        """Espace d'hyperparametres Optuna."""

    @abstractmethod
    def build(self, params: dict):
        """Construit un estimateur non ajuste."""

    def prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Met X dans la forme attendue par la famille. Par defaut, copie nue."""
        return X

    def fit(self, estimator, X: pd.DataFrame, y: pd.Series):
        estimator.fit(self.prepare(X), y)
        return estimator

    def predict_proba(self, estimator, X: pd.DataFrame) -> np.ndarray:
        return estimator.predict_proba(self.prepare(X))[:, 1]

    @abstractmethod
    def shap_values(
        self, estimator, X_sample: pd.DataFrame
    ) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, str]]:
        """Renvoie (matrice_shap, noms_colonnes, valeurs_affichage, colonne->variable).

        - matrice_shap : (n_lignes, n_colonnes) pour la classe positive ;
        - noms_colonnes : libelles des colonnes SHAP (post-encodage si besoin) ;
        - valeurs_affichage : matrice numerique pour la couleur du beeswarm ;
        - colonne->variable : agregation des colonnes vers la variable d'origine
          (identite pour les modeles a arbres, one-hot -> variable pour la logistique).
        """
