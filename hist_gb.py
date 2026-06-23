"""Challenger HistGradientBoosting (remplace la foret aleatoire).

Manquants numeriques natifs. Les categorielles sont passees en codes entiers
(0..k-1, NaN preserve) avec declaration explicite via categorical_features, ce
qui les fait traiter comme categorielles et non ordinales par le modele, et
laisse une matrice numerique exploitable par SHAP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
import shap
from sklearn.ensemble import HistGradientBoostingClassifier

from .base import ChallengerModel
from .xgboost import _positive_class, _display_matrix


class HistGBModel(ChallengerModel):
    name = "hist_gradient_boosting"

    def search_space(self, trial: optuna.Trial) -> dict:
        return {
            "max_iter": trial.suggest_int("max_iter", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        }

    def build(self, params: dict):
        return HistGradientBoostingClassifier(
            categorical_features=self.cat_features if self.cat_features else None,
            random_state=self.seed,
            early_stopping=False,
            **params,
        )

    def prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.cat_features:
            return X
        X = X.copy()
        for col in self.cat_features:
            codes = X[col].cat.codes.astype("float32")
            codes[codes < 0] = np.nan  # NaN categoriel -> manquant natif
            X[col] = codes
        return X

    def shap_values(self, estimator, X_sample: pd.DataFrame):
        prepared = self.prepare(X_sample)
        try:
            explainer = shap.TreeExplainer(estimator)
            sv = _positive_class(explainer.shap_values(prepared))
        except Exception as exc:
            print(
                f"[hist_gradient_boosting] TreeExplainer indisponible "
                f"({type(exc).__name__}: {exc}). Repli sur une importance par "
                f"permutation des observations (decalage de probabilite predite "
                f"quand une variable est permutee entre les lignes), beaucoup "
                f"moins couteux qu'un explainer generique et sans besoin de la "
                f"cible. Version de shap installee : {shap.__version__} -> une "
                f"mise a jour de shap peut retablir le TreeExplainer natif."
            )
            sv = _permutation_impact(estimator, prepared, self.seed)
        display = _display_matrix(X_sample)
        col_to_var = {c: c for c in X_sample.columns}
        return sv, list(X_sample.columns), display, col_to_var


def _permutation_impact(estimator, prepared: pd.DataFrame, seed: int) -> np.ndarray:
    """Substitut a SHAP quand TreeExplainer echoue.

    Pour chaque variable, permute ses valeurs entre les observations (les
    autres variables restant fixes) et mesure, pour chaque ligne, le decalage
    de probabilite predite par rapport a la prediction de reference. Ne
    necessite pas la cible. Cout : n_variables + 1 appels a predict_proba sur
    tout l'echantillon, contre des heures pour un explainer generique ligne
    par ligne. L'echelle n'est pas celle d'une contribution SHAP additive ;
    le classement des variables reste neanmoins exploitable.
    """
    rng = np.random.default_rng(seed)
    baseline = estimator.predict_proba(prepared)[:, 1]
    n = len(prepared)
    impact = np.zeros((n, prepared.shape[1]), dtype=float)
    for j, col in enumerate(prepared.columns):
        shuffled = prepared.copy()
        shuffled[col] = shuffled[col].to_numpy()[rng.permutation(n)]
        impact[:, j] = estimator.predict_proba(shuffled)[:, 1] - baseline
    return impact
