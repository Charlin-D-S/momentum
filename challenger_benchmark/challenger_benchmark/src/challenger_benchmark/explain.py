"""Calcul SHAP et agregation au niveau variable.

L'agregation somme les |SHAP| des colonnes d'une meme variable d'origine (utile
pour les colonnes one-hot de la logistique), produisant un classement comparable
entre familles et avec la grille.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models.base import ChallengerModel


def sample_for_shap(X: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if len(X) <= size:
        return X
    return X.sample(n=size, random_state=seed)


def compute_shap(model: ChallengerModel, estimator, X_sample: pd.DataFrame):
    """Renvoie (sv, feat_names, display, col_to_var)."""
    return model.shap_values(estimator, X_sample)


def variable_importance(
    sv: np.ndarray, feat_names: list[str], col_to_var: dict[str, str]
) -> pd.Series:
    """Importance par variable = somme des moyennes des |SHAP| de ses colonnes."""
    mean_abs = np.abs(sv).mean(axis=0)
    per_col = pd.Series(mean_abs, index=feat_names)
    variables = pd.Index([col_to_var[c] for c in feat_names], name="variable")
    agg = per_col.groupby(variables).sum().sort_values(ascending=False)
    return agg
