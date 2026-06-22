"""Challenger XGBoost. Categorielles via enable_categorical, manquants natifs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
import shap
from xgboost import XGBClassifier

from .base import ChallengerModel


class XGBoostModel(ChallengerModel):
    name = "xgboost"

    def search_space(self, trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

    def build(self, params: dict):
        return XGBClassifier(
            enable_categorical=True,
            tree_method="hist",
            eval_metric="auc",
            n_jobs=-1,
            random_state=self.seed,
            **params,
        )

    def shap_values(self, estimator, X_sample: pd.DataFrame):
        explainer = shap.TreeExplainer(estimator)
        sv = explainer.shap_values(self.prepare(X_sample))
        sv = _positive_class(sv)
        display = _display_matrix(X_sample)
        col_to_var = {c: c for c in X_sample.columns}
        return sv, list(X_sample.columns), display, col_to_var


def _positive_class(sv):
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, m, classes)
        return sv[:, :, 1]
    return sv


def _display_matrix(X: pd.DataFrame) -> np.ndarray:
    """Matrice numerique pour la couleur du beeswarm (codes pour les categorielles)."""
    out = []
    for col in X.columns:
        s = X[col]
        if str(s.dtype) == "category":
            out.append(s.cat.codes.to_numpy().astype(float))
        else:
            out.append(pd.to_numeric(s, errors="coerce").to_numpy().astype(float))
    return np.column_stack(out)
