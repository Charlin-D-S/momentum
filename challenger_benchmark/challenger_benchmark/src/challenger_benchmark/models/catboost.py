"""Challenger CatBoost.

CatBoost gere les manquants numeriques nativement mais refuse le NaN en
categoriel : on le remplace par le jeton MISSING dans une copie locale, sans
muter l'entree partagee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
import shap
from catboost import CatBoostClassifier, Pool

from .base import ChallengerModel, MISSING_TOKEN
from .xgboost import _positive_class, _display_matrix


class CatBoostModel(ChallengerModel):
    name = "catboost"

    def search_space(self, trial: optuna.Trial) -> dict:
        return {
            "iterations": trial.suggest_int("iterations", 200, 800, step=100),
            "depth": trial.suggest_int("depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        }

    def build(self, params: dict):
        return CatBoostClassifier(
            cat_features=self.cat_features,
            random_seed=self.seed,
            eval_metric="AUC",
            verbose=False,
            allow_writing_files=False,
            **params,
        )

    def prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.cat_features:
            return X
        X = X.copy()
        for col in self.cat_features:
            s = X[col].astype("object")
            X[col] = s.where(s.notna(), MISSING_TOKEN).astype(str)
        return X

    def shap_values(self, estimator, X_sample: pd.DataFrame):
        prepared = self.prepare(X_sample)
        pool = Pool(prepared, cat_features=self.cat_features)
        explainer = shap.TreeExplainer(estimator)
        sv = explainer.shap_values(pool)
        sv = _positive_class(sv)
        if sv.shape[1] == X_sample.shape[1] + 1:  # colonne de biais en fin
            sv = sv[:, :-1]
        display = _display_matrix(X_sample)
        col_to_var = {c: c for c in X_sample.columns}
        return sv, list(X_sample.columns), display, col_to_var
