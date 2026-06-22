"""Challenger regression logistique classique (sans binning ni stepwise).

Pretraitement encapsule dans un Pipeline :
- numeriques : imputation mediane + indicateur de manquant (preserve le signal
  MNAR, coherent avec le traitement par sentinelles de la grille) + standardisation ;
- categorielles : modalite MISSING puis one-hot.
SHAP via LinearExplainer sur la matrice transformee, puis agregation des colonnes
one-hot vers la variable d'origine pour un classement comparable aux modeles a arbres.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base import ChallengerModel, MISSING_TOKEN


class LogisticModel(ChallengerModel):
    name = "logistic_regression"

    def _preprocessor(self) -> ColumnTransformer:
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        return ColumnTransformer([
            ("num", num_pipe, self.num_features),
            ("cat", cat_pipe, self.cat_features),
        ])

    def search_space(self, trial: optuna.Trial) -> dict:
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    def build(self, params: dict):
        clf = LogisticRegression(
            solver="lbfgs", max_iter=2000,
            random_state=self.seed, **params,
        )
        return Pipeline([("preprocess", self._preprocessor()), ("clf", clf)])

    def shap_values(self, estimator, X_sample: pd.DataFrame):
        pre = estimator.named_steps["preprocess"]
        clf = estimator.named_steps["clf"]
        Z = pre.transform(X_sample)
        feat_names = list(pre.get_feature_names_out())

        explainer = shap.LinearExplainer(clf, Z)
        sv = np.asarray(explainer.shap_values(Z))
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        col_to_var = {name: self._origin_variable(name) for name in feat_names}
        return sv, feat_names, Z, col_to_var

    def _origin_variable(self, feature_name: str) -> str:
        """Remonte un nom de colonne transformee vers la variable d'origine."""
        if feature_name.startswith("num__"):
            rest = feature_name[len("num__"):]
            if rest.startswith("missingindicator_"):
                rest = rest[len("missingindicator_"):]
            return rest
        if feature_name.startswith("cat__"):
            rest = feature_name[len("cat__"):]
            candidates = [v for v in self.cat_features if rest.startswith(v + "_")]
            if candidates:
                return max(candidates, key=len)
            return rest
        return feature_name
