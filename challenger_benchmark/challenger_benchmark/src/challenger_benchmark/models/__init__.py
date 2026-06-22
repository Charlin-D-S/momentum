"""Registre : instancie les challengers demandes par la config."""
from __future__ import annotations

from .base import ChallengerModel
from .logistic import LogisticModel
from .hist_gb import HistGBModel
from .xgboost import XGBoostModel
from .catboost import CatBoostModel

_REGISTRY = {
    LogisticModel.name: LogisticModel,
    HistGBModel.name: HistGBModel,
    XGBoostModel.name: XGBoostModel,
    CatBoostModel.name: CatBoostModel,
}


def build_models(
    names, num_features: list[str], cat_features: list[str], seed: int
) -> list[ChallengerModel]:
    return [_REGISTRY[n](num_features, cat_features, seed) for n in names]
