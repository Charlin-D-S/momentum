"""Metriques de discrimination, sans seuil : AUC, Gini, KS, et ecart train/test."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def gini(auc: float) -> float:
    return 2.0 * auc - 1.0


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


@dataclass
class SplitMetrics:
    auc: float
    gini: float
    ks: float


@dataclass
class ModelMetrics:
    model: str
    train: SplitMetrics
    test: SplitMetrics
    auc_gap: float  # train - test, lecture du sur-apprentissage

    def to_row(self) -> dict:
        return {
            "model": self.model,
            "auc_train": self.train.auc,
            "auc_test": self.test.auc,
            "gini_train": self.train.gini,
            "gini_test": self.test.gini,
            "ks_train": self.train.ks,
            "ks_test": self.test.ks,
            "auc_gap": self.auc_gap,
        }


def _split(y_true, y_score) -> SplitMetrics:
    auc = roc_auc_score(y_true, y_score)
    return SplitMetrics(auc=auc, gini=gini(auc), ks=ks_statistic(y_true, y_score))


def evaluate(name, y_train, p_train, y_test, p_test) -> ModelMetrics:
    tr, te = _split(y_train, p_train), _split(y_test, p_test)
    return ModelMetrics(model=name, train=tr, test=te, auc_gap=tr.auc - te.auc)


def roc_points(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr
