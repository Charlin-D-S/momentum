"""Optimisation des hyperparametres par Optuna.

Validation croisee stratifiee (sans GroupKFold, choix assume : l'optimisme
intra-emprunteur contamine la selection d'hyperparametres, pas les chiffres
finaux rapportes sur un test reellement externe). Memoire : chaque fold libere
son estimateur avant le suivant.
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import TuningConfig
from .models.base import ChallengerModel

optuna.logging.set_verbosity(optuna.logging.WARNING)


def subsample_for_tuning(X: pd.DataFrame, y: pd.Series, frac: float, seed: int):
    """Sous-echantillon stratifie utilise uniquement pour le tuning Optuna.

    Le modele final est toujours reentraine sur l'integralite de X_train/y_train
    ailleurs dans le pipeline ; cette fonction ne reduit que le cout de la
    recherche d'hyperparametres. frac >= 1.0 -> aucun sous-echantillonnage,
    renvoie X, y inchanges (memes objets, pas de copie).
    """
    if frac >= 1.0:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=frac, stratify=y, random_state=seed
    )
    return X_sub, y_sub


def _cv_auc(model: ChallengerModel, params: dict, X, y, cfg: TuningConfig) -> float:
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        est = model.build(params)
        model.fit(est, X_tr, y_tr)
        proba = model.predict_proba(est, X_va)
        scores.append(roc_auc_score(y_va, proba))
        del est, X_tr, X_va, proba
        gc.collect()
    return float(np.mean(scores))


def tune(model: ChallengerModel, X, y, cfg: TuningConfig,
         callbacks=None, show_progress_bar=False) -> tuple[dict, optuna.Study]:
    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = model.search_space(trial)
        return _cv_auc(model, params, X, y, cfg)

    study.optimize(objective, n_trials=cfg.n_trials, callbacks=callbacks,
                   show_progress_bar=show_progress_bar)
    return study.best_params, study
