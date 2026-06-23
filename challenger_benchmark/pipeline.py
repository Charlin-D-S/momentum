"""Orchestration du benchmark.

Pour chaque modele : optimisation Optuna sur le train, reentrainement sur tout
le train, evaluation unique sur le test, SHAP, figures, export. La memoire est
liberee entre modeles : aucun objet lourd (estimateur ajuste, matrices SHAP) ne
survit au passage au modele suivant.
"""
from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd

from .config import Config, load_config
from . import data as data_mod
from .models import build_models
from .tuning import tune, subsample_for_tuning
from .evaluation import evaluate
from .explain import sample_for_shap, compute_shap, variable_importance
from . import plots, export


def run(config_path: str) -> pd.DataFrame:
    cfg = load_config(config_path)
    return run_with_config(cfg)


def run_with_config(cfg: Config) -> pd.DataFrame:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    drivers = data_mod.load_drivers(cfg)
    variables = data_mod.resolve_variables(cfg, drivers)
    num_features, cat_features = data_mod.split_feature_types(cfg, drivers, variables)
    descriptions = _descriptions(cfg, drivers, variables)

    X_train, y_train, X_test, y_test = data_mod.load_dataset(
        cfg, variables, num_features, cat_features
    )

    models = build_models(cfg.models, num_features, cat_features, cfg.tuning.seed)
    summary_rows: list[dict] = []
    importances: dict[str, pd.Series] = {}

    for model in models:
        X_tune, y_tune = subsample_for_tuning(
            X_train, y_train, cfg.tuning.sample_frac, cfg.tuning.seed
        )
        if cfg.tuning.sample_frac < 1.0:
            print(f"[{model.name}] optimisation sur un sous-echantillon "
                  f"({len(X_tune)}/{len(X_train)} lignes, "
                  f"{cfg.tuning.sample_frac:.0%}) ; {cfg.tuning.n_trials} essais...")
        else:
            print(f"[{model.name}] optimisation ({cfg.tuning.n_trials} essais)...")
        best_params, study = tune(model, X_tune, y_tune, cfg.tuning)
        del X_tune, y_tune

        estimator = model.build(best_params)
        model.fit(estimator, X_train, y_train)
        p_train = model.predict_proba(estimator, X_train)
        p_test = model.predict_proba(estimator, X_test)
        metrics = evaluate(model.name, y_train, p_train, y_test, p_test)
        summary_rows.append(metrics.to_row())
        print(f"[{model.name}] AUC test = {metrics.test.auc:.4f} | "
              f"Gini test = {metrics.test.gini:.4f} | ecart = {metrics.auc_gap:+.4f}")

        X_shap = sample_for_shap(X_test, cfg.shap.sample_size, cfg.tuning.seed)
        sv, feat_names, display, col_to_var = compute_shap(model, estimator, X_shap)
        importance = variable_importance(sv, feat_names, col_to_var)
        importances[model.name] = importance

        folder = export.model_dir(output_dir, model.name)
        plots.plot_roc(y_train, p_train, y_test, p_test,
                       metrics.train.auc, metrics.test.auc, folder / "roc.png")
        plots.plot_shap_importance(importance, folder / "shap_importance.png",
                                   cfg.shap.top_n, descriptions)
        plots.plot_shap_beeswarm(sv, display, feat_names,
                                 folder / "shap_impact.png", cfg.shap.top_n)
        plots.plot_optuna_history(study, folder / "optuna_history.png")
        export.save_model(estimator, best_params, importance, folder)

        del estimator, study, p_train, p_test, sv, display, X_shap, feat_names, col_to_var
        gc.collect()

    summary = pd.DataFrame(summary_rows)
    consolidated = pd.DataFrame(importances)
    plots.plot_comparison(summary, output_dir / "_comparison.png")
    plots.plot_consolidated_shap(importances, output_dir / "_shap_consolidated.png",
                                 cfg.shap.top_n, descriptions)
    export.write_summary_excel(summary, consolidated, descriptions,
                               output_dir / "_summary.xlsx")
    print(f"\nTermine. Resultats dans {output_dir.resolve()}")
    return summary


def _descriptions(cfg: Config, drivers, variables: list[str]) -> dict[str, str]:
    d = cfg.drivers
    if d.description_col not in drivers.columns:
        return {v: "" for v in variables}
    full = dict(zip(drivers[d.variable_col].to_list(), drivers[d.description_col].to_list()))
    return {v: ("" if full.get(v) is None else str(full.get(v))) for v in variables}
