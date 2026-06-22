"""Export des artefacts : un dossier par modele, un Excel recapitulatif a la racine."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


def model_dir(output_dir: Path, model_name: str) -> Path:
    d = output_dir / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


# Fichiers qui constituent un export "complet" pour un modele : si tous sont
# presents, rien ne doit etre recalcule (ni tuning, ni ajustement, ni SHAP).
MODEL_ARTIFACTS = ("model.joblib", "best_params.json", "shap_importance.csv")
FIGURE_ARTIFACTS = ("roc.png", "shap_importance.png", "shap_impact.png", "optuna_history.png")


def artifacts_complete(folder: Path) -> bool:
    """Vrai si ce dossier contient deja un export complet (donnees + figures)."""
    folder = Path(folder)
    if not folder.exists():
        return False
    return all((folder / f).exists() for f in MODEL_ARTIFACTS + FIGURE_ARTIFACTS)


def load_artifacts(folder: Path):
    """Recharge un modele deja exporte : (estimateur, best_params, importance SHAP)."""
    folder = Path(folder)
    estimator = joblib.load(folder / "model.joblib")
    with open(folder / "best_params.json", "r", encoding="utf-8") as fh:
        best_params = json.load(fh)
    importance = pd.read_csv(folder / "shap_importance.csv", index_col=0)["shap_importance"]
    return estimator, best_params, importance


def save_model(estimator, best_params: dict, importance: pd.Series, folder: Path) -> None:
    joblib.dump(estimator, folder / "model.joblib")
    with open(folder / "best_params.json", "w", encoding="utf-8") as fh:
        json.dump(best_params, fh, indent=2, ensure_ascii=False)
    importance.rename("shap_importance").to_frame().to_csv(
        folder / "shap_importance.csv", encoding="utf-8"
    )


def write_summary_excel(
    summary: pd.DataFrame,
    consolidated_shap: pd.DataFrame,
    descriptions: dict[str, str],
    path: Path,
) -> None:
    desc_df = (
        pd.Series(descriptions, name="description")
        .rename_axis("variable")
        .reset_index()
    )
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="performances", index=False)
        consolidated_shap.to_excel(xw, sheet_name="shap_consolide")
        desc_df.to_excel(xw, sheet_name="variables", index=False)
