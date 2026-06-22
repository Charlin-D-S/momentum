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
