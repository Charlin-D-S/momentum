"""Essai de bout en bout sur donnees synthetiques (les 4 modeles, n_trials reduit).

Cree base.parquet (avec colonne sample et codes categoriels), drivers.xlsx,
mapping.json, puis lance le pipeline et verifie la presence des artefacts.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from challenger_benchmark.config import _parse_config
from challenger_benchmark.pipeline import run_with_config


def build_synthetic(tmp: Path, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    # quantitatives
    anciennete = rng.normal(5, 2, n)
    ca = rng.lognormal(10, 1, n)
    ratio = rng.normal(0, 1, n)
    # categorielles encodees en codes
    secteur = rng.integers(1, 4, n)          # 1,2,3
    incident = rng.integers(0, 2, n)         # 0,1
    # quelques manquants
    ca[rng.random(n) < 0.15] = np.nan
    ratio[rng.random(n) < 0.10] = np.nan
    secteur_f = secteur.astype(float)
    secteur_f[rng.random(n) < 0.05] = np.nan

    logit = (-1.0 + 0.4 * (anciennete - 5) / 2 - 0.6 * np.nan_to_num(ratio)
             + 0.8 * incident + 0.3 * (secteur == 3))
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)

    sample = rng.choice(["train", "val", "test"], size=n, p=[0.55, 0.15, 0.30])

    df = pl.DataFrame({
        "anciennete": anciennete,
        "ca": ca,
        "ratio": ratio,
        "secteur": secteur_f,
        "incident": incident.astype(float),
        "bruit": rng.normal(0, 1, n),
        "defaut_12m": y,
        "sample": sample,
    })
    base = tmp / "base.parquet"
    df.write_parquet(base)

    drivers = pl.DataFrame({
        "Variable": ["anciennete", "ca", "ratio", "secteur", "incident", "bruit"],
        "Description": ["Anciennete", "Chiffre affaires", "Ratio comptable",
                        "Secteur", "Incident paiement", "Variable bruit"],
        "Type": ["q", "q", "q", "c", "c", "q"],
        "Risk_drivers": ["RISK_DRIVER"] * 5 + ["NO"],
    })
    drivers.write_excel(tmp / "drivers.xlsx")

    mapping = {
        "secteur": {"1": "Commerce", "2": "Services", "3": "Industrie"},
        "incident": {"0": "Non", "1": "Oui"},
    }
    import json
    (tmp / "mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    return base


def main():
    tmp = ROOT / "_smoke"
    tmp.mkdir(exist_ok=True)
    build_synthetic(tmp)

    raw = {
        "data": {"path": str(tmp / "base.parquet"), "target": "defaut_12m"},
        "drivers": {"path": str(tmp / "drivers.xlsx")},
        "category_mapping": {"path": str(tmp / "mapping.json")},
        "output_dir": str(tmp / "outputs"),
        "variables": "all",
        "tuning": {"n_trials": 3, "cv_folds": 3, "seed": 1},
        "shap": {"sample_size": 800, "top_n": 10},
    }
    cfg = _parse_config(raw)
    summary = run_with_config(cfg)
    print("\n", summary.to_string(index=False))

    out = Path(raw["output_dir"])
    expected = ["_summary.xlsx", "_comparison.png", "_shap_consolidated.png"]
    for f in expected:
        assert (out / f).exists(), f"manquant : {f}"
    for m in ["logistic_regression", "hist_gradient_boosting", "xgboost", "catboost"]:
        for f in ["roc.png", "shap_importance.png", "shap_impact.png",
                  "optuna_history.png", "model.joblib", "best_params.json"]:
            assert (out / m / f).exists(), f"manquant : {m}/{f}"
    print("\nSMOKE OK : tous les artefacts presents.")


if __name__ == "__main__":
    main()
