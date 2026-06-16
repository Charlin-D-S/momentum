"""
Génère un dataset synthétique et une scorecard de test pour valider l'app.

- Scorecard avec 6 variables (4 numériques + 2 catégorielles) + intercept
- 5 000 individus, mélange de valeurs manquantes, défaut cohérent avec le score
"""
from __future__ import annotations

import numpy as np
import polars as pl
from pathlib import Path


def build_scorecard() -> pl.DataFrame:
    """Construit une scorecard exemple en respectant le schéma attendu."""
    # Convention : sigmoid(score_logit) = P(défaut)
    # → coef positif = bin plus risqué ; points positifs = bin plus sain (sens inverse).
    rows = [
        # Variable, Label, coef, points_1000
        ("Intercept", "-", -2.0, 500),

        # --- anciennete (années) : long historique = moins risqué ---
        ("anciennete", "[-inf, 1.0)", 0.50, -45),
        ("anciennete", "[1.0, 3.0)", 0.20, -18),
        ("anciennete", "[3.0, 10.0)", -0.10, 9),
        ("anciennete", "[10.0, inf) + MISSING", -0.45, 40),

        # --- revenus (€/mois) : revenus élevés = moins risqué ---
        ("revenus", "[-inf, 1500.0)", 0.60, -55),
        ("revenus", "[1500.0, 3000.0)", 0.10, -9),
        ("revenus", "[3000.0, 6000.0)", -0.25, 22),
        ("revenus", "[6000.0, inf)", -0.55, 50),

        # --- nb_transactions : plus actif = moins risqué ---
        ("nb_transactions", "[-inf, 5.0)", 0.30, -27),
        ("nb_transactions", "[5.0, 20.0)", -0.05, 4),
        ("nb_transactions", "[20.0, 50.0)", -0.20, 18),
        ("nb_transactions", "[50.0, inf) + MISSING", -0.35, 32),

        # --- incidents (sentinelle -1 pour missing informatif) ---
        ("incidents", "[-1.0, 0.0)", 0.40, -36),
        ("incidents", "[0.0, 1.0)", -0.30, 27),
        ("incidents", "[1.0, 3.0)", 0.20, -18),
        ("incidents", "[3.0, inf)", 0.70, -63),

        # --- type_client ---
        ("type_client", "['Artisan', 'Commerçant']", -0.15, 13),
        ("type_client", "['Profession libérale']", -0.40, 36),
        ("type_client", "['Auto-entrepreneur', 'MISSING']", 0.25, -22),

        # --- region ---
        ("region", "['IDF', 'Sud-Est']", -0.20, 18),
        ("region", "['Nord', 'Ouest']", -0.05, 4),
        ("region", "['Centre', 'Est', 'MISSING']", 0.10, -9),
    ]
    return pl.DataFrame(
        rows,
        schema=["Variables", "Label", "coef", "points_1000"],
        orient="row",
    )


def build_dataset(n: int = 5000, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)

    # Variables explicatives
    anciennete = rng.exponential(scale=5.0, size=n)
    revenus = rng.lognormal(mean=8.0, sigma=0.5, size=n)
    nb_tx = rng.poisson(lam=15, size=n).astype(float)
    incidents = rng.choice([-1.0, 0.0, 1.0, 2.0, 3.0, 5.0],
                           size=n, p=[0.15, 0.55, 0.15, 0.08, 0.05, 0.02])
    types = rng.choice(
        ["Artisan", "Commerçant", "Profession libérale", "Auto-entrepreneur"],
        size=n, p=[0.35, 0.30, 0.15, 0.20],
    )
    regions = rng.choice(
        ["IDF", "Sud-Est", "Nord", "Ouest", "Centre", "Est"],
        size=n, p=[0.30, 0.18, 0.13, 0.15, 0.12, 0.12],
    )

    # Injection de valeurs manquantes (~5 %)
    miss_anc = rng.random(n) < 0.05
    miss_tx = rng.random(n) < 0.05
    miss_region = rng.random(n) < 0.03

    anciennete_col = np.where(miss_anc, np.nan, anciennete)
    nb_tx_col = np.where(miss_tx, np.nan, nb_tx)
    region_col = np.where(miss_region, None, regions)

    # Logit avec la même convention que la scorecard : positif = défaut.
    # Coefs miment ceux de la scorecard (sens identique).
    logit = (
        -2.0
        + np.where(np.isnan(anciennete_col), -0.45,
            np.where(anciennete_col < 1, 0.5,
            np.where(anciennete_col < 3, 0.2,
            np.where(anciennete_col < 10, -0.1, -0.45))))
        + np.where(revenus < 1500, 0.6,
           np.where(revenus < 3000, 0.1,
           np.where(revenus < 6000, -0.25, -0.55)))
        + np.where(np.isnan(nb_tx_col), -0.35,
           np.where(nb_tx_col < 5, 0.3,
           np.where(nb_tx_col < 20, -0.05,
           np.where(nb_tx_col < 50, -0.2, -0.35))))
        + np.where(incidents == -1, 0.4,
           np.where(incidents == 0, -0.3,
           np.where(incidents < 3, 0.2, 0.7)))
    )
    logit = logit + rng.normal(0, 0.5, size=n)
    proba_def = 1 / (1 + np.exp(-logit))   # sigmoid(logit) = P(défaut)
    defaut = (rng.random(n) < proba_def).astype(int)

    df = pl.DataFrame({
        "id_client": [f"C{i:06d}" for i in range(n)],
        "anciennete": anciennete_col,
        "revenus": revenus,
        "nb_transactions": nb_tx_col,
        "incidents": incidents,
        "type_client": types,
        "region": region_col,
        "defaut_obs": defaut,
    })
    # NaN → null (cohérent avec les écritures parquet usuelles)
    df = df.with_columns([
        pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
        for c in ("anciennete", "nb_transactions")
    ])
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    sc = build_scorecard()
    ds = build_dataset(n=5000)

    sc.write_parquet(out_dir / "scorecard.parquet")
    ds.write_parquet(out_dir / "dataset_predit.parquet")

    print(f"Scorecard écrite : {sc.height} lignes")
    print(f"Dataset écrit : {ds.height} lignes, {ds.width} colonnes")
    print(f"Colonnes : {ds.columns}")
    print(f"Taux de défaut : {ds['defaut_obs'].mean():.2%}")
