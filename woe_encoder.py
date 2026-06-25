"""
woe_encoder.py

Calcule le Weight of Evidence (WoE) sur des variables déjà discrétisées
(bins, intervalles ou catégories), puis applique le mapping obtenu à un
ou plusieurs DataFrames (train, validation, production...).

Convention : target vaut 1 pour un défaut (mauvais), 0 pour un bon.
Cette fonction ne discrétise pas les variables continues, elle suppose
que le binning a déjà été fait en amont (XGBoost depth-1, OptBinning,
ou autre).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def compute_woe(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    bad_label: int = 1,
    epsilon: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """
    Calcule le WoE et la contribution à l'IV par bin pour chaque variable.

    Parameters
    ----------
    df : DataFrame contenant les colonnes déjà binnées et la cible.
    target : nom de la colonne cible (0/1).
    features : colonnes déjà discrétisées (bins, intervalles ou catégories).
    bad_label : valeur de la cible correspondant à un défaut.
    epsilon : lissage de Laplace, évite les WoE infinis sur les bins purs
              (0 défaut ou 0 bon dans un bin).

    Returns
    -------
    detail : table variable / bin / n_obs / n_good / n_bad / woe / iv_contrib.
              IV total d'une variable = detail.groupby("variable")["iv_contrib"].sum()
    mapping : dict {variable: {bin_value: woe}}, prêt pour apply_woe().
    """
    is_bad = (df[target] == bad_label).astype(int)
    total_bad = is_bad.sum()
    total_good = len(df) - total_bad

    if total_good == 0 or total_bad == 0:
        raise ValueError("Il faut au moins un bon et un mauvais dans le DataFrame.")

    detail_rows = []
    mapping: dict[str, dict] = {}

    for col in features:
        grouped = is_bad.groupby(df[col], dropna=False)
        n_obs = grouped.count()
        n_bad = grouped.sum()
        n_good = n_obs - n_bad

        pct_good = (n_good + epsilon) / (total_good + epsilon * len(n_obs))
        pct_bad = (n_bad + epsilon) / (total_bad + epsilon * len(n_obs))

        woe = np.log(pct_good / pct_bad)
        iv_contrib = (pct_good - pct_bad) * woe

        bin_table = pd.DataFrame(
            {
                "variable": col,
                "bin": n_obs.index,
                "n_obs": n_obs.to_numpy(),
                "n_good": n_good.to_numpy(),
                "n_bad": n_bad.to_numpy(),
                "woe": woe.to_numpy(),
                "iv_contrib": iv_contrib.to_numpy(),
            }
        )
        detail_rows.append(bin_table)
        mapping[col] = dict(zip(n_obs.index, woe.to_numpy()))

    detail = pd.concat(detail_rows, ignore_index=True)
    return detail, mapping


def apply_woe(
    df: pd.DataFrame,
    mapping: dict[str, dict],
    missing_value: float = 0.0,
    suffix: str = "_woe",
    inplace_columns: bool = False,
) -> pd.DataFrame:
    """
    Applique un mapping WoE (issu de compute_woe) à un DataFrame quelconque.

    Les bins absents du mapping (catégorie jamais vue à l'apprentissage,
    typiquement un signe de dérive de population) sont remplacés par
    `missing_value` et déclenchent un avertissement.

    Parameters
    ----------
    df : DataFrame à transformer, potentiellement différent de celui
         utilisé pour compute_woe (cas train -> validation/production).
    mapping : dict {variable: {bin_value: woe}}.
    missing_value : valeur attribuée aux bins non vus à l'apprentissage.
    suffix : suffixe des colonnes WoE créées.
    inplace_columns : si True, remplace les colonnes d'origine au lieu
                      d'ajouter des colonnes <var><suffix>.

    Returns
    -------
    Copie de df avec les colonnes WoE ajoutées (ou remplacées).
    """
    out = df.copy()

    for col, woe_dict in mapping.items():
        if col not in out.columns:
            raise KeyError(f"Colonne absente du DataFrame à transformer : {col}")

        target_col = col if inplace_columns else f"{col}{suffix}"
        out[target_col] = out[col].map(woe_dict)

        n_missing = out[target_col].isna().sum()
        if n_missing > 0:
            warnings.warn(
                f"{col} : {n_missing} valeur(s) absente(s) du mapping WoE, "
                f"remplacée(s) par {missing_value}.",
                stacklevel=2,
            )
            out[target_col] = out[target_col].fillna(missing_value)

    return out


def fit_transform_woe(
    df_train: pd.DataFrame,
    df_other: pd.DataFrame,
    target: str,
    features: list[str],
    bad_label: int = 1,
    epsilon: float = 0.5,
    **apply_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Calcule le WoE sur df_train et applique le même mapping à df_train
    et à df_other (validation, production...). Garantit que les deux
    jeux partagent exactement le même WoE par bin.

    Returns
    -------
    df_train_woe, df_other_woe, detail, mapping
    """
    detail, mapping = compute_woe(
        df_train, target=target, features=features, bad_label=bad_label, epsilon=epsilon
    )
    df_train_woe = apply_woe(df_train, mapping, **apply_kwargs)
    df_other_woe = apply_woe(df_other, mapping, **apply_kwargs)
    return df_train_woe, df_other_woe, detail, mapping


if __name__ == "__main__":
    # Exemple minimal
    train = pd.DataFrame(
        {
            "age_bin": ["18-25", "26-40", "26-40", "41-60", "18-25", "41-60"],
            "default": [1, 0, 0, 0, 1, 1],
        }
    )
    prod = pd.DataFrame({"age_bin": ["26-40", "41-60", "60+"]})  # "60+" jamais vu en train

    train_woe, prod_woe, detail, mapping = fit_transform_woe(
        train, prod, target="default", features=["age_bin"]
    )

    print(detail)
    print(mapping)
    print(prod_woe)
