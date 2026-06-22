"""Chargement et preparation des donnees, en Polars.

Flux : drivers.xlsx -> resolution des variables -> scan parquet projete ->
decodage des codes en modalites -> conversion pandas avec dtypes contractuels
(category pour les qualitatives, float32 pour les quantitatives).
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Union

import pandas as pd
import polars as pl

from .config import Config


def load_drivers(cfg: Config) -> pl.DataFrame:
    """Lit le fichier drivers et verifie les colonnes attendues."""
    d = cfg.drivers
    df = pl.read_excel(d.path)
    expected = {d.variable_col, d.type_col}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes absentes de {d.path} : {sorted(missing)}")
    return df


def resolve_variables(cfg: Config, drivers: pl.DataFrame) -> list[str]:
    """Renvoie la liste finale des variables.

    'all' -> toutes les variables dont Risk_drivers == RISK_DRIVER.
    Liste -> la liste fournie, controlee contre drivers.
    """
    d = cfg.drivers
    known = set(drivers[d.variable_col].to_list())

    if cfg.use_all_variables:
        if d.risk_driver_col not in drivers.columns:
            raise ValueError(
                f"variables='all' requiert la colonne '{d.risk_driver_col}' dans {d.path}"
            )
        selected = (
            drivers.filter(pl.col(d.risk_driver_col) == d.risk_driver_flag)
            [d.variable_col].to_list()
        )
        if not selected:
            raise ValueError(
                f"Aucune variable avec {d.risk_driver_col} == '{d.risk_driver_flag}'"
            )
        return selected

    requested = list(cfg.variables)
    unknown = [v for v in requested if v not in known]
    if unknown:
        raise ValueError(f"Variables absentes de drivers : {unknown}")
    return requested


def split_feature_types(
    cfg: Config, drivers: pl.DataFrame, variables: list[str]
) -> tuple[list[str], list[str]]:
    """Separe les variables en quantitatives et qualitatives via la colonne Type."""
    d = cfg.drivers
    type_map = dict(zip(drivers[d.variable_col].to_list(), drivers[d.type_col].to_list()))
    num_features = [v for v in variables if str(type_map[v]).lower() == d.num_flag]
    cat_features = [v for v in variables if str(type_map[v]).lower() == d.cat_flag]

    classified = set(num_features) | set(cat_features)
    unclassified = [v for v in variables if v not in classified]
    if unclassified:
        raise ValueError(
            f"Type non reconnu (attendu '{d.num_flag}'/'{d.cat_flag}') pour : {unclassified}"
        )
    return num_features, cat_features


def _load_mapping(path: Union[str, Path]) -> dict[str, dict[str, str]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _decode_categoricals(
    lf: pl.LazyFrame, mapping: dict[str, dict[str, str]], cat_features: list[str]
) -> pl.LazyFrame:
    """Remplace les codes par leurs modalites reelles (en chaine), nulls preserves."""
    exprs = []
    for var in cat_features:
        if var in mapping:
            # cle JSON = chaine : on caste le code en Utf8 avant de mapper.
            table = {str(k): str(v) for k, v in mapping[var].items()}
            exprs.append(
                pl.col(var).cast(pl.Utf8).replace(table).alias(var)
            )
        else:
            exprs.append(pl.col(var).cast(pl.Utf8).alias(var))
    return lf.with_columns(exprs) if exprs else lf


def load_dataset(
    cfg: Config, variables: list[str], num_features: list[str], cat_features: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Charge le parquet, decode, et renvoie (X_train, y_train, X_test, y_test).

    Memoire : scan paresseux avec projection des seules colonnes utiles, decodage
    en lazy, un seul collect, puis liberation immediate des objets intermediaires.
    """
    data = cfg.data
    cols = list(dict.fromkeys(variables + [data.target, data.sample_col]))

    mapping = _load_mapping(cfg.category_mapping.path)

    lf = pl.scan_parquet(data.path).select(cols)
    lf = _decode_categoricals(lf, mapping, cat_features)
    df = lf.collect()
    del lf, mapping
    gc.collect()

    train_mask = pl.col(data.sample_col).is_in(list(data.train_values))
    test_mask = pl.col(data.sample_col) == data.test_value

    train_pl = df.filter(train_mask).drop(data.sample_col)
    test_pl = df.filter(test_mask).drop(data.sample_col)
    del df
    gc.collect()

    if train_pl.height == 0:
        raise ValueError(f"Aucune ligne train (valeurs {data.train_values})")
    if test_pl.height == 0:
        raise ValueError(f"Aucune ligne test (valeur '{data.test_value}')")

    X_train, y_train = _to_pandas(train_pl, data.target, num_features, cat_features)
    del train_pl
    gc.collect()
    X_test, y_test = _to_pandas(test_pl, data.target, num_features, cat_features)
    del test_pl
    gc.collect()

    return X_train, y_train, X_test, y_test


def _to_pandas(
    frame: pl.DataFrame, target: str, num_features: list[str], cat_features: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Conversion pandas avec dtypes contractuels et cible entiere."""
    pdf = frame.to_pandas()
    y = pdf[target].astype("int8")
    X = pdf.drop(columns=[target])

    for col in num_features:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype("float32")
    for col in cat_features:
        X[col] = X[col].astype("category")

    ordered = [c for c in (num_features + cat_features) if c in X.columns]
    return X[ordered], y
