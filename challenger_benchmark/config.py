"""Configuration du benchmark de challengers.

Les dataclasses decrivent la config ; `_parse_config` la construit a partir d'un
dict (donc testable sans lecture de fichier) ; `load_config` lit le YAML.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import yaml

ALL_MODELS = ["logistic_regression", "hist_gradient_boosting", "xgboost", "catboost"]


@dataclass(frozen=True)
class DataConfig:
    path: str
    target: str
    sample_col: str = "sample"
    train_values: tuple[str, ...] = ("train", "val")
    test_value: str = "test"


@dataclass(frozen=True)
class DriversConfig:
    path: str
    variable_col: str = "Variable"
    description_col: str = "Description"
    type_col: str = "Type"
    cat_flag: str = "c"
    num_flag: str = "q"
    risk_driver_col: str = "Risk_drivers"
    risk_driver_flag: str = "RISK_DRIVER"


@dataclass(frozen=True)
class MappingConfig:
    path: str  # JSON imbrique { "variable": { "code": "modalite" } }


@dataclass(frozen=True)
class TuningConfig:
    n_trials: int = 50
    cv_folds: int = 5
    seed: int = 42
    metric: str = "auc"
    # Fraction de X_train utilisee pour le TUNING SEUL (Optuna + CV). Le
    # modele final est toujours reentraine sur l'integralite de X_train,
    # quelle que soit cette valeur. 1.0 = pas de sous-echantillonnage
    # (comportement inchange).
    sample_frac: float = 1.0


@dataclass(frozen=True)
class ShapConfig:
    sample_size: int = 5000
    top_n: int = 20


@dataclass(frozen=True)
class Config:
    data: DataConfig
    drivers: DriversConfig
    category_mapping: MappingConfig
    output_dir: str
    variables: Union[str, tuple[str, ...]]  # "all" ou liste
    tuning: TuningConfig = field(default_factory=TuningConfig)
    shap: ShapConfig = field(default_factory=ShapConfig)
    models: tuple[str, ...] = tuple(ALL_MODELS)

    @property
    def use_all_variables(self) -> bool:
        return isinstance(self.variables, str) and self.variables.lower() == "all"


def _parse_config(raw: dict) -> Config:
    """Construit une Config a partir d'un dict deja charge. Aucune I/O."""
    for key in ("data", "drivers", "category_mapping", "output_dir", "variables"):
        if key not in raw:
            raise KeyError(f"Cle de configuration manquante : '{key}'")

    data = DataConfig(**raw["data"])
    drivers = DriversConfig(**raw["drivers"])
    mapping = MappingConfig(**raw["category_mapping"])
    tuning = TuningConfig(**raw.get("tuning", {}))
    shap_cfg = ShapConfig(**raw.get("shap", {}))

    variables = raw["variables"]
    if isinstance(variables, list):
        variables = tuple(variables)
    elif not (isinstance(variables, str) and variables.lower() == "all"):
        raise ValueError("'variables' doit etre une liste ou la chaine 'all'")

    models = tuple(raw.get("models", ALL_MODELS))
    unknown = set(models) - set(ALL_MODELS)
    if unknown:
        raise ValueError(f"Modeles inconnus : {sorted(unknown)}. Connus : {ALL_MODELS}")

    return Config(
        data=data,
        drivers=drivers,
        category_mapping=mapping,
        output_dir=raw["output_dir"],
        variables=variables,
        tuning=tuning,
        shap=shap_cfg,
        models=models,
    )


def load_config(path: Union[str, Path]) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _parse_config(raw)
