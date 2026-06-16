"""
Chargement de config.yaml avec cache Streamlit.

Usage dans n'importe quel module :
    from utils.config import cfg, col_label

    cfg.data.dataset_path          → Path
    cfg.columns.id                 → "id_client"
    col_label("anciennete")        → "Ancienneté (années)"  (si alias défini)
                                   → "anciennete"           (sinon)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
import streamlit as st

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


# ---------------------------------------------------------------------------
# Dataclasses miroir du YAML — typage fort, accès par attribut
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    dataset_path: Path = Path("data/dataset_predit.parquet")
    scorecard_path: Path = Path("data/scorecard.parquet")

    def __post_init__(self) -> None:
        root = Path(__file__).resolve().parent.parent
        self.dataset_path = root / self.dataset_path
        self.scorecard_path = root / self.scorecard_path


@dataclass
class ColumnsConfig:
    id: str = "id_client"
    target: Optional[str] = "defaut_obs"
    aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class FiltersConfig:
    vars: list[str] = field(default_factory=list)
    cols_per_row: int = 3


@dataclass
class DisplayConfig:
    n_boundary_profiles: int = 8
    default_n_bins_calibration: int = 10
    default_n_bins_default_rate: int = 10
    proba_to_points_neighbors: int = 50


@dataclass
class ThresholdsConfig:
    default_seuil1: float = 0.05
    default_seuil2: float = 0.15


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    columns: ColumnsConfig = field(default_factory=ColumnsConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
def _parse_config(path: Path) -> AppConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    d = raw.get("data", {})
    c = raw.get("columns", {})
    f = raw.get("filters", {})
    disp = raw.get("display", {})
    t = raw.get("thresholds", {})

    return AppConfig(
        data=DataConfig(
            dataset_path=Path(d.get("dataset_path", "data/dataset_predit.parquet")),
            scorecard_path=Path(d.get("scorecard_path", "data/scorecard.parquet")),
        ),
        columns=ColumnsConfig(
            id=c.get("id", "id_client"),
            target=c.get("target", "defaut_obs") or None,
            aliases=c.get("aliases") or {},
        ),
        filters=FiltersConfig(
            vars=f.get("vars") or [],
            cols_per_row=int(f.get("cols_per_row", 3)),
        ),
        display=DisplayConfig(
            n_boundary_profiles=int(disp.get("n_boundary_profiles", 8)),
            default_n_bins_calibration=int(disp.get("default_n_bins_calibration", 10)),
            default_n_bins_default_rate=int(disp.get("default_n_bins_default_rate", 10)),
            proba_to_points_neighbors=int(disp.get("proba_to_points_neighbors", 50)),
        ),
        thresholds=ThresholdsConfig(
            default_seuil1=float(t.get("default_seuil1", 0.05)),
            default_seuil2=float(t.get("default_seuil2", 0.15)),
        ),
    )


@st.cache_resource(show_spinner=False)
def _load_config() -> AppConfig:
    return _parse_config(_CONFIG_PATH)


def get_config() -> AppConfig:
    """Retourne la config (depuis le cache après le premier appel)."""
    return _load_config()


# ---------------------------------------------------------------------------
# Helper principal : nom d'affichage d'une variable
# ---------------------------------------------------------------------------
def col_label(var: str) -> str:
    """
    Retourne l'alias d'affichage d'une variable si défini dans config.yaml,
    sinon retourne le nom brut.

    Exemple :
        col_label("anciennete")  →  "Ancienneté (années)"
        col_label("revenus")     →  "revenus"   (si pas d'alias)
    """
    cfg = get_config()
    return cfg.columns.aliases.get(var, var)


# ---------------------------------------------------------------------------
# Raccourci module-level (permet `from utils.config import cfg`)
# ---------------------------------------------------------------------------
# NB: ne pas appeler get_config() à l'import (hors contexte Streamlit en tests).
# Les pages importent `get_config` et `col_label` directement.
