"""Tests unitaires du module de configuration."""
from __future__ import annotations

import sys
import textwrap
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Contournement : get_config() utilise st.cache_resource ; en mode test,
# on appelle directement _parse_config() qui n'a pas de dépendance Streamlit.
from utils.config import _parse_config, AppConfig


def green(msg: str) -> None:
    print(f"  \033[92m✓\033[0m {msg}")


def _write_yaml(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tmp.write(textwrap.dedent(content))
    tmp.close()
    return Path(tmp.name)


def test_default_values():
    """Un YAML vide retourne les valeurs par défaut."""
    p = _write_yaml("{}")
    cfg = _parse_config(p)
    assert cfg.columns.id == "id_client"
    assert cfg.columns.target == "defaut_obs"
    assert cfg.columns.aliases == {}
    assert cfg.filters.cols_per_row == 3
    assert cfg.display.n_boundary_profiles == 8
    assert cfg.thresholds.default_seuil1 == 0.05
    green("Valeurs par défaut correctes sur YAML vide.")


def test_aliases_loaded():
    """Les alias sont chargés depuis columns.aliases."""
    p = _write_yaml("""
        columns:
          id: client_id
          target: defaut
          aliases:
            anciennete: "Ancienneté (années)"
            revenus: "Revenus mensuels (€)"
    """)
    cfg = _parse_config(p)
    assert cfg.columns.id == "client_id"
    assert cfg.columns.aliases["anciennete"] == "Ancienneté (années)"
    assert cfg.columns.aliases["revenus"] == "Revenus mensuels (€)"
    green("Aliases chargés correctement depuis le YAML.")


def test_aliases_none_when_commented():
    """Quand aliases est absent (ou commenté), aliases = {} et pas d'erreur."""
    p = _write_yaml("""
        columns:
          id: id_client
          target: defaut_obs
    """)
    cfg = _parse_config(p)
    assert cfg.columns.aliases == {}
    green("Aliases absent → dict vide, pas d'erreur.")


def test_col_label_returns_alias():
    """col_label() retourne l'alias si défini."""
    p = _write_yaml("""
        columns:
          aliases:
            anciennete: "Ancienneté (années)"
    """)
    cfg = _parse_config(p)
    label = cfg.columns.aliases.get("anciennete", "anciennete")
    assert label == "Ancienneté (années)"
    green("col_label() retourne l'alias quand défini.")


def test_col_label_fallback():
    """col_label() retourne le nom brut si aucun alias."""
    p = _write_yaml("{}")
    cfg = _parse_config(p)
    label = cfg.columns.aliases.get("anciennete", "anciennete")
    assert label == "anciennete"
    green("col_label() retourne le nom brut sans alias.")


def test_target_null_handling():
    """target: null dans le YAML → cfg.columns.target is None."""
    p = _write_yaml("""
        columns:
          id: id_client
          target: null
    """)
    cfg = _parse_config(p)
    assert cfg.columns.target is None
    green("target: null → cfg.columns.target is None.")


def test_filter_vars_list():
    p = _write_yaml("""
        filters:
          vars:
            - anciennete
            - type_client
          cols_per_row: 4
    """)
    cfg = _parse_config(p)
    assert cfg.filters.vars == ["anciennete", "type_client"]
    assert cfg.filters.cols_per_row == 4
    green("Liste de filtres et cols_per_row chargés.")


def test_display_and_thresholds():
    p = _write_yaml("""
        display:
          n_boundary_profiles: 12
          default_n_bins_calibration: 15
          proba_to_points_neighbors: 30
        thresholds:
          default_seuil1: 0.08
          default_seuil2: 0.20
    """)
    cfg = _parse_config(p)
    assert cfg.display.n_boundary_profiles == 12
    assert cfg.display.default_n_bins_calibration == 15
    assert cfg.display.proba_to_points_neighbors == 30
    assert cfg.thresholds.default_seuil1 == 0.08
    assert cfg.thresholds.default_seuil2 == 0.20
    green("Paramètres display et thresholds chargés correctement.")


if __name__ == "__main__":
    print("\n=== Tests de configuration ===\n")
    tests = [
        test_default_values,
        test_aliases_loaded,
        test_aliases_none_when_commented,
        test_col_label_returns_alias,
        test_col_label_fallback,
        test_target_null_handling,
        test_filter_vars_list,
        test_display_and_thresholds,
    ]
    for t in tests:
        t()
    print(f"\n✅ {len(tests)} tests passés\n")
