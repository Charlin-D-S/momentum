"""Benchmark de challengers non lineaires pour la borne de performance (5.1bis)."""
import os as _os

# Doit s'executer avant tout import declenchant joblib/sklearn (HistGB,
# XGBoost, etc.). Sur certains postes Windows verrouilles (politique de
# securite, EDR), la detection du nombre de coeurs *physiques* par joblib
# echoue ("Acces refuse") puis se rabat sur une commande 'wmic' en
# sous-processus qui peut rester bloquee indefiniment (wmic deprecie ou
# intercepte). Fixer LOKY_MAX_CPU_COUNT a une valeur strictement inferieure
# au nombre de coeurs logiques fait sauter cette detection : voir
# joblib.externals.loky.backend.context.cpu_count, le chemin "coeurs
# physiques" (et donc wmic) n'est tente que si la limite utilisateur n'est
# pas deja plus stricte que le nombre de coeurs logiques.
if "LOKY_MAX_CPU_COUNT" not in _os.environ:
    _logical = _os.cpu_count() or 1
    if _logical > 1:
        _os.environ["LOKY_MAX_CPU_COUNT"] = str(_logical - 1)

from .config import Config, load_config, _parse_config
from .pipeline import run, run_with_config

__all__ = ["Config", "load_config", "_parse_config", "run", "run_with_config"]
