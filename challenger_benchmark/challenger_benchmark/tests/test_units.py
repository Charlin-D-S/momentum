"""Tests unitaires : config, resolution des variables, types, agregation SHAP, metriques."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from challenger_benchmark.config import _parse_config, ALL_MODELS
from challenger_benchmark import data as data_mod
from challenger_benchmark.evaluation import gini, ks_statistic, evaluate
from challenger_benchmark.explain import variable_importance


def _raw():
    return {
        "data": {"path": "x.parquet", "target": "y"},
        "drivers": {"path": "d.xlsx"},
        "category_mapping": {"path": "m.json"},
        "output_dir": "out",
        "variables": "all",
    }


def test_parse_defaults():
    cfg = _parse_config(_raw())
    assert cfg.use_all_variables
    assert cfg.tuning.n_trials == 50
    assert cfg.models == tuple(ALL_MODELS)
    assert cfg.data.train_values == ("train", "val")


def test_parse_variable_list():
    raw = _raw()
    raw["variables"] = ["a", "b"]
    cfg = _parse_config(raw)
    assert not cfg.use_all_variables
    assert cfg.variables == ("a", "b")


def test_parse_unknown_model_raises():
    raw = _raw()
    raw["models"] = ["xgboost", "lightgbm"]
    with pytest.raises(ValueError):
        _parse_config(raw)


def test_missing_key_raises():
    raw = _raw()
    del raw["output_dir"]
    with pytest.raises(KeyError):
        _parse_config(raw)


def _drivers_frame():
    return pl.DataFrame({
        "Variable": ["a", "b", "c", "d"],
        "Description": ["A", "B", "C", "D"],
        "Type": ["q", "c", "q", "c"],
        "Risk_drivers": ["RISK_DRIVER", "RISK_DRIVER", "NO", "RISK_DRIVER"],
    })


def test_resolve_all_uses_risk_driver_flag():
    cfg = _parse_config(_raw())
    got = data_mod.resolve_variables(cfg, _drivers_frame())
    assert got == ["a", "b", "d"]


def test_resolve_list_checks_membership():
    raw = _raw(); raw["variables"] = ["a", "z"]
    cfg = _parse_config(raw)
    with pytest.raises(ValueError):
        data_mod.resolve_variables(cfg, _drivers_frame())


def test_split_feature_types():
    cfg = _parse_config(_raw())
    num, cat = data_mod.split_feature_types(cfg, _drivers_frame(), ["a", "b", "d"])
    assert num == ["a"]
    assert cat == ["b", "d"]


def test_decode_categoricals_nested_mapping():
    lf = pl.DataFrame({"b": [1, 2, None]}).lazy()
    out = data_mod._decode_categoricals(lf, {"b": {"1": "oui", "2": "non"}}, ["b"]).collect()
    assert out["b"].to_list() == ["oui", "non", None]


def test_gini_and_ks_bounds():
    assert gini(0.5) == 0.0
    assert gini(1.0) == 1.0
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert ks_statistic(y, perfect) == pytest.approx(1.0)


def test_evaluate_gap():
    y = pd.Series([0, 1, 0, 1, 0, 1])
    p_tr = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85])
    p_te = np.array([0.6, 0.4, 0.45, 0.55, 0.4, 0.6])  # une paire inversee
    m = evaluate("x", y, p_tr, y, p_te)
    assert m.train.auc > m.test.auc
    assert m.auc_gap == pytest.approx(m.train.auc - m.test.auc)


def test_variable_importance_aggregates_onehot():
    sv = np.array([[1.0, -2.0, 0.5], [3.0, 2.0, -0.5]])
    feat_names = ["cat__v_a", "cat__v_b", "num__w"]
    col_to_var = {"cat__v_a": "v", "cat__v_b": "v", "num__w": "w"}
    imp = variable_importance(sv, feat_names, col_to_var)
    # v : moyenne|.|=2 + 2 = 4 ; w : 0.5
    assert imp["v"] == pytest.approx(4.0)
    assert imp["w"] == pytest.approx(0.5)
    assert imp.index[0] == "v"
