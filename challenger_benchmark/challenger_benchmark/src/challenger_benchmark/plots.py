"""Production et export des figures. Moteur unique : matplotlib.

Par modele : ROC train/test, importance SHAP (moyenne des |SHAP|), impact SHAP
(beeswarm). Au niveau racine : comparaison inter-modeles et SHAP consolide.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

_DPI = 150


def _label(var: str, descriptions: dict[str, str] | None) -> str:
    if descriptions and var in descriptions and descriptions[var]:
        return f"{var} — {descriptions[var]}"
    return var


def plot_roc(y_train, p_train, y_test, p_test, auc_train, auc_test, path: Path) -> None:
    from .evaluation import roc_points
    fpr_tr, tpr_tr = roc_points(y_train, p_train)
    fpr_te, tpr_te = roc_points(y_test, p_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_tr, tpr_tr, label=f"Train (AUC = {auc_train:.3f})", lw=1.6)
    ax.plot(fpr_te, tpr_te, label=f"Test (AUC = {auc_test:.3f})", lw=1.6)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def plot_shap_importance(
    importance: pd.Series, path: Path, top_n: int, descriptions=None
) -> None:
    top = importance.head(top_n).iloc[::-1]
    labels = [_label(v, descriptions) for v in top.index]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(top) + 1)))
    ax.barh(labels, top.values, color="#1f77b4")
    ax.set_xlabel("Importance SHAP (moyenne des |valeurs|, agregee par variable)")
    ax.set_title("Importance des variables")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def plot_shap_beeswarm(
    sv: np.ndarray, display: np.ndarray, feat_names, path: Path, top_n: int
) -> None:
    expl = shap.Explanation(values=sv, data=display, feature_names=list(feat_names))
    fig = plt.figure()
    shap.plots.beeswarm(expl, max_display=top_n, show=False)
    plt.title("Impact SHAP (distribution par observation)")
    plt.tight_layout()
    plt.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_optuna_history(study, path: Path) -> None:
    values = [t.value for t in study.trials if t.value is not None]
    best = np.maximum.accumulate(values) if values else []
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(values) + 1), values, "o", ms=3, alpha=0.5, label="Essai")
    ax.plot(range(1, len(best) + 1), best, "-", color="#d62728", label="Meilleur cumule")
    ax.set_xlabel("Essai")
    ax.set_ylabel("AUC (validation croisee)")
    ax.set_title("Historique d'optimisation Optuna")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def plot_comparison(summary: pd.DataFrame, path: Path) -> None:
    df = summary.sort_values("auc_test", ascending=True)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(df) + 1)))
    ax.barh(y - 0.2, df["auc_test"], height=0.4, label="AUC test", color="#1f77b4")
    ax.barh(y + 0.2, df["gini_test"], height=0.4, label="Gini test", color="#ff7f0e")
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("Score (test)")
    ax.set_title("Comparaison des challengers")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def plot_consolidated_shap(
    importances: dict[str, pd.Series], path: Path, top_n: int, descriptions=None
) -> None:
    """Carte de chaleur des rangs SHAP par variable across modeles."""
    frame = pd.DataFrame(importances)
    union = frame.fillna(0).sum(axis=1).sort_values(ascending=False).head(top_n).index
    ranks = frame.rank(ascending=False)
    ranks = ranks.loc[union]
    labels = [_label(v, descriptions) for v in ranks.index]

    fig, ax = plt.subplots(figsize=(1.8 * len(frame.columns) + 3, 0.45 * len(ranks) + 2))
    im = ax.imshow(ranks.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(ranks.columns)))
    ax.set_xticklabels(ranks.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(ranks.index)))
    ax.set_yticklabels(labels)
    for i in range(ranks.shape[0]):
        for j in range(ranks.shape[1]):
            val = ranks.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, int(val), ha="center", va="center", color="white", fontsize=8)
    ax.set_title("Rang SHAP des variables par modele (1 = plus important)")
    fig.colorbar(im, ax=ax, label="Rang")
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
