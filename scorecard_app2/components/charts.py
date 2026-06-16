"""Constructeurs de graphiques Plotly avec thème BNP."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl

from utils.theme import (
    BNP_GREEN, BNP_GREEN_DARK, BORDER, DECISION_GREEN, DECISION_RED,
    PLOTLY_LAYOUT, TEXT_PRIMARY, TEXT_SECONDARY,
)


def _apply_layout(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, height=height, title=title)
    return fig


def chart_points_by_bin(sc_var: pl.DataFrame, variable: str) -> go.Figure:
    """
    Barres horizontales des points_1000 par bin pour une variable.
    Vert = points positifs (réduit le risque), rouge = négatifs.
    """
    df = sc_var.sort("points_1000")
    labels = df.get_column("Label").to_list()
    points = df.get_column("points_1000").to_list()
    colors = [DECISION_GREEN if p >= 0 else DECISION_RED for p in points]

    fig = go.Figure(go.Bar(
        x=points,
        y=labels,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{int(p):+d}" for p in points],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Points: %{x:+d}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=BORDER, width=1))
    fig.update_xaxes(title="Points 1000")
    fig.update_yaxes(title=None)
    return _apply_layout(fig, height=max(180, 60 + 32 * len(labels)),
                         title=f"Contribution de {variable}")


def chart_variable_importance(importance_df: pl.DataFrame) -> go.Figure:
    """Barres horizontales de l'étendue (max-min) des points par variable."""
    df = importance_df.sort("importance")
    fig = go.Figure(go.Bar(
        x=df.get_column("importance").to_list(),
        y=df.get_column("Variables").to_list(),
        orientation="h",
        marker=dict(color=BNP_GREEN),
        text=[f"{int(v)}" for v in df.get_column("importance").to_list()],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Étendue points: %{x}<extra></extra>",
    ))
    fig.update_xaxes(title="Étendue des points (max − min)")
    return _apply_layout(fig, height=max(220, 40 + 32 * df.height),
                         title="Importance des variables")


def chart_calibration_quantile(
    df: pl.DataFrame, n_bins: int = 10, target_col: str = "defaut_obs",
) -> tuple[go.Figure, pl.DataFrame]:
    """
    Courbe de calibration par quantiles : pour chaque quantile de proba,
    on compare la moyenne des probas prédites au taux de défaut observé.
    Retourne aussi le tableau lié.
    """
    if target_col not in df.columns or df.is_empty():
        return go.Figure(), pl.DataFrame()

    sub = df.select(["score_proba", target_col]).drop_nulls()
    if sub.is_empty():
        return go.Figure(), pl.DataFrame()

    n_bins = max(2, min(n_bins, sub.height))
    bin_edges = np.quantile(
        sub["score_proba"].to_numpy(),
        np.linspace(0, 1, n_bins + 1),
    )
    bin_edges = np.unique(bin_edges)  # évite quantiles dégénérés
    if len(bin_edges) < 3:
        return go.Figure(), pl.DataFrame()

    bins = (
        sub
        .with_columns(
            pl.col("score_proba")
              .cut(breaks=bin_edges[1:-1].tolist(),
                   labels=[str(i + 1) for i in range(len(bin_edges) - 1)])
              .alias("_bin")
        )
        .group_by("_bin")
        .agg([
            pl.col("score_proba").mean().alias("proba_moyenne"),
            pl.col(target_col).mean().alias("taux_defaut_obs"),
            pl.col(target_col).sum().alias("n_defauts"),
            pl.len().alias("effectif"),
        ])
        .sort("proba_moyenne")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color=BORDER, dash="dash"),
        name="Calibration parfaite", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=bins["proba_moyenne"].to_list(),
        y=bins["taux_defaut_obs"].to_list(),
        mode="lines+markers",
        line=dict(color=BNP_GREEN, width=2),
        marker=dict(size=10, color=BNP_GREEN_DARK),
        name="Observé",
        customdata=np.stack([
            bins["effectif"].to_numpy(),
            bins["n_defauts"].to_numpy(),
        ], axis=-1),
        hovertemplate=(
            "Proba prédite moy.: %{x:.3f}<br>"
            "Taux défaut obs.: %{y:.3f}<br>"
            "Effectif: %{customdata[0]}<br>"
            "Défauts: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig.update_xaxes(title="Probabilité prédite moyenne", range=[0, max(bins["proba_moyenne"].max() * 1.1, 0.1)])
    fig.update_yaxes(title="Taux de défaut observé", range=[0, max(bins["taux_defaut_obs"].max() * 1.1, 0.1)])
    fig = _apply_layout(fig, height=420, title=f"Courbe de calibration ({n_bins} quantiles)")
    return fig, bins


def chart_default_rate_by_score(
    df: pl.DataFrame, n_bins: int = 10, target_col: str = "defaut_obs",
) -> go.Figure:
    """Taux de défaut par tranche de score_points (quantiles)."""
    if target_col not in df.columns or df.is_empty():
        return go.Figure()

    sub = df.select(["score_points", target_col]).drop_nulls()
    if sub.is_empty():
        return go.Figure()

    n_bins = max(2, min(n_bins, sub.height))
    edges = np.quantile(sub["score_points"].to_numpy(),
                        np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return go.Figure()

    bins = (
        sub
        .with_columns(
            pl.col("score_points")
              .cut(breaks=edges[1:-1].tolist(),
                   labels=[str(i + 1) for i in range(len(edges) - 1)])
              .alias("_bin")
        )
        .group_by("_bin")
        .agg([
            pl.col("score_points").mean().alias("score_moyen"),
            pl.col(target_col).mean().alias("taux_defaut"),
            pl.col(target_col).sum().alias("n_defauts"),
            pl.len().alias("effectif"),
        ])
        .sort("score_moyen")
    )

    fig = go.Figure(go.Bar(
        x=bins["score_moyen"].to_list(),
        y=bins["taux_defaut"].to_list(),
        marker=dict(color=BNP_GREEN),
        text=[f"{v:.1%}" for v in bins["taux_defaut"].to_list()],
        textposition="outside",
        customdata=np.stack([
            bins["effectif"].to_numpy(),
            bins["n_defauts"].to_numpy(),
        ], axis=-1),
        hovertemplate=(
            "Score moyen: %{x:.0f}<br>"
            "Taux défaut: %{y:.2%}<br>"
            "Effectif: %{customdata[0]}<br>"
            "Défauts: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig.update_xaxes(title="Score points 1000 (moyenne par tranche)")
    fig.update_yaxes(title="Taux de défaut observé", tickformat=".1%")
    return _apply_layout(fig, height=380, title="Évolution du taux de défaut par score")
