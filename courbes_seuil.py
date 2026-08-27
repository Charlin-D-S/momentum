"""
Courbes Plotly pour le choix du seuil d'octroi (à afficher dans Streamlit).

Trois graphiques, tous construits à partir de la SEULE table renvoyée par
table_seuils() — aucun recalcul sur les données individuelles :

  1. courbe_risque      : taux de défaut des acceptés en fonction du seuil
  2. courbe_volume      : taux d'acceptation en fonction du seuil
  3. courbe_strategie   : taux d'acceptation en fonction du taux de défaut
                          accepté (la "strategy curve" : ce que coûte en volume
                          chaque point de risque en moins)

Chaque figure accepte un objet Seuil pour matérialiser le point retenu.
"""

import plotly.graph_objects as go
import polars as pl

from seuil_octroi import (Seuil, seuil_pour_acceptation, seuil_pour_defaut,
                          table_seuils)

GRILLE = dict(showgrid=True, gridcolor="rgba(128,128,128,.18)", zeroline=False)
BLEU, ORANGE, VERT = "#2c6fbb", "#e08a1e", "#1a7f37"


def _mise_en_forme(fig, titre, x, y, pct_x=False, pct_y=True):
    fig.update_layout(
        title=titre,
        xaxis_title=x,
        yaxis_title=y,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
        template="plotly_white",
        showlegend=False,
    )
    fig.update_xaxes(**GRILLE, tickformat=".0%" if pct_x else None)
    fig.update_yaxes(**GRILLE, tickformat=".1%" if pct_y else None)
    return fig


def _point(fig, x, y, texte):
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="markers+text", text=[texte], textposition="top left",
        marker=dict(size=11, color="#c0392b", symbol="diamond"),
        hovertemplate=texte + "<extra></extra>",
    ))


def courbe_risque(t: pl.DataFrame, s: Seuil | None = None,
                  lisse: bool = True, cible: float | None = None) -> go.Figure:
    """Taux de défaut des acceptés (axe y) selon le seuil (axe x)."""
    y = t["taux_defaut"].to_list()
    fig = go.Figure(go.Scatter(
        x=t["proba"].to_list(), y=y, mode="lines", line=dict(color=BLEU, width=2),
        name="taux de défaut", customdata=t["n_cum"].to_list(),
        hovertemplate="seuil %{x:.4f}<br>défaut acceptés %{y:.2%}"
                      "<br>%{customdata:,.0f} acceptés<extra></extra>",
    ))
    if lisse:
        fig.add_trace(go.Scatter(
            x=t["proba"].to_list(),
            y=t.select(pl.col("taux_defaut").cum_max())["taux_defaut"].to_list(),
            mode="lines", line=dict(color=BLEU, width=1, dash="dot"),
            hoverinfo="skip",
        ))
    if cible is not None:
        fig.add_hline(y=cible, line=dict(color="#c0392b", dash="dash", width=1),
                      annotation_text=f"cible {cible:.1%}",
                      annotation_position="top left")
    if s and s.atteint:
        fig.add_vline(x=s.seuil, line=dict(color="#c0392b", dash="dash", width=1))
        _point(fig, s.seuil, s.taux_defaut, f"{s.taux_defaut:.2%}")
    return _mise_en_forme(fig, "Risque des acceptés selon le seuil",
                          "seuil de probabilité", "taux de défaut des acceptés")


def courbe_volume(t: pl.DataFrame, s: Seuil | None = None,
                  cible: float | None = None) -> go.Figure:
    """Proportion d'acceptés (axe y) selon le seuil (axe x)."""
    fig = go.Figure(go.Scatter(
        x=t["proba"].to_list(), y=t["taux_acceptation"].to_list(),
        mode="lines", line=dict(color=ORANGE, width=2),
        customdata=t["n_cum"].to_list(),
        hovertemplate="seuil %{x:.4f}<br>acceptés %{y:.2%}"
                      "<br>%{customdata:,.0f} dossiers<extra></extra>",
    ))
    if cible is not None:
        fig.add_hline(y=cible, line=dict(color="#c0392b", dash="dash", width=1),
                      annotation_text=f"cible {cible:.1%}",
                      annotation_position="top left")
    if s and s.atteint:
        fig.add_vline(x=s.seuil, line=dict(color="#c0392b", dash="dash", width=1))
        _point(fig, s.seuil, s.taux_acceptation, f"{s.taux_acceptation:.1%}")
    return _mise_en_forme(fig, "Volume accepté selon le seuil",
                          "seuil de probabilité", "part d'acceptés")


def courbe_risque_volume(
    t: pl.DataFrame,
    s: Seuil | None = None,
    cible_risque: float | None = None,
    cible_volume: float | None = None,
    lisse: bool = True,
) -> go.Figure:
    """
    Les deux lectures sur un seul graphique, en fonction du seuil :
    taux de défaut des acceptés (axe gauche) et part d'acceptés (axe droit).
    """
    x = t["proba"].to_list()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=t["taux_defaut"].to_list(), name="défaut des acceptés",
        mode="lines", line=dict(color=BLEU, width=2), yaxis="y",
        customdata=t["n_cum"].to_list(),
        hovertemplate="défaut acceptés %{y:.2%}"
                      "<br>%{customdata:,.0f} dossiers<extra></extra>",
    ))
    if lisse:
        fig.add_trace(go.Scatter(
            x=x,
            y=t.select(pl.col("taux_defaut").cum_max())["taux_defaut"].to_list(),
            name="défaut (lissé)", mode="lines", yaxis="y",
            line=dict(color=BLEU, width=1, dash="dot"), hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=x, y=t["taux_acceptation"].to_list(), name="part d'acceptés",
        mode="lines", line=dict(color=ORANGE, width=2), yaxis="y2",
        hovertemplate="acceptés %{y:.2%}<extra></extra>",
    ))

    if cible_risque is not None:
        fig.add_hline(y=cible_risque, yref="y",
                      line=dict(color=BLEU, dash="dash", width=1),
                      annotation_text=f"cible risque {cible_risque:.1%}",
                      annotation_position="top left",
                      annotation_font=dict(color=BLEU))
    if cible_volume is not None:
        fig.add_hline(y=cible_volume, yref="y2",
                      line=dict(color=ORANGE, dash="dash", width=1),
                      annotation_text=f"cible volume {cible_volume:.0%}",
                      annotation_position="bottom right",
                      annotation_font=dict(color=ORANGE))
    if s and s.atteint:
        fig.add_vline(x=s.seuil, line=dict(color="#c0392b", dash="dash", width=1),
                      annotation_text=f"seuil {s.seuil:.4f}",
                      annotation_position="top right")

    fig.update_layout(
        title="Risque et volume des acceptés selon le seuil",
        xaxis=dict(title="seuil de probabilité", **GRILLE),
        yaxis=dict(title=dict(text="taux de défaut des acceptés",
                              font=dict(color=BLEU)),
                   tickformat=".1%", tickfont=dict(color=BLEU), **GRILLE),
        yaxis2=dict(title=dict(text="part d'acceptés", font=dict(color=ORANGE)),
                    tickformat=".0%", tickfont=dict(color=ORANGE),
                    overlaying="y", side="right", showgrid=False,
                    rangemode="tozero"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.14, x=0),
        margin=dict(l=10, r=10, t=70, b=10),
        height=430,
        template="plotly_white",
    )
    return fig


def courbe_strategie(t: pl.DataFrame, s: Seuil | None = None) -> go.Figure:
    """
    Taux d'acceptation en fonction du taux de défaut des acceptés.
    Lue de gauche à droite : combien de volume gagne-t-on en tolérant un point
    de risque supplémentaire.
    """
    tt = t.filter(pl.col("n_cum") > 0)
    fig = go.Figure(go.Scatter(
        x=tt["taux_defaut"].to_list(), y=tt["taux_acceptation"].to_list(),
        mode="lines", line=dict(color=VERT, width=2),
        customdata=tt.select(["proba", "n_cum"]).to_numpy(),
        hovertemplate="défaut acceptés %{x:.2%}<br>acceptés %{y:.2%}"
                      "<br>seuil %{customdata[0]:.4f}"
                      "<br>%{customdata[1]:,.0f} dossiers<extra></extra>",
    ))
    if s and s.atteint:
        _point(fig, s.taux_defaut, s.taux_acceptation,
               f"seuil {s.seuil:.4f}")
    return _mise_en_forme(fig, "Arbitrage volume / risque",
                          "taux de défaut des acceptés", "part d'acceptés",
                          pct_x=True)


# --------------------------------------------------------- page Streamlit type
def page_seuil(st, df, proba="proba", cible="defaut", poids=None):
    """
    Bloc Streamlit complet. La courbe est calculée UNE fois puis réutilisée
    par les trois graphiques et par la recherche de seuil.
    """

    @st.cache_data(show_spinner=False)
    def _table(_df, proba, cible, poids):
        return table_seuils(_df, proba, cible, poids)

    t = _table(df, proba, cible, poids)

    mode = st.radio("Piloter par", ["taux de défaut", "taux d'acceptation"],
                    horizontal=True)
    if mode == "taux de défaut":
        c = st.slider("Taux de défaut maximal des acceptés", 0.0, 0.30, 0.03, 0.005,
                      format="%.1f%%")
        n_min = st.number_input("Volume minimal d'acceptés", 0, value=0, step=50)
        s = seuil_pour_defaut(None, c, n_min=n_min, table=t)
        cible_risque, cible_volume = c, None
    else:
        c = st.slider("Part d'acceptés maximale", 0.0, 1.0, 0.60, 0.01,
                      format="%.0f%%")
        s = seuil_pour_acceptation(None, c, table=t)
        cible_risque, cible_volume = None, c

    k1, k2, k3 = st.columns(3)
    k1.metric("Seuil", "—" if not s.atteint else f"{s.seuil:.4f}")
    k2.metric("Acceptés", f"{s.n_acceptes:,}".replace(",", " "),
              f"{s.taux_acceptation:.1%}")
    k3.metric("Défaut des acceptés", f"{s.taux_defaut:.2%}")

    st.plotly_chart(
        courbe_risque_volume(t, s, cible_risque=cible_risque,
                             cible_volume=cible_volume),
        use_container_width=True)
    st.plotly_chart(courbe_strategie(t, s), use_container_width=True)

    with st.expander("Voir les deux courbes séparément"):
        st.plotly_chart(courbe_risque(t, s, cible=cible_risque),
                        use_container_width=True)
        st.plotly_chart(courbe_volume(t, s, cible=cible_volume),
                        use_container_width=True)
    return s, t


if __name__ == "__main__":
    import time

    import numpy as np

    rng = np.random.default_rng(0)

    for n, k in [(100_000, 1000), (1_000_000, 1000), (1_000_000, 100)]:
        score = rng.integers(0, k, n)                 # score discret
        p = np.clip((score / k) ** 2 * .4 + .002, 0, 1)
        df = pl.DataFrame({"proba": np.round(p, 6), "defaut": rng.binomial(1, p)})

        t0 = time.perf_counter()
        t = table_seuils(df)
        t1 = time.perf_counter()
        s = seuil_pour_defaut(None, 0.05, table=t)
        t2 = time.perf_counter()
        figs = [courbe_risque(t, s), courbe_volume(t, s), courbe_strategie(t, s)]
        t3 = time.perf_counter()
        print(f"n={n:>9,} niveaux={t.height:>5} | table {1e3*(t1-t0):6.1f} ms"
              f" | seuil {1e3*(t2-t1):5.2f} ms | 3 figures {1e3*(t3-t2):5.1f} ms"
              f" | {s.seuil:.4f}")
