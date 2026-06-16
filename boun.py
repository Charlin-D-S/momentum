def find_boundary_individuals(
    df: pl.DataFrame,
    threshold_points: int,
    n: int = 10,
    proba_min: float | None = None,
    proba_max: float | None = None,
) -> pl.DataFrame:
    """
    Retourne les n individus dont score_points est le plus proche du seuil,
    filtrés sur la zone [proba_min, proba_max] (bornes optionnelles).
    """
    if df.is_empty():
        return df
    sub = df
    if proba_min is not None:
        sub = sub.filter(pl.col("score_proba") > proba_min)
    if proba_max is not None:
        sub = sub.filter(pl.col("score_proba") <= proba_max)
    if sub.is_empty():
        return sub
    return (
        sub
        .with_columns(
            (pl.col("score_points") - threshold_points).abs().alias("_dist_seuil")
        )
        .sort("_dist_seuil")
        .head(n)
        .drop("_dist_seuil")
    )


def render_profile_card(
    row: dict,
    variables: list[str],
    id_col: str,
    idx: int,
    border_color: str | None = None,    # ← nouveau
) -> None:
    score = int(row.get("score_points", 0))
    proba = row.get("score_proba", 0.0)
    id_val = row.get(id_col, f"#{idx}")
    décomposition = decompose_individual(row, variables)

    title = (
        f"Profil {html_lib.escape(str(id_val))}"
        f"  ·  Score : {score} pts"
        f"  ·  Proba : {proba:.2%}"
    )
    with st.expander(title, expanded=False):
        # Bandeau coloré si une couleur de zone est fournie
        if border_color:
            st.markdown(
                f'<div style="'
                f'border-left:4px solid {border_color};'
                f'padding:4px 0 4px 10px;'
                f'margin-bottom:8px;'
                f'font-size:11px;color:{border_color};font-weight:600;">'
                f'Score {score:+d} pts · Proba {proba:.2%}'
                f'</div>',
                unsafe_allow_html=True,
            )
        html_rows = []
        for d in décomposition:
            label = col_label(d["variable"])
            html_rows.append(
                f'<div class="profile-row">'
                f'  <span class="var">{html_lib.escape(label)}</span>'
                f'  <span class="bin">{html_lib.escape(str(d["bin"]))}</span>'
                f'  {_pts_html(d["points"])}'
                f'</div>'
            )
        html_rows.append(
            f'<div class="profile-row" style="border-top:2px solid #000;margin-top:6px;">'
            f'  <span class="var" style="font-weight:600">TOTAL</span>'
            f'  <span class="bin"></span>'
            f'  <span class="pts-zero" style="font-weight:700;color:#1A1A1A">{score:+d}</span>'
            f'</div>'
        )
        st.markdown("".join(html_rows), unsafe_allow_html=True)



def render_boundary_by_zone(
    segment_df: pl.DataFrame,
    threshold_points: int,
    variables: list[str],
    id_col: str,
    zone_left: dict,
    zone_right: dict,
    n: int = 4,
    title: str = "Profils à la frontière",
) -> None:
    """
    Affiche les profils frontière groupés et colorés par zone, en deux colonnes.

    zone_left / zone_right : dict avec les clés
        label     : str        — ex. "VERT"
        color     : str        — couleur hex
        proba_min : float|None — borne inférieure de proba (exclue)
        proba_max : float|None — borne supérieure de proba (incluse)
    """
    st.markdown(f"### {title}")
    st.caption(
        f"Les {n} individus les plus proches de **{threshold_points} pts** "
        f"de chaque côté de la frontière."
    )

    col_l, col_r = st.columns(2)

    for col, zone in [(col_l, zone_left), (col_r, zone_right)]:
        candidates = find_boundary_individuals(
            segment_df,
            threshold_points=threshold_points,
            n=n,
            proba_min=zone.get("proba_min"),
            proba_max=zone.get("proba_max"),
        )
        with col:
            # En-tête de zone coloré
            st.markdown(
                f'<div style="'
                f'background:{zone["color"]};color:white;'
                f'padding:8px 14px;border-radius:4px;'
                f'font-weight:600;font-size:13px;'
                f'letter-spacing:.5px;margin-bottom:10px;">'
                f'{zone["label"]} — {len(candidates)} profil(s)'
                f'</div>',
                unsafe_allow_html=True,
            )
            if candidates.is_empty():
                st.info("Aucun individu dans cette zone.")
                continue
            for i, row in enumerate(candidates.to_dicts()):
                render_profile_card(
                    row, variables, id_col, idx=i,
                    border_color=zone["color"],
                )


st.markdown("---")

from components.profile_cards import render_boundary_by_zone

n_profiles = st.slider(
    "Nombre de profils par zone et par seuil",
    min_value=2, max_value=10,
    value=cfg.display.n_boundary_profiles // 2,
    step=1,
)

if not tricolore:
    render_boundary_by_zone(
        segment_df=segment,
        threshold_points=seuil1_pts,
        variables=variables,
        id_col=id_col,
        zone_left={
            "label": "VERT",
            "color": DECISION_GREEN,
            "proba_min": None,
            "proba_max": seuil1_proba,
        },
        zone_right={
            "label": "ROUGE",
            "color": DECISION_RED,
            "proba_min": seuil1_proba,
            "proba_max": None,
        },
        n=n_profiles,
        title=f"Profils à la frontière — Seuil 1 ({seuil1_pts} pts · {seuil1_proba:.2%})",
    )

else:
    s1, s2 = sorted([seuil1_proba, seuil2_proba])
    pts1, pts2 = sorted([seuil1_pts, seuil2_pts], reverse=True)

    render_boundary_by_zone(
        segment_df=segment,
        threshold_points=pts1,
        variables=variables,
        id_col=id_col,
        zone_left={
            "label": "VERT",
            "color": DECISION_GREEN,
            "proba_min": None,
            "proba_max": s1,
        },
        zone_right={
            "label": "ORANGE",
            "color": DECISION_ORANGE,
            "proba_min": s1,
            "proba_max": s2,
        },
        n=n_profiles,
        title=f"Profils à la frontière — Seuil 1 ({pts1} pts · {s1:.2%})",
    )

    st.markdown(" ")

    render_boundary_by_zone(
        segment_df=segment,
        threshold_points=pts2,
        variables=variables,
        id_col=id_col,
        zone_left={
            "label": "ORANGE",
            "color": DECISION_ORANGE,
            "proba_min": s1,
            "proba_max": s2,
        },
        zone_right={
            "label": "ROUGE",
            "color": DECISION_RED,
            "proba_min": s2,
            "proba_max": None,
        },
        n=n_profiles,
        title=f"Profils à la frontière — Seuil 2 ({pts2} pts · {s2:.2%})",
    )
