# ── Noms des colonnes — à renseigner ─────────────────────────────────────
COL_SEGMENT = "cat_SEG"        # segment métier (Pro Cœur de cible, Pro autres, ER)
COL_FEU     = "diag_V_R"       # feu tricolore (Vert, Orange, Rouge)
COL_DEFAUT  = "defaut"         # binaire (0/1)
COL_MONTANT = "montant"        # montant en euros par dossier
COL_PD      = "score_proba"    # probabilité de défaut calibrée


@st.cache_data
def charger_données(_lf: pl.LazyFrame) -> pd.DataFrame:
    """
    Construit le DataFrame agrégé (segment × feu) depuis le LazyFrame.
    Calcule les totaux par segment et le total global.
    """
    # ── Agrégation segment × feu ─────────────────────────────────────────
    agg = (
        _lf
        .group_by([COL_SEGMENT, COL_FEU])
        .agg([
            pl.len()                    .alias("nb_dossiers"),
            pl.col(COL_DEFAUT).sum()    .alias("nb_defaut"),
            pl.col(COL_MONTANT).sum()   .alias("montant_total"),
            pl.col(COL_MONTANT)
              .filter(pl.col(COL_DEFAUT) == 1)
              .sum()                    .alias("mtn_defaut"),
        ])
        .collect()
        .to_pandas()
    )

    # ── Totaux par segment ────────────────────────────────────────────────
    totaux_seg = (
        agg
        .groupby(COL_SEGMENT)
        .agg(
            nb_dossiers  =("nb_dossiers",   "sum"),
            nb_defaut    =("nb_defaut",      "sum"),
            montant_total=("montant_total",  "sum"),
            mtn_defaut   =("mtn_defaut",     "sum"),
        )
        .reset_index()
    )
    totaux_seg[COL_FEU] = "Total"

    # ── Total global ──────────────────────────────────────────────────────
    total_global = pd.DataFrame([{
        COL_SEGMENT:   "Total",
        COL_FEU:       feu,
        "nb_dossiers": agg[agg[COL_FEU] == feu]["nb_dossiers"].sum(),
        "nb_defaut":   agg[agg[COL_FEU] == feu]["nb_defaut"].sum(),
        "montant_total": agg[agg[COL_FEU] == feu]["montant_total"].sum(),
        "mtn_defaut":  agg[agg[COL_FEU] == feu]["mtn_defaut"].sum(),
    } for feu in ["Vert", "Orange", "Rouge"]])

    total_global_tot = pd.DataFrame([{
        COL_SEGMENT:   "Total",
        COL_FEU:       "Total",
        "nb_dossiers": agg["nb_dossiers"].sum(),
        "nb_defaut":   agg["nb_defaut"].sum(),
        "montant_total": agg["montant_total"].sum(),
        "mtn_defaut":  agg["mtn_defaut"].sum(),
    }])

    # ── Assemblage ────────────────────────────────────────────────────────
    df = pd.concat([agg, totaux_seg, total_global, total_global_tot], ignore_index=True)

    # ── Métriques calculées ───────────────────────────────────────────────
    # Répartition % au sein de chaque segment
    nb_total_par_seg = df.groupby(COL_SEGMENT)["nb_dossiers"].transform("sum")
    df["repartition"]     = (df["nb_dossiers"] / nb_total_par_seg * 100).round(1)

    # Taux de défaut
    df["tx_defaut"]       = (df["nb_defaut"] / df["nb_dossiers"] * 100).round(3)

    # Montant en m€
    df["montant"]         = (df["montant_total"] / 1e6).round(1)
    df["mtn_defaut_m"]    = (df["mtn_defaut"]    / 1e6).round(1)

    # Taux montant défaut
    df["mtn_defaut_pct"]  = (df["mtn_defaut"] / df["montant_total"] * 100).round(3)

    # Renommage pour compatibilité avec le tableau Streamlit
    df = df.rename(columns={
        COL_SEGMENT:  "segment",
        COL_FEU:      "feu",
        "mtn_defaut_m": "mtn_defaut",
    })

    return df.drop(columns=["montant_total"])
