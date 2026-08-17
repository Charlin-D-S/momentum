def diag_V_R(
    lf: pl.LazyFrame,
    nv_seuil_vert_coeurcible: float,
    nv_seuil_rouge_coeurcible: float,
    nv_seuil_vert_pro: float,
    nv_seuil_rouge_pro: float,
    # ... ajouter les seuils des autres segments (ER, etc.)
) -> pl.LazyFrame:
    """
    Traduit la logique de feux tricolores en expressions Polars natives.
    Évite le apply/row-wise — vectorisé et compatible LazyFrame.
    """

    # ── Segment : Client PRO Cœur de cible ───────────────────────────────
    est_pro_coeur = (
        (pl.col("flag_coeur_cible") == 1) &
        (pl.col("cat_SEG") == "Grp_PRO")
    )

    feu_pro_coeur = (
        pl.when(pl.col("new_PD") < nv_seuil_vert_coeurcible)
          .then(
              pl.when(pl.col("flag_RM_rouge_hors_seuil") > 0).then(pl.lit("Rouge"))
                .when(pl.col("flag_RM_orange_hors_seuil_VF") > 0).then(pl.lit("Orange"))
                .otherwise(pl.lit("Vert"))
          )
          .when(
              (pl.col("new_PD") >= nv_seuil_vert_coeurcible) &
              (pl.col("new_PD") <  nv_seuil_rouge_coeurcible)
          )
          .then(
              pl.when(pl.col("flag_RM_rouge_hors_seuil") > 0).then(pl.lit("Rouge"))
                .otherwise(pl.lit("Orange"))
          )
          .when(pl.col("new_PD") >= nv_seuil_rouge_coeurcible)
          .then(pl.lit("Rouge"))
          .otherwise(pl.lit(None))  # cas non couvert
    )

    # ── Segment : Client PRO Hors cœur de cible ──────────────────────────
    est_pro_hors = (
        (pl.col("flag_coeur_cible") == 0) &
        (pl.col("cat_SEG") == "Grp_PRO")
    )

    feu_pro_hors = (
        pl.when(pl.col("new_PD") < nv_seuil_vert_pro)
          .then(
              pl.when(pl.col("flag_RM_rouge_hors_seuil") > 0).then(pl.lit("Rouge"))
                .when(pl.col("flag_RM_orange_hors_seuil_VF") > 0).then(pl.lit("Orange"))
                .otherwise(pl.lit("Vert"))
          )
          .when(
              (pl.col("new_PD") >= nv_seuil_vert_pro) &
              (pl.col("new_PD") <  nv_seuil_rouge_pro)
          )
          .then(
              pl.when(pl.col("flag_RM_rouge_hors_seuil") > 0).then(pl.lit("Rouge"))
                .otherwise(pl.lit("Orange"))
          )
          .when(pl.col("new_PD") >= nv_seuil_rouge_pro)
          .then(pl.lit("Rouge"))
          .otherwise(pl.lit(None))
    )

    # ── Combinaison des segments ──────────────────────────────────────────
    # Chaque segment est mutuellement exclusif — on prend le premier non-null
    feu_final = (
        pl.when(est_pro_coeur).then(feu_pro_coeur)
          .when(est_pro_hors) .then(feu_pro_hors)
          # .when(est_ER)      .then(feu_ER)   ← ajouter les autres segments ici
          .otherwise(pl.lit(None))
          .alias("diag_V_R")
    )

    return lf.with_columns(feu_final)


lf_result = diag_V_R(
    lf,
    nv_seuil_vert_coeurcible=0.05,
    nv_seuil_rouge_coeurcible=0.15,
    nv_seuil_vert_pro=0.07,
    nv_seuil_rouge_pro=0.20,
)
