# Après le collect() et to_pandas() de l'agrégat

# ── Toutes les combinaisons segment × feu possibles ───────────────────────
segments  = agg[COL_SEGMENT].unique()
feux      = ["Vert", "Orange", "Rouge"]

combinaisons = pd.DataFrame([
    {COL_SEGMENT: seg, COL_FEU: feu}
    for seg in segments
    for feu in feux
])

# ── Left join pour insérer les zéros sur les combinaisons manquantes ───────
cols_numeriques = ["nb_dossiers", "nb_defaut", "montant_total", "mtn_defaut"]

agg = (
    combinaisons
    .merge(agg, on=[COL_SEGMENT, COL_FEU], how="left")
    .fillna({col: 0 for col in cols_numeriques})
)

# Remettre les bons types
agg[cols_numeriques] = agg[cols_numeriques].astype(int)
