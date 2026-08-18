# Répartition — calculée uniquement sur les lignes hors Total
mask_hors_total = df["feu"] != "Total"

nb_total_par_seg = (
    df[mask_hors_total]
    .groupby("segment")["nb_dossiers"]
    .transform("sum")
)

df.loc[mask_hors_total, "repartition"] = (
    df.loc[mask_hors_total, "nb_dossiers"] / nb_total_par_seg * 100
).round(1)

# Les lignes Total = 100% par définition
df.loc[~mask_hors_total, "repartition"] = 100.0
