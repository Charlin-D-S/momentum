from tirage_panel import tirage_annuel, controler, repartition_mensuelle

ech = tirage_annuel(df, n_par_an=30_000, seed=42)   # ou taux_individus=0.5
print(repartition_mensuelle(ech))                   # 2500 par mois, 12 mois
