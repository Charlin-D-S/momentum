total = lf.select(pl.len()).collect().item()

exprs = []
for col in lf.columns:
    exprs += [
        pl.col(col).is_null().sum().alias(f"{col}__nb_missing"),
        (pl.col(col).is_null() & (pl.col('defaut') == 1)).sum().alias(f"{col}__nb_def_miss"),
        (pl.col(col).is_not_null() & (pl.col('defaut') == 1)).sum().alias(f"{col}__nb_def_non_miss"),
    ]

stats = lf.select(exprs).collect().to_dict(as_series=False)

results = []
for col in lf.columns:
    nb_miss = stats[f"{col}__nb_missing"][0]
    nb_non_miss = total - nb_miss
    nb_def_miss = stats[f"{col}__nb_def_miss"][0]
    nb_def_non_miss = stats[f"{col}__nb_def_non_miss"][0]

    results.append({
        'Variable': col,
        'NB_Missing': nb_miss,
        'pct_Missing': nb_miss / total,
        'NB_defaut_Missing': nb_def_miss,
        'pct_defaut_Missing': nb_def_miss / nb_miss if nb_miss > 0 else None,
        'NB_defaut_non_Missing': nb_def_non_miss,
        'pct_defaut_non_Missing': nb_def_non_miss / nb_non_miss if nb_non_miss > 0 else None,
    })

df_missing = pd.DataFrame(results)



results = []

for col in lf.columns:
    total = lf.select(pl.len()).collect().item()
    
    nb_missing = lf.filter(pl.col(col).is_null()).select(pl.len()).collect().item()
    pct_missing = nb_missing / total

    nb_defaut_missing = lf.filter(pl.col(col).is_null() & (pl.col('defaut') == 1)).select(pl.len()).collect().item()
    pct_defaut_missing = nb_defaut_missing / nb_missing if nb_missing > 0 else None

    nb_defaut_non_missing = lf.filter(pl.col(col).is_not_null() & (pl.col('defaut') == 1)).select(pl.len()).collect().item()
    nb_non_missing = total - nb_missing
    pct_defaut_non_missing = nb_defaut_non_missing / nb_non_missing if nb_non_missing > 0 else None

    results.append({
        'Variable': col,
        'NB_Missing': nb_missing,
        'pct_Missing': pct_missing,
        'NB_defaut_Missing': nb_defaut_missing,
        'pct_defaut_Missing': pct_defaut_missing,
        'NB_defaut_non_Missing': nb_defaut_non_missing,
        'pct_defaut_non_Missing': pct_defaut_non_missing
    })

df_missing = pd.DataFrame(results)
