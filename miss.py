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
