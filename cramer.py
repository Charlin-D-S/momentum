import polars as pl
import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd

def cramers_v(x, y):
    """Calcule le V de Cramér entre deux variables catégorielles."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    k = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * k)) if k > 0 else 0.0


def cramers_v_matrix(df: pl.DataFrame) -> pd.DataFrame:
    """
    Calcule une matrice de corrélation basée sur le V de Cramér
    pour toutes les colonnes catégorielles d'un DataFrame Polars.
    """
    # Conversion en pandas pour faciliter le crosstab
    pdf = df.to_pandas()
    cols = pdf.columns.tolist()
    
    matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                matrix.loc[col1, col2] = 1.0
            else:
                matrix.loc[col1, col2] = cramers_v(pdf[col1], pdf[col2])
    
    return matrix




import json

def extract_thresholds(xgb_model):
    booster = xgb_model.get_booster()
    trees = booster.get_dump(dump_format="json")
    
    thresholds = {}

    for tree_str in trees:
        tree = json.loads(tree_str)
        
        # stump → racine directe
        feature = tree.get("split")
        
        # skip catégoriel
        if "categories" in tree:
            continue
        
        if feature is None:
            continue
        
        threshold = tree.get("split_condition")
        
        if threshold is None:
            continue
        
        thresholds.setdefault(feature, set()).add(threshold)

    # tri final
    thresholds = {f: sorted(list(v)) for f, v in thresholds.items()}
    
    return thresholds


import numpy as np
import pandas as pd

def transform_numeric_column(col, bins):
    """
    col: pandas Series
    bins: sorted thresholds
    """
    # résultat initial
    res = np.digitize(col, bins)
    
    # gestion NA → -1
    res[pd.isna(col)] = -1
    
    return res

def transform_categorical_column(col):
    """
    Remplace NA par 'MISSING'
    """
    return col.astype(object).where(~col.isna(), "MISSING")


def transform_dataframe(df, thresholds):
    """
    df: pandas DataFrame
    thresholds: dict {feature: [bins]}
    """
    df_out = df.copy()

    for col in df.columns:
        
        # cas numérique (présent dans thresholds)
        if col in thresholds:
            df_out[col] = transform_numeric_column(df[col], thresholds[col])
        
        # cas catégoriel (non dans thresholds)
        else:
            df_out[col] = transform_categorical_column(df[col])

    return df_out


def build_bin_description(thresholds):
    """
    Retourne les intervalles lisibles pour debug / scorecard
    """
    bin_desc = {}

    for f, bins in thresholds.items():
        intervals = []
        prev = -np.inf
        
        for t in bins:
            intervals.append((prev, t))
            prev = t
        
        intervals.append((prev, np.inf))
        intervals.append("MISSING")  # bin NA
        
        bin_desc[f] = intervals
    
    return bin_desc


import polars as pl

def impute_missing_values(
    lazyframe: pl.LazyFrame, 
    columns: list[str], 
    fill_value: any
) -> pl.LazyFrame:
    """
    Impute les valeurs manquantes d'une liste de colonnes par une valeur précise.
    
    Parameters
    ----------
    lazyframe : pl.LazyFrame
        Le LazyFrame Polars contenant les données
    columns : list[str]
        Liste des noms de colonnes à imputer
    fill_value : any
        Valeur à utiliser pour remplacer les valeurs manquantes (None/null)
    
    Returns
    -------
    pl.LazyFrame
        Un nouveau LazyFrame avec les valeurs manquantes imputées
    """
    return lazyframe.with_columns(
        [
            pl.col(col).fill_null(fill_value).alias(col) 
            for col in columns
        ]
    )

# Exemple d'utilisation
df = pl.LazyFrame({
    'a': [1, None, 3, 4],
    'b': [None, 2, None, 5],
    'c': ['x', 'y', None, 'z'],
    'd': [10, 20, 30, 40]
})

# Imputer les colonnes 'a' et 'b' avec 0, et 'c' avec 'inconnu'
result = (impute_missing_values(df, ['a', 'b'], 0)
          .pipe(impute_missing_values, ['c'], 'inconnu'))

# Exécuter et afficher le résultat
print(result.collect())
