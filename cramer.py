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
