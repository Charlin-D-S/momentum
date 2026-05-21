def extraire_règles_feuilles(model: lgb.Booster, df_lgb: pd.DataFrame) -> dict[int, str]:
    """
    Extrait les règles de segmentation en langage naturel pour chaque feuille.
    Gère correctement les splits numériques ET catégoriels.
    """
    tree_df = model.trees_to_dataframe()
    tree_df = tree_df[tree_df["tree_index"] == 0].copy()

    # Mapping feature → catégories (pour reconstruire les labels depuis les indices)
    # LightGBM encode les catégories en entiers — on récupère le mapping depuis le df original
    cat_mappings = {}
    for col in df_lgb.select_dtypes(include="category").columns:
        cat_mappings[col] = dict(enumerate(df_lgb[col].cat.categories))

    def est_numérique(seuil) -> bool:
        try:
            float(seuil)
            return True
        except (ValueError, TypeError):
            return False

    def décoder_cats(feature: str, seuil_str: str, est_gauche: bool) -> str:
        """Décode les indices catégoriels LightGBM en labels lisibles."""
        indices_gauche = [int(i) for i in str(seuil_str).split("||")]
        mapping = cat_mappings.get(feature, {})
        labels_gauche = [str(mapping.get(i, i)) for i in indices_gauche]
        if est_gauche:
            return f"{feature} ∈ {{{', '.join(labels_gauche)}}}"
        else:
            return f"{feature} ∉ {{{', '.join(labels_gauche)}}}"

    def chemin_vers_feuille(leaf_id: str) -> list[str]:
        conditions = []
        nœud_courant = leaf_id

        while True:
            parent = tree_df[
                (tree_df["left_child"] == nœud_courant) |
                (tree_df["right_child"] == nœud_courant)
            ]
            if parent.empty:
                break

            parent     = parent.iloc[0]
            feature    = parent["split_feature"]
            seuil      = parent["threshold"]
            est_gauche = parent["left_child"] == nœud_courant

            if est_numérique(seuil):
                # Split numérique
                seuil_f = float(seuil)
                if est_gauche:
                    conditions.append(f"{feature} <= {seuil_f:.4g}")
                else:
                    conditions.append(f"{feature} > {seuil_f:.4g}")
            else:
                # Split catégoriel
                conditions.append(décoder_cats(feature, seuil, est_gauche))

            nœud_courant = parent["node_index"]

        conditions.reverse()
        return conditions

    feuilles = tree_df[tree_df["node_index"].str.startswith("0-L")]
    règles   = {}
    for _, feuille in feuilles.iterrows():
        leaf_id = feuille["node_index"]
        chemin  = chemin_vers_feuille(leaf_id)
        règles[leaf_id] = " ET ".join(chemin) if chemin else "Population totale"

    return règles
    
    
    import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy import stats


def extraire_règles_feuilles(model: lgb.Booster, feature_names: list[str]) -> dict[int, str]:
    """
    Extrait les règles de segmentation en langage naturel pour chaque feuille
    d'un arbre LightGBM unique (n_estimators=1).
    Retourne un dict {leaf_id: règle_textuelle}.
    """
    tree_df = model.trees_to_dataframe()
    tree_df = tree_df[tree_df["tree_index"] == 0].copy()

    # Index des nœuds par node_depth
    nœuds = tree_df.set_index("node_index")

    def chemin_vers_feuille(leaf_id: str) -> list[str]:
        """Remonte l'arbre depuis une feuille jusqu'à la racine."""
        conditions = []
        nœud_courant = leaf_id

        while True:
            # Trouver le parent de ce nœud
            parent = tree_df[
                (tree_df["left_child"] == nœud_courant) |
                (tree_df["right_child"] == nœud_courant)
            ]
            if parent.empty:
                break

            parent = parent.iloc[0]
            feature  = parent["split_feature"]
            seuil    = parent["threshold"]
            est_gauche = parent["left_child"] == nœud_courant

            # Règle textuelle selon le type de split
            if str(seuil).replace(".", "").replace("-", "").isdigit() or \
               isinstance(seuil, (int, float)):
                # Split numérique
                if est_gauche:
                    conditions.append(f"{feature} <= {seuil:.4g}")
                else:
                    conditions.append(f"{feature} > {seuil:.4g}")
            else:
                # Split catégoriel : LightGBM encode "cat1||cat2" pour gauche
                cats_gauche = str(seuil).split("||")
                if est_gauche:
                    conditions.append(f"{feature} ∈ {{{', '.join(cats_gauche)}}}")
                else:
                    conditions.append(f"{feature} ∉ {{{', '.join(cats_gauche)}}}")

            nœud_courant = parent["node_index"]

        conditions.reverse()
        return conditions

    feuilles = tree_df[tree_df["node_index"].str.startswith("0-L")]
    règles = {}
    for _, feuille in feuilles.iterrows():
        leaf_id = feuille["node_index"]
        chemin  = chemin_vers_feuille(leaf_id)
        règles[leaf_id] = " ET ".join(chemin) if chemin else "Population totale"

    return règles


def diagnostiquer_calibration(
    df: pd.DataFrame,
    col_cible: str,
    col_proba: str,
    features: list[str],
    cat_features: list[str] | None = None,
    taille_min_feuille: float = 0.05,
    alpha: float = 0.05,
    max_leaves: int = 16,
) -> pd.DataFrame:
    """
    Identifie les segments où le modèle sur-estime ou sous-estime
    via un arbre LightGBM unique entraîné sur les résidus.

    Paramètres
    ----------
    df                 : DataFrame pandas avec cible, proba et features.
    col_cible          : Colonne binaire (0/1) — défaut observé.
    col_proba          : Colonne probabilité prédite.
    features           : Liste de toutes les variables de segmentation.
    cat_features       : Liste des variables catégorielles (None = aucune).
    taille_min_feuille : Taille minimale d'une feuille en proportion (défaut 5%).
    alpha              : Seuil de significativité du test Z (défaut 0.05).
    max_leaves         : Nombre maximum de feuilles de l'arbre (défaut 16).

    Retourne
    --------
    DataFrame avec une ligne par feuille et les colonnes :
        feuille, règle, effectif, pct_population,
        taux_defaut, pd_moyenne, z_stat, p_value, statut
    """
    n_total = len(df)
    min_data_leaf = max(1, int(taille_min_feuille * n_total))

    # Résidu = proba prédite - défaut observé
    résidu = df[col_proba] - df[col_cible]

    # Préparation du dataset LightGBM
    cat_features = cat_features or []

    # Conversion des catégorielles en dtype category si nécessaire
    df_lgb = df[features].copy()
    for col in cat_features:
        df_lgb[col] = df_lgb[col].astype("category")

    dataset = lgb.Dataset(
        data=df_lgb,
        label=résidu,
        categorical_feature=cat_features if cat_features else "auto",
        free_raw_data=False,
    )

    # Arbre unique — régression sur les résidus
    params = {
        "objective":        "regression",
        "num_leaves":       max_leaves,
        "min_data_in_leaf": min_data_leaf,
        "n_estimators":     1,
        "num_iterations":   1,
        "learning_rate":    1.0,
        "verbose":          -1,
        "force_row_wise":   True,
    }

    model = lgb.train(
        params=params,
        train_set=dataset,
        num_boost_round=1,
    )

    # Affectation de chaque individu à une feuille
    feuille_ids = model.predict(df_lgb, pred_leaf=True).flatten()
    df_analyse  = df[[col_cible, col_proba]].copy()
    df_analyse["feuille_id"] = feuille_ids

    # Extraction des règles
    feature_names = model.feature_name()
    règles_dict   = extraire_règles_feuilles(model, feature_names)

    # Calcul des métriques et test Z par feuille
    résultats = []
    for feuille_id, groupe in df_analyse.groupby("feuille_id"):
        n          = len(groupe)
        nb_def     = groupe[col_cible].sum()
        taux_def   = nb_def / n
        pd_moy     = groupe[col_proba].mean()

        # Test Z de proportion : H0 = PD moyenne == taux défaut observé
        # Z = (PD_moy - taux_def) / sqrt(PD_moy * (1 - PD_moy) / n)
        denom  = np.sqrt(pd_moy * (1 - pd_moy) / n)
        z_stat = (pd_moy - taux_def) / denom if denom > 0 else 0.0
        p_val  = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # bilatéral

        # Statut
        if p_val >= alpha:
            statut = "✅ Bien calibré"
        elif z_stat > 0:
            statut = "🔴 Sur-estimation"
        else:
            statut = "🟡 Sous-estimation"

        # Récupération de la règle textuelle
        # LightGBM numérote les feuilles — on cherche la feuille correspondante
        leaf_key = f"0-L{feuille_id}"
        règle    = règles_dict.get(leaf_key, f"Feuille {feuille_id}")

        résultats.append({
            "feuille":       feuille_id,
            "règle":         règle,
            "effectif":      n,
            "pct_population": round(n / n_total * 100, 1),
            "taux_defaut":   round(taux_def, 4),
            "pd_moyenne":    round(pd_moy,   4),
            "z_stat":        round(z_stat,   3),
            "p_value":       round(p_val,    4),
            "statut":        statut,
        })

    return (
        pd.DataFrame(résultats)
          .sort_values("statut", ascending=True)
          .reset_index(drop=True)
    )



# df_scored = ton DataFrame pandas avec cible, proba et features
résultat = diagnostiquer_calibration(
    df=df_scored,
    col_cible="defaut",
    col_proba="score_proba",
    features=["age", "revenu", "secteur", "anciennete", "encours"],
    cat_features=["secteur"],
    taille_min_feuille=0.05,
    alpha=0.05,
    max_leaves=16,
)

print(résultat)
