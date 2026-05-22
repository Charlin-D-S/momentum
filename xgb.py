def appliquer_mapping(
    lf: pl.LazyFrame,
    mapping: dict[str, dict],
) -> pl.LazyFrame:
    """
    Remplace les valeurs numériques d'un LazyFrame par leurs modalités
    via un mapping {variable: {valeur: modalité}}.
    """
    exprs = [
        pl.col(var)
          .cast(pl.Utf8)  # cast pour éviter les problèmes de type
          .replace(
              {str(k): str(v) for k, v in modalités.items()}
          )
          .alias(var)
        for var, modalités in mapping.items()
        if var in lf.columns
    ]
    return lf.with_columns(exprs)
    
    
    import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from scipy import stats


def préparer_features_xgb(
    df: pd.DataFrame,
    features: list[str],
    cat_features: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Prépare les features pour XGBoost :
    - One-hot encoding des catégorielles
    - Conserve les numériques telles quelles
    - Retourne le DataFrame transformé et le mapping dummy → variable originale

    Retourne
    --------
    df_xgb    : DataFrame avec dummies
    mapping   : {dummy_col: variable_originale} pour reconstituer l'importance par variable
    """
    cat_features = cat_features or []
    num_features = [f for f in features if f not in cat_features]

    # One-hot encoding
    df_cat = pd.DataFrame()
    mapping = {}

    for col in cat_features:
        dummies = pd.get_dummies(
            df[col].astype(str).fillna("MISSING"),
            prefix=col,
            drop_first=False,
            dtype=float,
        )
        for dummy_col in dummies.columns:
            mapping[dummy_col] = col
        df_cat = pd.concat([df_cat, dummies], axis=1)

    # Numériques
    df_num = df[num_features].copy()
    for col in num_features:
        mapping[col] = col

    df_xgb = pd.concat([df_num, df_cat], axis=1)
    return df_xgb, mapping


def extraire_règles_feuilles_xgb(
    model: xgb.Booster,
    df_xgb: pd.DataFrame,
    mapping: dict,
) -> dict[int, str]:
    """
    Extrait les règles de segmentation en langage naturel depuis un arbre XGBoost unique.
    Regroupe les dummies par variable originale dans les règles.
    """
    tree_df = model.trees_to_dataframe()
    tree_df = tree_df[tree_df["Tree"] == 0].copy()

    # Index des nœuds
    nœuds = tree_df.set_index("ID")

    def chemin_vers_feuille(leaf_id: str) -> list[str]:
        conditions = []
        nœud_courant = leaf_id

        while True:
            parent = tree_df[
                (tree_df["Yes"] == nœud_courant) |
                (tree_df["No"]  == nœud_courant)
            ]
            if parent.empty:
                break

            parent     = parent.iloc[0]
            feature    = parent["Feature"]
            seuil      = float(parent["Split"])
            est_yes    = parent["Yes"] == nœud_courant  # Yes = condition vraie (<=)

            var_orig = mapping.get(feature, feature)

            # Dummy binaire (0/1) — reformuler en langage naturel
            if feature != var_orig:
                # C'est une dummy — extraire la modalité depuis le nom "variable_modalite"
                modalité = feature[len(var_orig) + 1:]  # retire le préfixe "var_"
                if est_yes:
                    # dummy <= 0.5 → dummy == 0 → modalité absente
                    conditions.append(f"{var_orig} ≠ {modalité}")
                else:
                    # dummy > 0.5 → dummy == 1 → modalité présente
                    conditions.append(f"{var_orig} = {modalité}")
            else:
                # Variable numérique
                if est_yes:
                    conditions.append(f"{feature} <= {seuil:.4g}")
                else:
                    conditions.append(f"{feature} > {seuil:.4g}")

            nœud_courant = parent["ID"]

        conditions.reverse()
        return conditions

    feuilles = tree_df[tree_df["Feature"] == "Leaf"]
    règles   = {}
    for _, feuille in feuilles.iterrows():
        leaf_id = feuille["ID"]
        chemin  = chemin_vers_feuille(leaf_id)
        règles[leaf_id] = " ET ".join(chemin) if chemin else "Population totale"

    return règles


def analyser_shap_xgb(
    model: xgb.Booster,
    df_xgb: pd.DataFrame,
    mapping: dict,
    max_display: int = 15,
) -> pd.DataFrame:
    """
    Calcule et visualise les valeurs SHAP du modèle XGBoost de diagnostic.
    Agrège les importances des dummies par variable originale.
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_xgb)

    # Importance par dummy
    importance_dummy = pd.DataFrame({
        "dummy":              df_xgb.columns.tolist(),
        "variable":           [mapping.get(c, c) for c in df_xgb.columns],
        "importance_mean_abs": np.abs(shap_values).mean(axis=0),
        "effet_moyen_signe":   shap_values.mean(axis=0),
    })

    # Agrégation par variable originale
    importance_df = (
        importance_dummy
        .groupby("variable")
        .agg(
            importance_mean_abs=("importance_mean_abs", "sum"),
            effet_moyen_signe=("effet_moyen_signe",   "sum"),
            importance_std=("importance_mean_abs",     "std"),
        )
        .reset_index()
        .sort_values("importance_mean_abs", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["importance_pct"] = (
        importance_df["importance_mean_abs"] /
        importance_df["importance_mean_abs"].sum() * 100
    ).round(1)

    # ── Plot 1 : Bar plot importance + effet signé ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    top = importance_df.head(max_display)

    axes[0].barh(
        top["variable"][::-1],
        top["importance_mean_abs"][::-1],
        color="#1f77b4", edgecolor="white",
    )
    for i, (_, row) in enumerate(top[::-1].iterrows()):
        axes[0].text(
            row["importance_mean_abs"] * 1.01, i,
            f"{row['importance_pct']:.1f}%",
            va="center", fontsize=8,
        )
    axes[0].set_xlabel("Importance SHAP moyenne (|SHAP|)")
    axes[0].set_title("Importance des variables\n(contribution aux erreurs de calibration)",
                      fontweight="bold")

    couleurs = ["#d62728" if v > 0 else "#2ca02c" for v in top["effet_moyen_signe"][::-1]]
    axes[1].barh(
        top["variable"][::-1],
        top["effet_moyen_signe"][::-1],
        color=couleurs, edgecolor="white",
    )
    axes[1].axvline(0, color="black", linewidth=1.2)
    axes[1].set_xlabel("Effet SHAP moyen signé")
    axes[1].set_title("Direction de l'effet\n(rouge = pousse vers sur-estimation)",
                      fontweight="bold")

    plt.tight_layout()
    plt.show()

    # ── Plot 2 : Beeswarm sur les dummies (plus granulaire) ────────────────
    shap.summary_plot(
        shap_values,
        df_xgb,
        max_display=max_display,
        show=True,
        plot_type="dot",
    )

    return importance_df


def diagnostiquer_calibration_xgb(
    df: pd.DataFrame,
    col_cible: str,
    col_proba: str,
    features: list[str],
    cat_features: list[str] | None = None,
    taille_min_feuille: float = 0.05,
    alpha: float = 0.05,
    max_leaves: int = 16,
    tracer: bool = True,
    shap: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifie les segments où le modèle sur/sous-estime via un arbre XGBoost unique.
    Les catégorielles sont encodées en dummies.

    Retourne le DataFrame de diagnostic, et optionnellement le DataFrame SHAP.
    """
    n_total       = len(df)
    min_child     = max(1, int(taille_min_feuille * n_total))
    cat_features  = cat_features or []
    résidu        = (df[col_proba] - df[col_cible]).values

    # Préparation features
    df_xgb, mapping = préparer_features_xgb(df, features, cat_features)

    # Dataset XGBoost
    dtrain = xgb.DMatrix(df_xgb, label=résidu)

    # Arbre unique
    params = {
        "objective":        "reg:squarederror",
        "max_leaves":       max_leaves,
        "min_child_weight": min_child,
        "eta":              1.0,
        "max_depth":        0,      # 0 = non limité si max_leaves est défini
        "tree_method":      "hist",
        "grow_policy":      "lossguide",
        "verbosity":        0,
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1,
    )

    # Tracé de l'arbre
    if tracer:
        xgb.plot_tree(model, num_trees=0, figsize=(24, 12))
        plt.title("Arbre de diagnostic de calibration", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    # Affectation aux feuilles
    feuille_ids  = model.predict(dtrain, pred_leaf=True).flatten().astype(int)
    df_analyse   = df[[col_cible, col_proba]].copy()
    df_analyse["feuille_id"] = feuille_ids

    # Extraction des règles
    règles_dict = extraire_règles_feuilles_xgb(model, df_xgb, mapping)

    # Métriques et test Z par feuille
    résultats = []
    for feuille_id, groupe in df_analyse.groupby("feuille_id"):
        n              = len(groupe)
        nb_def         = int(groupe[col_cible].sum())
        taux_def       = nb_def / n
        pd_moy         = groupe[col_proba].mean()
        erreur_moyenne = (groupe[col_proba] - groupe[col_cible]).mean()

        denom  = np.sqrt(pd_moy * (1 - pd_moy) / n)
        z_stat = (pd_moy - taux_def) / denom if denom > 0 else 0.0
        p_val  = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        if p_val >= alpha:
            statut = "✅ Bien calibré"
        elif z_stat > 0:
            statut = "🔴 Sur-estimation"
        else:
            statut = "🟡 Sous-estimation"

        règle = règles_dict.get(f"0-{feuille_id}", f"Feuille {feuille_id}")

        résultats.append({
            "feuille":         feuille_id,
            "règle":           règle,
            "effectif":        n,
            "nb_defauts":      nb_def,
            "pct_population":  round(n / n_total * 100, 1),
            "taux_defaut":     round(taux_def,      4),
            "pd_moyenne":      round(pd_moy,        4),
            "erreur_moyenne":  round(erreur_moyenne, 4),
            "z_stat":          round(z_stat,        3),
            "p_value":         round(p_val,         4),
            "statut":          statut,
        })

    résultat_df = (
        pd.DataFrame(résultats)
          .sort_values("statut")
          .reset_index(drop=True)
    )

    if shap:
        shap_df = analyser_shap_xgb(model, df_xgb, mapping)
        return résultat_df, shap_df

    return résultat_df



# Sans SHAP
résultat = diagnostiquer_calibration_xgb(
    df=df_scored,
    col_cible="defaut",
    col_proba="score_proba",
    features=["age", "revenu", "secteur", "anciennete"],
    cat_features=["secteur"],
    taille_min_feuille=0.05,
    alpha=0.05,
    max_leaves=16,
    tracer=True,
)

# Avec SHAP
résultat, importance_shap = diagnostiquer_calibration_xgb(
    df=df_scored,
    col_cible="defaut",
    col_proba="score_proba",
    features=["age", "revenu", "secteur", "anciennete"],
    cat_features=["secteur"],
    shap=True,
)
