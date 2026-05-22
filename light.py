import shap
import matplotlib.pyplot as plt


def analyser_shap_calibration(
    model: lgb.Booster,
    df_lgb: pd.DataFrame,
    cat_mappings: dict,
    max_display: int = 15,
) -> pd.DataFrame:
    """
    Calcule et visualise les valeurs SHAP du modèle LightGBM de diagnostic
    de calibration.

    Paramètres
    ----------
    model       : Modèle LightGBM entraîné sur les résidus.
    df_lgb      : DataFrame utilisé pour l'entraînement (features uniquement).
    cat_mappings: Mapping catégoriel {col: {index: label}}.
    max_display : Nombre max de variables affichées sur les plots.

    Retourne
    --------
    DataFrame d'importance SHAP trié par importance décroissante.
    """
    # Calcul des valeurs SHAP
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_lgb)

    # DataFrame d'importance : moyenne des |SHAP| par variable
    importance_df = (
        pd.DataFrame({
            "variable":       df_lgb.columns.tolist(),
            "importance_mean_abs": np.abs(shap_values).mean(axis=0),
            "importance_std":      shap_values.std(axis=0),
            "effet_moyen_signe":   shap_values.mean(axis=0),
        })
        .sort_values("importance_mean_abs", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["importance_pct"] = (
        importance_df["importance_mean_abs"] /
        importance_df["importance_mean_abs"].sum() * 100
    ).round(1)

    # ── Plot 1 : Bar plot importance moyenne ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    top_vars = importance_df.head(max_display)
    axes[0].barh(
        top_vars["variable"][::-1],
        top_vars["importance_mean_abs"][::-1],
        color="#1f77b4", edgecolor="white", linewidth=0.5,
    )
    axes[0].set_xlabel("Importance SHAP moyenne (|SHAP|)", fontsize=11)
    axes[0].set_title("Importance des variables\n(contribution aux erreurs de calibration)",
                      fontsize=12, fontweight="bold")
    axes[0].axvline(0, color="black", linewidth=0.8)

    # Annotations % sur les barres
    for i, (_, row) in enumerate(top_vars[::-1].iterrows()):
        axes[0].text(
            row["importance_mean_abs"] * 1.01,
            i,
            f"{row['importance_pct']:.1f}%",
            va="center", fontsize=8,
        )

    # ── Plot 2 : Effet moyen signé (sur/sous estimation par variable) ──────
    couleurs = [
        "#d62728" if v > 0 else "#2ca02c"
        for v in top_vars["effet_moyen_signe"][::-1]
    ]
    axes[1].barh(
        top_vars["variable"][::-1],
        top_vars["effet_moyen_signe"][::-1],
        color=couleurs, edgecolor="white", linewidth=0.5,
    )
    axes[1].set_xlabel("Effet SHAP moyen signé", fontsize=11)
    axes[1].set_title("Direction de l'effet\n(rouge = pousse vers sur-estimation)",
                      fontsize=12, fontweight="bold")
    axes[1].axvline(0, color="black", linewidth=1.2)

    plt.tight_layout()
    plt.show()

    # ── Plot 3 : Beeswarm SHAP summary ────────────────────────────────────
    shap.summary_plot(
        shap_values,
        df_lgb,
        max_display=max_display,
        show=True,
        plot_type="dot",
    )

    return importance_df
    
 def diagnostiquer_calibration(..., shap: bool = False):
    ...
    # À la fin, juste avant le return
    if shap:
        shap_df = analyser_shap_calibration(model, df_lgb, cat_mappings)
        return pd.DataFrame(résultats).sort_values("statut").reset_index(drop=True), shap_df

    return pd.DataFrame(résultats).sort_values("statut").reset_index(drop=True)   
    
    def extraire_règles_feuilles(model: lgb.Booster, cat_mappings: dict) -> dict[int, str]:
    tree_df = model.trees_to_dataframe()
    tree_df = tree_df[tree_df["tree_index"] == 0].copy()

    # Colonnes catégorielles connues
    cat_cols = set(cat_mappings.keys())

    def décoder_cats(feature: str, seuil, est_gauche: bool) -> str:
        # LightGBM encode "0||2||5" ou juste "2" pour les catégorielles
        indices_gauche = [int(i) for i in str(seuil).split("||")]
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

            # Distinction via le nom de la feature — pas via le type du seuil
            if feature in cat_cols:
                conditions.append(décoder_cats(feature, seuil, est_gauche))
            else:
                seuil_f = float(seuil)
                if est_gauche:
                    conditions.append(f"{feature} <= {seuil_f:.4g}")
                else:
                    conditions.append(f"{feature} > {seuil_f:.4g}")

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


def diagnostiquer_calibration(
    df: pd.DataFrame,
    col_cible: str,
    col_proba: str,
    features: list[str],
    cat_features: list[str] | None = None,
    taille_min_feuille: float = 0.05,
    alpha: float = 0.05,
    max_leaves: int = 16,
    tracer: bool = True,
) -> pd.DataFrame:

    n_total       = len(df)
    min_data_leaf = max(1, int(taille_min_feuille * n_total))
    cat_features  = cat_features or []
    résidu        = df[col_proba] - df[col_cible]

    # Mapping catégoriel AVANT conversion
    cat_mappings = {
        col: dict(enumerate(df[col].astype("category").cat.categories))
        for col in cat_features
    }

    # Préparation du dataset
    df_lgb = df[features].copy()
    for col in cat_features:
        df_lgb[col] = df_lgb[col].astype("category")

    dataset = lgb.Dataset(
        data=df_lgb,
        label=résidu,
        categorical_feature=cat_features if cat_features else "auto",
        free_raw_data=False,
    )

    params = {
        "objective":        "regression",
        "num_leaves":       max_leaves,
        "min_data_in_leaf": min_data_leaf,
        "num_iterations":   1,
        "learning_rate":    1.0,
        "verbose":          -1,
        "force_row_wise":   True,
    }

    model = lgb.train(params=params, train_set=dataset, num_boost_round=1)

    # Tracé
    if tracer:
        tracer_arbre(model)

    # Affectation aux feuilles
    feuille_ids = model.predict(df_lgb, pred_leaf=True).flatten()
    df_analyse  = df[[col_cible, col_proba]].copy()
    df_analyse["feuille_id"] = feuille_ids

    # Extraction des règles
    règles_dict = extraire_règles_feuilles(model, cat_mappings)

    # Métriques par feuille
    résultats = []
    for feuille_id, groupe in df_analyse.groupby("feuille_id"):
        n               = len(groupe)
        nb_def          = int(groupe[col_cible].sum())
        taux_def        = nb_def / n
        pd_moy          = groupe[col_proba].mean()
        erreur_moyenne  = (groupe[col_proba] - groupe[col_cible]).mean()

        denom  = np.sqrt(pd_moy * (1 - pd_moy) / n)
        z_stat = (pd_moy - taux_def) / denom if denom > 0 else 0.0
        p_val  = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        if p_val >= alpha:
            statut = "✅ Bien calibré"
        elif z_stat > 0:
            statut = "🔴 Sur-estimation"
        else:
            statut = "🟡 Sous-estimation"

        leaf_key = f"0-L{feuille_id}"
        règle    = règles_dict.get(leaf_key, f"Feuille {feuille_id}")

        résultats.append({
            "feuille":         feuille_id,
            "règle":           règle,
            "effectif":        n,
            "nb_defauts":      nb_def,
            "pct_population":  round(n / n_total * 100, 1),
            "taux_defaut":     round(taux_def,     4),
            "pd_moyenne":      round(pd_moy,       4),
            "erreur_moyenne":  round(erreur_moyenne,4),
            "z_stat":          round(z_stat,       3),
            "p_value":         round(p_val,        4),
            "statut":          statut,
        })

    return (
        pd.DataFrame(résultats)
          .sort_values("statut")
          .reset_index(drop=True)
    )

import lightgbm as lgb
import matplotlib.pyplot as plt


def tracer_arbre(
    model: lgb.Booster,
    figsize: tuple = (24, 12),
) -> None:
    """
    Trace l'arbre LightGBM unique avec les règles de split.
    """
    ax = lgb.plot_tree(
        model,
        tree_index=0,
        figsize=figsize,
        show_info=["split_gain", "internal_count", "leaf_count"],
        precision=3,
    )
    ax.set_title("Arbre de diagnostic de calibration", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
