import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings("ignore")


def analyse_categorielle(df, var_cat, var_cible, var_mois=None, figsize_base=(12, 5)):
    """
    Analyse complète d'une variable catégorielle dans un contexte de scoring crédit.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les données.
    var_cat : str
        Nom de la variable catégorielle à analyser.
    var_cible : str
        Nom de la variable cible binaire (0/1, défaut).
    var_mois : str, optional
        Nom de la colonne de période (mois). Si fourni, l'analyse temporelle est réalisée.
    figsize_base : tuple
        Taille de base des figures.

    Retourne
    -------
    dict : dictionnaire contenant les tables de résultats (stats, bivarie, temporel).
    """

    resultats = {}
    palette = ["#2E4057", "#048A81", "#8B575C", "#D4A843", "#6A5ACD", "#E07A5F", "#3D405B"]

    # =========================================================================
    # 1. ANALYSE UNIVARIÉE
    # =========================================================================
    print("=" * 70)
    print(f"  1. ANALYSE UNIVARIÉE : {var_cat}")
    print("=" * 70)

    n_total = len(df)
    n_missing = df[var_cat].isna().sum()
    pct_missing = n_missing / n_total * 100
    n_modalites = df[var_cat].nunique()

    stats = (
        df[var_cat]
        .value_counts()
        .reset_index()
        .rename(columns={"count": "effectif"})
    )
    stats["frequence_%"] = (stats["effectif"] / n_total * 100).round(2)
    stats["frequence_cum_%"] = stats["frequence_%"].cumsum().round(2)

    # Entropie de Shannon
    probs = stats["effectif"] / stats["effectif"].sum()
    entropie = -np.sum(probs * np.log2(probs))
    entropie_max = np.log2(n_modalites) if n_modalites > 1 else 1
    entropie_norm = entropie / entropie_max  # 1 = distribution uniforme

    print(f"\n  Nombre d'observations : {n_total:,}")
    print(f"  Valeurs manquantes    : {n_missing:,} ({pct_missing:.2f}%)")
    print(f"  Nombre de modalités   : {n_modalites}")
    print(f"  Entropie normalisée   : {entropie_norm:.3f}  (1 = uniforme)")
    print(f"\n{stats.to_string(index=False)}\n")

    resultats["univarie"] = stats

    # --- Graphique univarié ---
    fig, ax = plt.subplots(figsize=figsize_base)
    bars = ax.bar(
        stats[var_cat].astype(str),
        stats["frequence_%"],
        color=palette[: len(stats)],
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, val in zip(bars, stats["frequence_%"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title(f"Distribution de {var_cat}", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Fréquence (%)")
    ax.set_xlabel(var_cat)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"univarie_{var_cat}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # =========================================================================
    # 2. ANALYSE BIVARIÉE : lien avec la cible
    # =========================================================================
    print("=" * 70)
    print(f"  2. LIEN AVEC LA CIBLE : {var_cat} × {var_cible}")
    print("=" * 70)

    # Table croisée
    ct = pd.crosstab(df[var_cat], df[var_cible])
    chi2, p_value, dof, _ = chi2_contingency(ct)
    n = ct.sum().sum()
    k = min(ct.shape)
    v_cramer = np.sqrt(chi2 / (n * (k - 1))) if (k - 1) > 0 else 0

    # Taux de défaut + WoE / IV
    bivarie = df.groupby(var_cat)[var_cible].agg(["count", "sum"]).reset_index()
    bivarie.columns = [var_cat, "effectif", "nb_defaut"]
    bivarie["nb_non_defaut"] = bivarie["effectif"] - bivarie["nb_defaut"]
    bivarie["taux_defaut_%"] = (bivarie["nb_defaut"] / bivarie["effectif"] * 100).round(2)

    total_defaut = bivarie["nb_defaut"].sum()
    total_non_defaut = bivarie["nb_non_defaut"].sum()

    # WoE et IV
    bivarie["dist_defaut"] = bivarie["nb_defaut"] / total_defaut
    bivarie["dist_non_defaut"] = bivarie["nb_non_defaut"] / total_non_defaut
    bivarie["WoE"] = np.log(
        bivarie["dist_non_defaut"].clip(1e-10) / bivarie["dist_defaut"].clip(1e-10)
    ).round(4)
    bivarie["IV_modalite"] = (
        (bivarie["dist_non_defaut"] - bivarie["dist_defaut"]) * bivarie["WoE"]
    ).round(6)
    iv_total = bivarie["IV_modalite"].sum()

    # Interprétation IV
    if iv_total < 0.02:
        iv_interp = "Non prédictif"
    elif iv_total < 0.1:
        iv_interp = "Faiblement prédictif"
    elif iv_total < 0.3:
        iv_interp = "Moyennement prédictif"
    elif iv_total < 0.5:
        iv_interp = "Fortement prédictif"
    else:
        iv_interp = "Suspicieusement prédictif (vérifier)"

    print(f"\n  Chi-2       : {chi2:.2f}  (p-value = {p_value:.4e})")
    print(f"  V de Cramér : {v_cramer:.4f}")
    print(f"  IV total    : {iv_total:.4f}  → {iv_interp}")
    print(f"\n  Taux de défaut global : {total_defaut / n * 100:.2f}%\n")

    cols_affichage = [var_cat, "effectif", "nb_defaut", "taux_defaut_%", "WoE", "IV_modalite"]
    print(bivarie[cols_affichage].to_string(index=False))
    print()

    resultats["bivarie"] = bivarie
    resultats["tests"] = {
        "chi2": chi2,
        "p_value": p_value,
        "v_cramer": v_cramer,
        "iv_total": iv_total,
        "iv_interpretation": iv_interp,
    }

    # --- Graphique bivarié : barplot effectif + courbe taux de défaut ---
    fig, ax1 = plt.subplots(figsize=figsize_base)
    modalites = bivarie[var_cat].astype(str)
    x = np.arange(len(modalites))
    width = 0.5

    bars = ax1.bar(x, bivarie["effectif"], width, color="#2E4057", alpha=0.8, label="Effectif")
    ax1.set_ylabel("Effectif", color="#2E4057", fontsize=11)
    ax1.set_xlabel(var_cat)
    ax1.set_xticks(x)
    ax1.set_xticklabels(modalites)
    ax1.tick_params(axis="y", labelcolor="#2E4057")

    ax2 = ax1.twinx()
    ax2.plot(x, bivarie["taux_defaut_%"], color="#E07A5F", marker="o", linewidth=2.5,
             markersize=8, label="Taux de défaut (%)", zorder=5)
    for i, val in enumerate(bivarie["taux_defaut_%"]):
        ax2.annotate(f"{val:.1f}%", (x[i], val), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=10, fontweight="bold", color="#E07A5F")
    ax2.set_ylabel("Taux de défaut (%)", color="#E07A5F", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#E07A5F")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Ligne du taux global
    taux_global = total_defaut / n * 100
    ax2.axhline(y=taux_global, color="#E07A5F", linestyle="--", alpha=0.4, linewidth=1)
    ax2.text(len(modalites) - 1, taux_global, f"  global: {taux_global:.1f}%",
             va="bottom", color="#E07A5F", alpha=0.6, fontsize=9)

    ax1.set_title(f"{var_cat} — Effectif & Taux de défaut", fontsize=14, fontweight="bold", pad=15)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"bivarie_{var_cat}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- Graphique WoE ---
    fig, ax = plt.subplots(figsize=figsize_base)
    colors_woe = ["#048A81" if w >= 0 else "#8B575C" for w in bivarie["WoE"]]
    bars = ax.barh(modalites, bivarie["WoE"], color=colors_woe, edgecolor="white", height=0.5)
    ax.axvline(x=0, color="grey", linewidth=0.8, linestyle="-")
    for i, val in enumerate(bivarie["WoE"]):
        offset = 0.02 if val >= 0 else -0.02
        ax.text(val + offset, i, f"{val:.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=10, fontweight="bold")
    ax.set_title(f"{var_cat} — Weight of Evidence (IV = {iv_total:.4f})",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("WoE")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"woe_{var_cat}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # =========================================================================
    # 3. ANALYSE TEMPORELLE
    # =========================================================================
    if var_mois is not None:
        print("=" * 70)
        print(f"  3. ANALYSE TEMPORELLE : {var_cat} × {var_cible} × {var_mois}")
        print("=" * 70)

        # Taux de défaut par modalité et mois
        temporel = (
            df.groupby([var_mois, var_cat])[var_cible]
            .agg(["count", "sum"])
            .reset_index()
        )
        temporel.columns = [var_mois, var_cat, "effectif", "nb_defaut"]
        temporel["taux_defaut_%"] = (temporel["nb_defaut"] / temporel["effectif"] * 100).round(2)

        # Distribution des modalités par mois
        distrib_temp = temporel.pivot_table(
            index=var_mois, columns=var_cat, values="effectif", fill_value=0
        )
        distrib_temp_pct = distrib_temp.div(distrib_temp.sum(axis=1), axis=0) * 100

        # PSI entre première et dernière période
        periodes = sorted(df[var_mois].dropna().unique())
        if len(periodes) >= 2:
            p_ref = distrib_temp_pct.iloc[0].values / 100
            p_act = distrib_temp_pct.iloc[-1].values / 100
            p_ref = np.clip(p_ref, 1e-10, None)
            p_act = np.clip(p_act, 1e-10, None)
            psi = np.sum((p_act - p_ref) * np.log(p_act / p_ref))

            if psi < 0.1:
                psi_interp = "Stable"
            elif psi < 0.2:
                psi_interp = "Dérive légère"
            else:
                psi_interp = "Dérive significative"

            print(f"\n  PSI ({periodes[0]} → {periodes[-1]}) : {psi:.4f}  → {psi_interp}")
        else:
            psi = None
            psi_interp = "Non calculable (une seule période)"
            print(f"\n  PSI : {psi_interp}")

        resultats["temporel"] = temporel
        resultats["psi"] = {"psi": psi, "interpretation": psi_interp}

        # --- Graphique : évolution du taux de défaut par modalité ---
        fig, ax = plt.subplots(figsize=(max(figsize_base[0], 14), figsize_base[1] + 1))
        pivot_taux = temporel.pivot_table(
            index=var_mois, columns=var_cat, values="taux_defaut_%"
        )
        for i, col in enumerate(pivot_taux.columns):
            ax.plot(
                pivot_taux.index.astype(str),
                pivot_taux[col],
                marker="o",
                linewidth=2,
                markersize=5,
                label=str(col),
                color=palette[i % len(palette)],
            )
        ax.set_title(f"{var_cat} — Taux de défaut par mois", fontsize=14, fontweight="bold", pad=15)
        ax.set_ylabel("Taux de défaut (%)")
        ax.set_xlabel(var_mois)
        ax.legend(title=var_cat, loc="best", framealpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"temporel_taux_{var_cat}.png", dpi=150, bbox_inches="tight")
        plt.show()

        # --- Graphique : évolution de la distribution (stacked barplot) ---
        fig, ax = plt.subplots(figsize=(max(figsize_base[0], 14), figsize_base[1] + 1))
        bottom = np.zeros(len(distrib_temp_pct))
        for i, col in enumerate(distrib_temp_pct.columns):
            vals = distrib_temp_pct[col].values
            ax.bar(
                distrib_temp_pct.index.astype(str),
                vals,
                bottom=bottom,
                label=str(col),
                color=palette[i % len(palette)],
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += vals

        ax.set_title(f"{var_cat} — Répartition des modalités par mois (PSI = {psi:.4f})" if psi else
                     f"{var_cat} — Répartition des modalités par mois",
                     fontsize=14, fontweight="bold", pad=15)
        ax.set_ylabel("Répartition (%)")
        ax.set_xlabel(var_mois)
        ax.legend(title=var_cat, loc="best", framealpha=0.9)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"temporel_distrib_{var_cat}.png", dpi=150, bbox_inches="tight")
        plt.show()

        # --- Heatmap : taux de défaut (modalité × mois) ---
        fig, ax = plt.subplots(figsize=(max(figsize_base[0], 14), figsize_base[1]))
        heatmap_data = pivot_taux.T
        im = ax.imshow(heatmap_data.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(np.arange(len(heatmap_data.columns)))
        ax.set_xticklabels(heatmap_data.columns.astype(str), rotation=45, ha="right")
        ax.set_yticks(np.arange(len(heatmap_data.index)))
        ax.set_yticklabels(heatmap_data.index.astype(str))

        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                val = heatmap_data.values[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > heatmap_data.values[~np.isnan(heatmap_data.values)].mean() else "black"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                            fontsize=9, color=text_color, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Taux de défaut (%)", shrink=0.8)
        ax.set_title(f"{var_cat} — Heatmap taux de défaut (modalité × mois)",
                     fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(f"heatmap_{var_cat}.png", dpi=150, bbox_inches="tight")
        plt.show()

    print("=" * 70)
    print("  ANALYSE TERMINÉE")
    print("=" * 70)

    return resultats


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================
if __name__ == "__main__":
    # Génération de données fictives pour démonstration
    np.random.seed(42)
    n = 5000
    mois = pd.date_range("2024-01", periods=12, freq="MS").strftime("%Y-%m")

    df_demo = pd.DataFrame({
        "type_contrat": np.random.choice(["CDI", "CDD", "Freelance"], size=n, p=[0.6, 0.3, 0.1]),
        "mois_octroi": np.random.choice(mois, size=n),
    })

    # Taux de défaut variable selon la modalité
    proba_defaut = df_demo["type_contrat"].map({"CDI": 0.05, "CDD": 0.12, "Freelance": 0.18})
    df_demo["defaut"] = np.random.binomial(1, proba_defaut)

    # Lancement de l'analyse
    res = analyse_categorielle(
        df=df_demo,
        var_cat="type_contrat",
        var_cible="defaut",
        var_mois="mois_octroi",
    )
