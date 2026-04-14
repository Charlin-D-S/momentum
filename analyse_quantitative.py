import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")


def analyse_quantitative(df, var_quanti, var_cible, var_mois=None,
                         seuils=None, n_tranches=5,
                         winsor_limits=(0.01, 0.99), figsize_base=(12, 5)):
    """
    Analyse complète d'une variable quantitative dans un contexte de scoring crédit.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les données.
    var_quanti : str
        Nom de la variable quantitative à analyser.
    var_cible : str
        Nom de la variable cible binaire (0/1, défaut).
    var_mois : str, optional
        Nom de la colonne de période (mois). Si fourni, l'analyse temporelle est réalisée.
    seuils : list, optional
        Liste de seuils pour discrétiser la variable (ex: [0, 1000, 5000, 10000]).
        Si None, découpage automatique en quantiles via n_tranches.
    n_tranches : int
        Nombre de tranches si seuils=None (défaut : 5 = quintiles).
    winsor_limits : tuple
        Percentiles bas et haut pour la winsorisation de l'histogramme (défaut : 1% / 99%).
    figsize_base : tuple
        Taille de base des figures.

    Retourne
    -------
    dict : dictionnaire contenant les tables de résultats.
    """

    resultats = {}
    palette = ["#2E4057", "#048A81", "#8B575C", "#D4A843", "#6A5ACD", "#E07A5F", "#3D405B",
               "#5B8C5A", "#C06C84", "#355C7D"]

    # =========================================================================
    # 1. ANALYSE UNIVARIÉE
    # =========================================================================
    print("=" * 70)
    print(f"  1. ANALYSE UNIVARIÉE : {var_quanti}")
    print("=" * 70)

    serie = df[var_quanti]
    n_total = len(serie)
    n_missing = serie.isna().sum()
    pct_missing = n_missing / n_total * 100
    serie_clean = serie.dropna()

    stats = {
        "count": n_total,
        "missing": n_missing,
        "missing_%": round(pct_missing, 2),
        "mean": round(serie_clean.mean(), 4),
        "median": round(serie_clean.median(), 4),
        "std": round(serie_clean.std(), 4),
        "min": round(serie_clean.min(), 4),
        "max": round(serie_clean.max(), 4),
        "skewness": round(serie_clean.skew(), 4),
        "kurtosis": round(serie_clean.kurtosis(), 4),
    }

    quantiles_pct = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    quantiles_vals = serie_clean.quantile(quantiles_pct)
    for q, v in zip(quantiles_pct, quantiles_vals):
        label = f"Q{int(q * 100)}"
        stats[label] = round(v, 4)

    stats_df = pd.DataFrame([stats])
    print(f"\n{stats_df.T.to_string(header=False)}\n")

    resultats["univarie"] = stats

    # --- Histogramme + KDE avec winsorisation ---
    low = serie_clean.quantile(winsor_limits[0])
    high = serie_clean.quantile(winsor_limits[1])
    serie_winsor = serie_clean.clip(lower=low, upper=high)

    fig, ax = plt.subplots(figsize=figsize_base)
    ax.hist(serie_winsor, bins=50, color="#2E4057", alpha=0.7, edgecolor="white",
            linewidth=0.5, density=True, label="Histogramme")
    serie_winsor.plot.kde(ax=ax, color="#E07A5F", linewidth=2.5, label="KDE")
    ax.axvline(serie_clean.mean(), color="#048A81", linestyle="--", linewidth=1.5,
               label=f"Moyenne = {serie_clean.mean():.2f}")
    ax.axvline(serie_clean.median(), color="#D4A843", linestyle="--", linewidth=1.5,
               label=f"Médiane = {serie_clean.median():.2f}")
    ax.set_title(f"Distribution de {var_quanti} (winsorisé P{int(winsor_limits[0]*100)}-P{int(winsor_limits[1]*100)})",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(var_quanti)
    ax.set_ylabel("Densité")
    ax.legend(loc="best", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"univarie_{var_quanti}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # =========================================================================
    # 2. ANALYSE BIVARIÉE : lien avec la cible
    # =========================================================================
    print("=" * 70)
    print(f"  2. LIEN AVEC LA CIBLE : {var_quanti} × {var_cible}")
    print("=" * 70)

    df_clean = df[[var_quanti, var_cible]].dropna()
    grp0 = df_clean.loc[df_clean[var_cible] == 0, var_quanti]
    grp1 = df_clean.loc[df_clean[var_cible] == 1, var_quanti]

    # Moyennes et médianes par groupe
    print(f"\n  {'':30s} {'Non défaut':>12s} {'Défaut':>12s}")
    print(f"  {'Moyenne':30s} {grp0.mean():>12.4f} {grp1.mean():>12.4f}")
    print(f"  {'Médiane':30s} {grp0.median():>12.4f} {grp1.median():>12.4f}")
    print(f"  {'Écart-type':30s} {grp0.std():>12.4f} {grp1.std():>12.4f}")
    print(f"  {'Effectif':30s} {len(grp0):>12d} {len(grp1):>12d}")

    # Mann-Whitney
    mw_stat, mw_pvalue = mannwhitneyu(grp0, grp1, alternative="two-sided")
    print(f"\n  Mann-Whitney U : {mw_stat:.2f}  (p-value = {mw_pvalue:.4e})")

    # AUC
    auc = roc_auc_score(df_clean[var_cible], df_clean[var_quanti])
    auc_adj = max(auc, 1 - auc)  # ajusté si relation inverse
    print(f"  AUC            : {auc:.4f}  (ajusté = {auc_adj:.4f})")

    resultats["tests"] = {
        "mann_whitney_U": mw_stat,
        "mann_whitney_pvalue": mw_pvalue,
        "auc": auc,
        "auc_ajuste": auc_adj,
        "moyenne_non_defaut": grp0.mean(),
        "moyenne_defaut": grp1.mean(),
        "mediane_non_defaut": grp0.median(),
        "mediane_defaut": grp1.median(),
    }

    # --- Discrétisation ---
    if seuils is not None:
        # Seuils manuels
        bins = [-np.inf] + sorted(seuils) + [np.inf]
        labels_tranches = []
        for i in range(len(bins) - 1):
            if bins[i] == -np.inf:
                labels_tranches.append(f"≤ {bins[i+1]}")
            elif bins[i + 1] == np.inf:
                labels_tranches.append(f"> {bins[i]}")
            else:
                labels_tranches.append(f"]{bins[i]}, {bins[i+1]}]")
        df_clean["tranche"] = pd.cut(df_clean[var_quanti], bins=bins, labels=labels_tranches,
                                     include_lowest=True)
    else:
        # Quantiles automatiques
        df_clean["tranche"] = pd.qcut(df_clean[var_quanti], q=n_tranches, duplicates="drop")
        df_clean["tranche"] = df_clean["tranche"].astype(str)

    # Table par tranche
    bivarie = df_clean.groupby("tranche", observed=False)[var_cible].agg(["count", "sum"]).reset_index()
    bivarie.columns = ["tranche", "effectif", "nb_defaut"]
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
    bivarie["IV_tranche"] = (
        (bivarie["dist_non_defaut"] - bivarie["dist_defaut"]) * bivarie["WoE"]
    ).round(6)
    iv_total = bivarie["IV_tranche"].sum()

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

    print(f"\n  IV total       : {iv_total:.4f}  → {iv_interp}")
    taux_global = total_defaut / (total_defaut + total_non_defaut) * 100
    print(f"  Taux de défaut global : {taux_global:.2f}%\n")

    cols_affichage = ["tranche", "effectif", "nb_defaut", "taux_defaut_%", "WoE", "IV_tranche"]
    print(bivarie[cols_affichage].to_string(index=False))
    print()

    resultats["bivarie"] = bivarie
    resultats["iv"] = {"iv_total": iv_total, "interpretation": iv_interp}

    # --- Graphique bivarié : barplot effectif + courbe taux de défaut ---
    fig, ax1 = plt.subplots(figsize=figsize_base)
    tranches = bivarie["tranche"].astype(str)
    x = np.arange(len(tranches))
    width = 0.5

    bars = ax1.bar(x, bivarie["effectif"], width, color="#2E4057", alpha=0.8, label="Effectif")
    ax1.set_ylabel("Effectif", color="#2E4057", fontsize=11)
    ax1.set_xlabel(var_quanti)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tranches, rotation=30, ha="right")
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

    ax2.axhline(y=taux_global, color="#E07A5F", linestyle="--", alpha=0.4, linewidth=1)
    ax2.text(len(tranches) - 1, taux_global, f"  global: {taux_global:.1f}%",
             va="bottom", color="#E07A5F", alpha=0.6, fontsize=9)

    ax1.set_title(f"{var_quanti} — Effectif & Taux de défaut par tranche (IV = {iv_total:.4f})",
                  fontsize=14, fontweight="bold", pad=15)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"bivarie_{var_quanti}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # =========================================================================
    # 3. ANALYSE TEMPORELLE
    # =========================================================================
    if var_mois is not None:
        print("=" * 70)
        print(f"  3. ANALYSE TEMPORELLE : {var_quanti} × {var_cible} × {var_mois}")
        print("=" * 70)

        # Réappliquer la discrétisation sur le df complet avec mois
        df_temp = df[[var_quanti, var_cible, var_mois]].dropna()
        if seuils is not None:
            df_temp["tranche"] = pd.cut(df_temp[var_quanti], bins=bins, labels=labels_tranches,
                                        include_lowest=True)
        else:
            df_temp["tranche"] = pd.qcut(df_temp[var_quanti], q=n_tranches, duplicates="drop")
            df_temp["tranche"] = df_temp["tranche"].astype(str)

        # Taux de défaut par tranche et mois
        temporel = (
            df_temp.groupby([var_mois, "tranche"], observed=False)[var_cible]
            .agg(["count", "sum"])
            .reset_index()
        )
        temporel.columns = [var_mois, "tranche", "effectif", "nb_defaut"]
        temporel["taux_defaut_%"] = (temporel["nb_defaut"] / temporel["effectif"] * 100).round(2)

        # PSI entre première et dernière période
        periodes = sorted(df_temp[var_mois].dropna().unique())
        if len(periodes) >= 2:
            distrib_temp = df_temp.groupby([var_mois, "tranche"], observed=False).size().unstack(fill_value=0)
            distrib_temp_pct = distrib_temp.div(distrib_temp.sum(axis=1), axis=0)

            p_ref = distrib_temp_pct.loc[periodes[0]].values
            p_act = distrib_temp_pct.loc[periodes[-1]].values
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

        # --- Line plot : taux de défaut par tranche au fil du temps ---
        fig, ax = plt.subplots(figsize=(max(figsize_base[0], 14), figsize_base[1] + 1))
        pivot_taux = temporel.pivot_table(
            index=var_mois, columns="tranche", values="taux_defaut_%"
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
        psi_text = f" (PSI = {psi:.4f})" if psi is not None else ""
        ax.set_title(f"{var_quanti} — Taux de défaut par tranche et par mois{psi_text}",
                     fontsize=14, fontweight="bold", pad=15)
        ax.set_ylabel("Taux de défaut (%)")
        ax.set_xlabel(var_mois)
        ax.legend(title="Tranche", loc="best", framealpha=0.9, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"temporel_taux_{var_quanti}.png", dpi=150, bbox_inches="tight")
        plt.show()

    print("=" * 70)
    print("  ANALYSE TERMINÉE")
    print("=" * 70)

    return resultats


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    n = 5000
    mois = pd.date_range("2024-01", periods=12, freq="MS").strftime("%Y-%m")

    df_demo = pd.DataFrame({
        "revenu": np.random.lognormal(mean=7.5, sigma=0.8, size=n),
        "mois_octroi": np.random.choice(mois, size=n),
    })

    # Probabilité de défaut inversement liée au revenu
    proba = 1 / (1 + np.exp((df_demo["revenu"] - 1500) / 500))
    df_demo["defaut"] = np.random.binomial(1, proba)

    # --- Avec quantiles automatiques ---
    res = analyse_quantitative(
        df=df_demo,
        var_quanti="revenu",
        var_cible="defaut",
        var_mois="mois_octroi",
        n_tranches=5,
    )

    # --- Avec seuils manuels ---
    # res = analyse_quantitative(
    #     df=df_demo,
    #     var_quanti="revenu",
    #     var_cible="defaut",
    #     var_mois="mois_octroi",
    #     seuils=[500, 1000, 2000, 5000],
    # )
