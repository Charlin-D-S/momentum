"""Binner — version Polars.

Differences clés par rapport à la version pandas :
- self.X est un pl.DataFrame (immutable) ; les methodes qui modifient une
  colonne retournent toujours la Series modifiee ET mettent a jour self.X
  via with_columns (pas de mutation en place).
- mask : pl.Series booleenne ou pl.Expr, passee a filter() au lieu de __getitem__.
- v_cramer_t_tschuprow : prend une pl.Series (au lieu d'une pd.Series).
- Les plots restent matplotlib/seaborn ; une conversion .to_pandas() minimale
  est faite juste avant chaque appel seaborn, en dehors de toute boucle lourde.
"""
def plot_bin_stability_over_time(
    self, var_binned, ref_period=None, min_obs=1, min_pop=0.05, mask=None
) -> tuple:
    """Retourne (fig_volume, fig_dr) pour compatibilite Streamlit et notebook."""
    # ... (agrégation et PSI inchangés) ...

    # -- Graphe 1 : volumes + PSI --
    fig1, ax1 = plt.subplots(figsize=(13, 5))
    sns.lineplot(data=agg_pd, x="period", y="pct_obs", hue="bin", marker="o", ax=ax1)
    ax1.axhline(min_pop, color="grey", linestyle="--", linewidth=1, label="Seuil 5 %")
    ax1.set_ylabel("Share of population")
    ax1.set_xlabel("Period")
    ax1.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_title("Evolution of population shares by class with PSI")

    ax2 = ax1.twinx()
    ax2.plot(psi_pd["period"], psi_pd["psi"],
             color="black", linestyle="--", marker="s", label="PSI")
    ax2.axhline(0.10, color="orange", linestyle=":")
    ax2.axhline(0.25, color="red",    linestyle=":")
    ax2.set_ylabel("PSI")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               bbox_to_anchor=(1.02, 1), loc="upper left")
    fig1.tight_layout()

    # -- Graphe 2 : taux de défaut --
    fig2, ax3 = plt.subplots(figsize=(13, 5))
    sns.lineplot(
        data=agg_pd[agg_pd["n_obs"] >= min_obs],
        x="period", y="default_rate", hue="bin", marker="o", ax=ax3,
    )
    ax3.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax3.axhline(global_dr, color="black", linestyle="--", label="DR global")
    ax3.set_title("Evolution of default rate by class")
    ax3.set_ylabel("Default rate (%)")
    ax3.set_xlabel("Period")
    ax3.tick_params(axis="x", rotation=45)
    ax3.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig2.tight_layout()

    return fig1, fig2   # <-- plus de plt.show()

import streamlit as st
import polars as pl
from binner_polars import Binner

st.title("Stabilité des bins")

# --- Sidebar : paramètres ---
var      = st.sidebar.selectbox("Variable binnée", options=df.columns)
ref_year = st.sidebar.selectbox("Période de référence", options=sorted(df["annee"].unique().to_list()))
min_obs  = st.sidebar.slider("Observations minimum (DR)", 1, 100, 10)
min_pop  = st.sidebar.slider("Seuil volume (%)", 0.0, 0.2, 0.05, step=0.01)

# --- Calcul ---
b = Binner(df, cible_col="default", date_col="annee")

fig_vol, fig_dr = b.plot_bin_stability_over_time(
    var_binned=var,
    ref_period=ref_year,
    min_obs=min_obs,
    min_pop=min_pop,
)

# --- Affichage ---
st.subheader("Volumes et PSI")
st.pyplot(fig_vol)

st.subheader("Taux de défaut par classe")
st.pyplot(fig_dr)
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy.stats import chi2_contingency


class Binner:

    def __init__(
        self,
        X: pl.DataFrame,
        cible_col: str = "default_t_plus_1",
        date_col: str = "obs_year",
    ):
        self.X = X
        self.cible_col = cible_col
        self.date_col = date_col

    # ------------------------------------------------------------------
    # UTILITAIRES INTERNES
    # ------------------------------------------------------------------

    def _filter(self, mask=None) -> pl.DataFrame:
        """Applique un masque (pl.Series bool ou pl.Expr) ou renvoie self.X."""
        if mask is None:
            return self.X
        if isinstance(mask, pl.Expr):
            return self.X.filter(mask)
        return self.X.filter(mask)

    def compute_psi(
        self,
        ref_dist: np.ndarray,
        cur_dist: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """PSI entre deux distributions (vecteurs numpy normalisés)."""
        return float(
            np.sum(
                (ref_dist - cur_dist)
                * np.log((ref_dist + eps) / (cur_dist + eps))
            )
        )

    # ------------------------------------------------------------------
    # STABILITE DES BINS DANS LE TEMPS
    # ------------------------------------------------------------------

    def plot_bin_stability_over_time(
        self,
        var_binned: str,
        ref_period=None,
        min_obs: int = 1,
        min_pop: float = 0.05,
        mask=None,
    ) -> None:
        """Volumes par bin et taux de défaut au fil du temps, avec PSI."""
        X = self._filter(mask)

        # -- Agrégation Polars ----------------------------------------
        agg = (
            X.select([var_binned, self.cible_col, self.date_col])
            .rename(
                {var_binned: "bin", self.cible_col: "target", self.date_col: "period"}
            )
            .group_by(["period", "bin"])
            .agg(
                pl.col("target").count().alias("n_obs"),
                pl.col("target").cast(pl.Int64).sum().alias("n_defaults"),
            )
            .with_columns(
                (pl.col("n_defaults") / pl.col("n_obs")).alias("default_rate")
            )
        )

        total_per_period = (
            agg.group_by("period")
            .agg(pl.col("n_obs").sum().alias("total_period"))
        )

        agg = (
            agg.join(total_per_period, on="period")
            .with_columns(
                (pl.col("n_obs") / pl.col("total_period")).alias("pct_obs")
            )
            .sort(["period", "bin"])
        )

        # -- PSI -------------------------------------------------------
        if ref_period is None:
            ref_period = agg["period"].min()

        ref_rows = (
            agg.filter(pl.col("period") == ref_period)
            .select(["bin", "n_obs"])
        )
        ref_total = ref_rows["n_obs"].sum()
        ref_dict = {
            row["bin"]: row["n_obs"] / ref_total
            for row in ref_rows.to_dicts()
        }

        psi_records = []
        for period in sorted(agg["period"].unique().to_list()):
            cur_rows = (
                agg.filter(pl.col("period") == period)
                .select(["bin", "n_obs"])
            )
            cur_total = cur_rows["n_obs"].sum()
            cur_dict = {row["bin"]: row["n_obs"] / cur_total for row in cur_rows.to_dicts()}

            all_bins = list(ref_dict.keys())
            ref_arr = np.array([ref_dict.get(b, 0.0) for b in all_bins])
            cur_arr = np.array([cur_dict.get(b, 0.0) for b in all_bins])
            psi_records.append({"period": period, "psi": self.compute_psi(ref_arr, cur_arr)})

        psi_df = pl.DataFrame(psi_records)

        # -- Conversion pandas minimale pour seaborn ------------------
        agg_pd  = agg.to_pandas()
        psi_pd  = psi_df.to_pandas()
        global_dr = X[self.cible_col].cast(pl.Float64).mean()

        # -- Graphe 1 : volumes + PSI ---------------------------------
        fig, ax1 = plt.subplots(figsize=(13, 5))
        sns.lineplot(
            data=agg_pd, x="period", y="pct_obs", hue="bin", marker="o", ax=ax1
        )
        ax1.axhline(min_pop, color="grey", linestyle="--", linewidth=1, label="Seuil 5 %")
        ax1.set_ylabel("Share of population")
        ax1.set_xlabel("Period")
        ax1.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax1.tick_params(axis="x", rotation=45)
        ax1.set_title("Evolution of population shares by class with PSI")

        ax2 = ax1.twinx()
        ax2.plot(
            psi_pd["period"], psi_pd["psi"],
            color="black", linestyle="--", marker="s", label="PSI",
        )
        ax2.axhline(0.10, color="orange", linestyle=":")
        ax2.axhline(0.25, color="red",    linestyle=":")
        ax2.set_ylabel("PSI")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2, labels1 + labels2,
            bbox_to_anchor=(1.02, 1), loc="upper left",
        )
        plt.tight_layout()
        plt.show()

        # -- Graphe 2 : taux de défaut par bin ------------------------
        plt.figure(figsize=(13, 5))
        sns.lineplot(
            data=agg_pd[agg_pd["n_obs"] >= min_obs],
            x="period", y="default_rate", hue="bin", marker="o",
        )
        plt.gca().yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        plt.axhline(global_dr, color="black", linestyle="--", label="DR global")
        plt.title("Evolution of default rate by class")
        plt.ylabel("Default rate (%)")
        plt.xlabel("Period")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # DISTRIBUTION D'UNE VARIABLE CATEGORIELLE
    # ------------------------------------------------------------------

    def plot_categorical_distribution(self, var: str, mask=None) -> None:
        X = self._filter(mask)
        cramers_v, _ = self.v_cramer_t_tschuprow(X[var])

        # Tableau de contingence normalisé par ligne (normalize='index')
        ct = (
            X.select([var, self.cible_col])
            .group_by([var, self.cible_col])
            .agg(pl.len().alias("n"))
            .pivot(on=self.cible_col, index=var, values="n")
            .fill_null(0)
        )
        target_cols = [c for c in ct.columns if c != var]
        ct_pd = ct.to_pandas().set_index(var)
        ct_pd[target_cols] = ct_pd[target_cols].div(
            ct_pd[target_cols].sum(axis=1), axis=0
        ) * 100

        table = ct_pd.reset_index().melt(
            id_vars=var, var_name=self.cible_col, value_name="percentage"
        )

        plt.figure(figsize=(8, 5))
        ax = sns.barplot(
            x=var, y="percentage", hue=self.cible_col,
            data=table, palette="Set2",
        )
        plt.title("Default distribution (%) by categorical variable")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", label_type="edge", fontsize=8)
        plt.suptitle(f"V Cramer: {cramers_v:.3f}", y=1.02, fontsize=10)
        plt.show()

    # ------------------------------------------------------------------
    # CRAMÉR'S V & TSCHUPROW'S T
    # ------------------------------------------------------------------

    def v_cramer_t_tschuprow(
        self, var_col: pl.Series
    ) -> tuple[float, float]:
        """Prend une pl.Series ; utilise self.X[self.cible_col] comme cible."""
        # Tableau de contingence via Polars puis conversion numpy pour scipy
        temp = pl.DataFrame(
            {"_var": var_col.cast(pl.String), "_target": self.X[self.cible_col]}
        )
        ct_pl = (
            temp.group_by(["_var", "_target"])
            .agg(pl.len().alias("n"))
            .pivot(on="_target", index="_var", values="n")
            .fill_null(0)
        )
        ct_np = ct_pl.drop("_var").to_numpy().astype(float)

        n = ct_np.sum()
        chi2, _, _, _ = chi2_contingency(ct_np, correction=False)
        r, k = ct_np.shape

        denom_v = n * (min(r - 1, k - 1))
        cramers_v = float(np.sqrt(chi2 / denom_v)) if denom_v > 0 else float("nan")

        denom_t = n * np.sqrt((r - 1) * (k - 1))
        tschuprows_t = float(np.sqrt(chi2 / denom_t)) if denom_t > 0 else float("nan")

        return cramers_v, tschuprows_t

    # ------------------------------------------------------------------
    # DISCRETISATION MANUELLE
    # ------------------------------------------------------------------

    def discretise_with_manual_thresholds(
        self,
        var_quant: str,
        thresholds: list,
        labels: list | None = None,
        missing_label: str = "Missing",
    ) -> pl.Series:
        """Discrétise une variable quantitative avec des seuils manuels.

        Equivalent à pd.cut(..., right=False) → left_closed=True dans Polars.
        Les NaN (null) sont préservés puis remplacés par missing_label.
        Met à jour self.X et retourne la Series résultante.
        """
        if sorted(thresholds) != thresholds:
            raise ValueError("Les seuils doivent être strictement croissants.")

        result = (
            self.X[var_quant]
            .cut(breaks=thresholds, labels=labels, left_closed=True)
            .cast(pl.String)
            .fill_null(missing_label)
            .alias(var_quant)
        )
        self.X = self.X.with_columns(result)
        return result

    # ------------------------------------------------------------------
    # FUSION DE MODALITES
    # ------------------------------------------------------------------

    def merge_modalities(self, col: str, mapping: dict) -> None:
        """Fusionne des modalités via un dictionnaire {ancienne: nouvelle}.

        Lève ValueError si des valeurs restent non mappées (null après replace).
        Met à jour self.X.
        """
        merged = (
            self.X[col]
            .cast(pl.String)
            .replace(mapping)
        )

        nulls_after = merged.is_null()
        if nulls_after.any():
            unmapped = (
                self.X.filter(nulls_after)[col]
                .unique()
                .to_list()
            )
            raise ValueError(f"Modalités non mappées détectées : {unmapped}")

        self.X = self.X.with_columns(
            merged.cast(pl.Categorical).alias(col)
        )

    # ------------------------------------------------------------------
    # EXTRACTION ET APPLICATION DES SEUILS
    # ------------------------------------------------------------------

    def extract_binning_thresholds(self, X_binned: pl.DataFrame) -> dict:
        """Extrait les seuils depuis les libellés de bins de type [a, b).

        Retourne : {variable: {'cuts': [...], 'has_missing': bool}}
        """
        thresholds = {}
        for var in X_binned.columns:
            categories = (
                X_binned[var]
                .cast(pl.String)
                .unique()
                .drop_nulls()
                .to_list()
            )
            has_missing = "Missing" in categories
            bounds: set[float] = set()

            for cat in categories:
                if cat == "Missing":
                    continue
                match = re.match(r"[\[\(]([^,]+),\s*([^\]\)]+)[\]\)]", cat)
                if match:
                    left, right = match.groups()
                    if left.strip() not in ("-inf", "-Inf"):
                        bounds.add(float(left))
                    if right.strip() not in ("inf", "Inf"):
                        bounds.add(float(right))

            thresholds[var] = {"cuts": sorted(bounds), "has_missing": has_missing}

        return thresholds

    def apply_binning_thresholds(
        self,
        X_binned: pl.DataFrame,
        X_new: pl.DataFrame,
        suffix: str = "",
    ) -> pl.DataFrame:
        """Applique les seuils extraits de X_binned à X_new.

        Retourne un nouveau pl.DataFrame avec les colonnes binnées.
        """
        thresholds = self.extract_binning_thresholds(X_binned)
        cols = []

        for var, info in thresholds.items():
            binned = (
                X_new[var]
                .cut(breaks=info["cuts"], left_closed=True)
                .cast(pl.String)
                .fill_null("Missing")
                .cast(pl.Categorical)
                .alias(var + suffix)
            )
            cols.append(binned)

        return pl.DataFrame(cols)
