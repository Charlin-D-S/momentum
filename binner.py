"""Binner — version Polars + Plotly.

Changements cles :
- Graphiques en Plotly (hover riche : effectifs, nb defauts, taux, PSI...).
- plot_bin_stability_over_time retourne (fig_vol, fig_dr) : go.Figure.
- plot_categorical_distribution retourne un go.Figure.
- Usage Streamlit : st.plotly_chart(fig, use_container_width=True)
- Usage notebook  : fig.show()
- Reste inchange  : compute_psi, v_cramer_t_tschuprow, discretise_with_manual_thresholds,
                    merge_modalities, extract_binning_thresholds, apply_binning_thresholds.
"""
fig1, fig2 = b.plot_bin_stability_over_time("ma_var")
st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

fig_cat = b.plot_categorical_distribution("ma_var_cat")
st.plotly_chart(fig_cat, use_container_width=True)

from __future__ import annotations

import re

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        if mask is None:
            return self.X
        return self.X.filter(mask)

    def compute_psi(
        self,
        ref_dist: np.ndarray,
        cur_dist: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        return float(
            np.sum(
                (ref_dist - cur_dist)
                * np.log((ref_dist + eps) / (cur_dist + eps))
            )
        )

    def _psi_label(self, psi: float) -> str:
        if psi < 0.10:
            return f"{psi:.4f} ✅ stable"
        if psi < 0.25:
            return f"{psi:.4f} ⚠️ attention"
        return f"{psi:.4f} 🔴 instable"

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
    ) -> tuple[go.Figure, go.Figure]:
        """Retourne (fig_vol, fig_dr) — figures Plotly interactives.

        fig_vol : volumes par bin (%) + PSI sur axe secondaire.
        fig_dr  : taux de defaut par bin.
        Hover : effectif, nb defauts, taux de defaut, part population, PSI.
        """
        X = self._filter(mask)
        global_dr = float(X[self.cible_col].cast(pl.Float64).mean())

        # -- Agregation Polars ----------------------------------------
        agg = (
            X.select([var_binned, self.cible_col, self.date_col])
            .rename({var_binned: "bin", self.cible_col: "target", self.date_col: "period"})
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

        ref_rows = agg.filter(pl.col("period") == ref_period).select(["bin", "n_obs"])
        ref_total = ref_rows["n_obs"].sum()
        ref_dict = {r["bin"]: r["n_obs"] / ref_total for r in ref_rows.to_dicts()}

        psi_records = []
        for period in sorted(agg["period"].unique().to_list()):
            cur_rows = agg.filter(pl.col("period") == period).select(["bin", "n_obs"])
            cur_total = cur_rows["n_obs"].sum()
            cur_dict = {r["bin"]: r["n_obs"] / cur_total for r in cur_rows.to_dicts()}
            all_bins = list(ref_dict.keys())
            psi_val = self.compute_psi(
                np.array([ref_dict.get(b, 0.0) for b in all_bins]),
                np.array([cur_dict.get(b, 0.0) for b in all_bins]),
            )
            psi_records.append({"period": period, "psi": psi_val})

        psi_df = pl.DataFrame(psi_records).sort("period")

        # -- Conversion pandas pour iteration Plotly ------------------
        agg_pd = agg.to_pandas()
        psi_pd = psi_df.to_pandas()

        # Ordre des bins (tri sur la borne gauche si format [a, b))
        def _bin_sort_key(b: str) -> float:
            m = re.match(r"[\[\(]([^,]+),", str(b))
            if m:
                v = m.group(1).strip()
                return float("-inf") if v in ("-inf", "-Inf") else float(v)
            return float("inf")

        bins_ordered = sorted(agg_pd["bin"].unique(), key=_bin_sort_key)

        # -- Figure 1 : volumes + PSI ---------------------------------
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])

        for bin_name in bins_ordered:
            d = agg_pd[agg_pd["bin"] == bin_name].sort_values("period")
            fig1.add_trace(
                go.Scatter(
                    x=d["period"],
                    y=d["pct_obs"],
                    name=str(bin_name),
                    mode="lines+markers",
                    customdata=d[["n_obs", "n_defaults", "default_rate", "total_period"]].values,
                    hovertemplate=(
                        "<b>Période : %{x}</b><br>"
                        f"Bin : {bin_name}<br>"
                        "Part population : <b>%{y:.1%}</b><br>"
                        "Effectif : %{customdata[0]:,.0f}<br>"
                        "Nb défauts : %{customdata[1]:,.0f}<br>"
                        "Taux de défaut : %{customdata[2]:.2%}<br>"
                        "Total période : %{customdata[3]:,.0f}"
                        "<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

        # Seuil population minimum
        fig1.add_hline(
            y=min_pop, line_dash="dash", line_color="grey",
            annotation_text=f"Seuil {min_pop:.0%}",
            annotation_position="bottom right",
            secondary_y=False,
        )

        # Courbe PSI
        psi_pd["psi_label"] = psi_pd["psi"].apply(self._psi_label)
        fig1.add_trace(
            go.Scatter(
                x=psi_pd["period"],
                y=psi_pd["psi"],
                name="PSI",
                mode="lines+markers",
                marker=dict(symbol="square", color="black", size=7),
                line=dict(color="black", dash="dash"),
                customdata=psi_pd[["psi_label"]].values,
                hovertemplate=(
                    "<b>Période : %{x}</b><br>"
                    "PSI : %{customdata[0]}<br>"
                    f"Référence : {ref_period}"
                    "<extra></extra>"
                ),
            ),
            secondary_y=True,
        )

        # Seuils PSI
        for y_val, color, label in [(0.10, "orange", "PSI 0.10"), (0.25, "red", "PSI 0.25")]:
            fig1.add_hline(
                y=y_val, line_dash="dot", line_color=color,
                annotation_text=label,
                annotation_position="top right",
                secondary_y=True,
            )

        fig1.update_layout(
            title=f"Evolution des parts de population par bin — PSI (ref : {ref_period})",
            hovermode="x unified",
            legend=dict(orientation="v", x=1.08),
            height=500,
        )
        fig1.update_yaxes(title_text="Part de population (%)", tickformat=".0%", secondary_y=False)
        fig1.update_yaxes(title_text="PSI", secondary_y=True)
        fig1.update_xaxes(title_text="Période")

        # -- Figure 2 : taux de défaut --------------------------------
        fig2 = go.Figure()

        agg_filtered = agg_pd[agg_pd["n_obs"] >= min_obs]

        for bin_name in bins_ordered:
            d = agg_filtered[agg_filtered["bin"] == bin_name].sort_values("period")
            if d.empty:
                continue
            fig2.add_trace(
                go.Scatter(
                    x=d["period"],
                    y=d["default_rate"],
                    name=str(bin_name),
                    mode="lines+markers",
                    customdata=d[["n_obs", "n_defaults", "pct_obs", "total_period"]].values,
                    hovertemplate=(
                        "<b>Période : %{x}</b><br>"
                        f"Bin : {bin_name}<br>"
                        "Taux de défaut : <b>%{y:.2%}</b><br>"
                        "Effectif : %{customdata[0]:,.0f}<br>"
                        "Nb défauts : %{customdata[1]:,.0f}<br>"
                        "Part population : %{customdata[2]:.1%}<br>"
                        "Total période : %{customdata[3]:,.0f}"
                        "<extra></extra>"
                    ),
                )
            )

        # DR global
        fig2.add_hline(
            y=global_dr, line_dash="dash", line_color="black",
            annotation_text=f"DR global {global_dr:.2%}",
            annotation_position="bottom right",
        )

        fig2.update_layout(
            title="Evolution du taux de défaut par bin",
            hovermode="x unified",
            legend=dict(orientation="v", x=1.08),
            height=500,
            yaxis=dict(title="Taux de défaut (%)", tickformat=".2%"),
            xaxis=dict(title="Période"),
        )

        return fig1, fig2

    # ------------------------------------------------------------------
    # DISTRIBUTION D'UNE VARIABLE CATEGORIELLE
    # ------------------------------------------------------------------

    def plot_categorical_distribution(self, var: str, mask=None) -> go.Figure:
        """Retourne un go.Figure — barres groupees avec hover riche."""
        X = self._filter(mask)
        cramers_v, tschuprow_t = self.v_cramer_t_tschuprow(X[var])

        # Tableau de contingence : effectifs bruts + pourcentages
        ct_raw = (
            X.select([var, self.cible_col])
            .group_by([var, self.cible_col])
            .agg(pl.len().alias("n"))
            .pivot(on=self.cible_col, index=var, values="n")
            .fill_null(0)
        )
        target_cols = [c for c in ct_raw.columns if c != var]
        ct_pd = ct_raw.to_pandas().set_index(var)
        row_totals = ct_pd[target_cols].sum(axis=1)
        ct_pct = ct_pd[target_cols].div(row_totals, axis=0) * 100

        fig = go.Figure()
        for t in target_cols:
            fig.add_trace(
                go.Bar(
                    name=f"Cible = {t}",
                    x=ct_pct.index.astype(str),
                    y=ct_pct[t],
                    customdata=np.column_stack([
                        ct_pd[t].values,
                        row_totals.values,
                    ]),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        f"Cible : {t}<br>"
                        "Part : <b>%{y:.1f}%</b><br>"
                        "Effectif : %{customdata[0]:,.0f}<br>"
                        "Total modalité : %{customdata[1]:,.0f}"
                        "<extra></extra>"
                    ),
                    text=ct_pct[t].apply(lambda v: f"{v:.1f}%"),
                    textposition="outside",
                )
            )

        fig.update_layout(
            barmode="group",
            title=(
                f"Distribution de la cible par modalité — {var}<br>"
                f"<sup>V de Cramér : {cramers_v:.3f} | T de Tschuprow : {tschuprow_t:.3f}</sup>"
            ),
            yaxis=dict(title="Part (%)"),
            xaxis=dict(title=var),
            legend=dict(orientation="h", y=-0.15),
            height=480,
        )

        return fig

    # ------------------------------------------------------------------
    # CRAMER'S V & TSCHUPROW'S T
    # ------------------------------------------------------------------

    def v_cramer_t_tschuprow(self, var_col: pl.Series) -> tuple[float, float]:
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
        if sorted(thresholds) != thresholds:
            raise ValueError("Les seuils doivent etre strictement croissants.")
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
        merged = self.X[col].cast(pl.String).replace(mapping)
        nulls_after = merged.is_null()
        if nulls_after.any():
            unmapped = self.X.filter(nulls_after)[col].unique().to_list()
            raise ValueError(f"Modalites non mappees : {unmapped}")
        self.X = self.X.with_columns(merged.cast(pl.Categorical).alias(col))

    # ------------------------------------------------------------------
    # EXTRACTION ET APPLICATION DES SEUILS
    # ------------------------------------------------------------------

    def extract_binning_thresholds(self, X_binned: pl.DataFrame) -> dict:
        thresholds = {}
        for var in X_binned.columns:
            categories = (
                X_binned[var].cast(pl.String).unique().drop_nulls().to_list()
            )
            has_missing = "Missing" in categories
            bounds: set[float] = set()
            for cat in categories:
                if cat == "Missing":
                    continue
                m = re.match(r"[\[\(]([^,]+),\s*([^\]\)]+)[\]\)]", cat)
                if m:
                    left, right = m.groups()
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
