"""
calibration_tests.py
====================

Boîte à outils pour l'évaluation de la calibration d'un score de crédit.

Implémente le protocole en six étapes décrit dans la section « Le score
prédit-il le bon niveau de risque ? » du mémoire :

    1. Courbe de calibration lissée + ECE / MCE          -> calibration_curve, ece_mce
    2. Intercept et pente de calibration + test de Cox   -> calibration_intercept_slope, cox_calibration_test
    3. Test de Spiegelhalter (global)                    -> spiegelhalter_test
       Test de Hosmer-Lemeshow (pour mémoire)            -> hosmer_lemeshow_test
    4. Test de Jeffreys par classe de risque             -> jeffreys_test, grade_report
    5. Feux tricolores                                   -> traffic_light
    6. Décomposition de Brier                            -> brier_decomposition

Deux corrections propres au design de l'étude :
    - bootstrap par grappes (emprunteur) pour tous les intervalles de confiance,
      la pseudo-indépendance des observations empilées à deux dates étant fausse
                                                         -> cluster_bootstrap_ci
    - seuil critique de Vasicek intégrant une corrélation d'actifs rho
                                                         -> vasicek_critical_default_rate

Dépendances : numpy, scipy, pandas, statsmodels, matplotlib.

Convention : `y` est le vecteur binaire de défaut observé à 12 mois,
`p` le vecteur de probabilités de défaut prédites (dans ]0, 1[).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

__all__ = [
    "logit",
    "expit",
    "calibration_curve",
    "ece_mce",
    "calibration_intercept_slope",
    "cox_calibration_test",
    "spiegelhalter_test",
    "hosmer_lemeshow_test",
    "jeffreys_test",
    "binomial_test",
    "vasicek_critical_default_rate",
    "traffic_light",
    "grade_report",
    "brier_decomposition",
    "cluster_bootstrap_ci",
    "plot_calibration_curve",
    "full_calibration_report",
]

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #

def _clip(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Borne les probabilités pour éviter les logits infinis."""
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def logit(p):
    p = _clip(p)
    return np.log(p / (1.0 - p))


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _check(y, p):
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError("y et p doivent avoir la même longueur.")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y doit être binaire (0/1).")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p doit être dans [0, 1].")
    return y, p


# --------------------------------------------------------------------------- #
# 1. Courbe de calibration, ECE / MCE
# --------------------------------------------------------------------------- #

def calibration_curve(
    y,
    p,
    n_bins: int = 20,
    strategy: str = "quantile",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Courbe de calibration par regroupement (reliability diagram).

    strategy : "quantile" (bins d'effectif égal, recommandé sur PD faibles)
               ou "uniform" (bins de largeur égale).

    Renvoie un DataFrame avec, par bin : effectif, PD moyenne prédite,
    taux de défaut observé et intervalle de Wilson.
    """
    y, p = _check(y, p)

    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    elif strategy == "uniform":
        edges = np.linspace(p.min(), p.max(), n_bins + 1)
    else:
        raise ValueError("strategy doit valoir 'quantile' ou 'uniform'.")

    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        d = int(y[m].sum())
        lo, hi = _wilson_ci(d, n, alpha)
        rows.append(
            dict(
                bin=b,
                n=n,
                n_defauts=d,
                pd_moyenne=float(p[m].mean()),
                taux_observe=d / n,
                ic_bas=lo,
                ic_haut=hi,
            )
        )
    return pd.DataFrame(rows)


def _wilson_ci(d: int, n: int, alpha: float = 0.05):
    """Intervalle de Wilson (Newcombe, 1998), robuste aux proportions faibles."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    phat = d / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    demi = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - demi), min(1.0, centre + demi))


def ece_mce(y, p, n_bins: int = 20, strategy: str = "quantile") -> dict:
    """Expected et Maximum Calibration Error.

    ECE = somme pondérée des |taux observé - PD moyenne| par bin.
    MCE = maximum de ces écarts.
    """
    tab = calibration_curve(y, p, n_bins=n_bins, strategy=strategy)
    w = tab["n"] / tab["n"].sum()
    ecart = (tab["taux_observe"] - tab["pd_moyenne"]).abs()
    return {
        "ECE": float((w * ecart).sum()),
        "MCE": float(ecart.max()),
        "n_bins_effectifs": int(len(tab)),
    }


# --------------------------------------------------------------------------- #
# 2. Intercept, pente de calibration et test de Cox
# --------------------------------------------------------------------------- #

@dataclass
class CalibrationRegression:
    intercept: float          # alpha, pente libre
    intercept_se: float
    slope: float              # beta
    slope_se: float
    slope_ci: tuple
    intercept_large: float    # calibration-in-the-large (beta fixe a 1, offset)
    intercept_large_se: float
    loglik_full: float
    loglik_null: float

    def to_dict(self):
        return asdict(self)


def calibration_intercept_slope(y, p) -> CalibrationRegression:
    """Régression de calibration : logit P(Y=1) = alpha + beta * logit(p_hat).

    beta = 1 et alpha = 0 caractérisent la calibration dite *weak*
    (Van Calster et al., 2016). beta < 1 signale un surajustement :
    les prédictions sont trop dispersées.

    L'intercept dit *in-the-large* est estimé séparément en contraignant
    beta = 1 par un offset, ce qui isole le biais de niveau.
    """
    y, p = _check(y, p)
    lp = logit(p)

    X = sm.add_constant(lp, has_constant="add")
    full = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    null = sm.GLM(
        y,
        np.ones((len(y), 1)),
        family=sm.families.Binomial(),
        offset=lp,
    ).fit()

    # modèle totalement contraint (alpha=0, beta=1) : pas de paramètre libre
    ll_constrained = float(np.sum(y * np.log(_clip(p)) + (1 - y) * np.log(1 - _clip(p))))

    ci = full.conf_int()[1]
    return CalibrationRegression(
        intercept=float(full.params[0]),
        intercept_se=float(full.bse[0]),
        slope=float(full.params[1]),
        slope_se=float(full.bse[1]),
        slope_ci=(float(ci[0]), float(ci[1])),
        intercept_large=float(null.params[0]),
        intercept_large_se=float(null.bse[0]),
        loglik_full=float(full.llf),
        loglik_null=ll_constrained,
    )


def cox_calibration_test(y, p) -> dict:
    """Test de Cox (1958) : H0 : (alpha, beta) = (0, 1), 2 degrés de liberté.

    Rapport de vraisemblance entre la régression de calibration libre et le
    modèle sans paramètre libre. Rejeter H0 signifie que le score n'est pas
    calibré au sens *weak*.
    """
    reg = calibration_intercept_slope(y, p)
    lr = 2.0 * (reg.loglik_full - reg.loglik_null)
    lr = max(lr, 0.0)
    return {
        "statistique_LR": float(lr),
        "ddl": 2,
        "p_value": float(stats.chi2.sf(lr, 2)),
        "intercept": reg.intercept,
        "pente": reg.slope,
        "pente_ic95": reg.slope_ci,
        "intercept_in_the_large": reg.intercept_large,
    }


# --------------------------------------------------------------------------- #
# 3. Tests globaux : Spiegelhalter et Hosmer-Lemeshow
# --------------------------------------------------------------------------- #

def spiegelhalter_test(y, p) -> dict:
    """Test de Spiegelhalter (1986), Z asymptotiquement normal sous H0.

    Repose sur la décomposition du score de Brier : sous H0, l'espérance de
    (y - p)^2 vaut p(1 - p). La statistique est standardisée sans aucun
    découpage en groupes, ce qui la rend préférable à Hosmer-Lemeshow.
    """
    y, p = _check(y, p)
    num = np.sum((y - p) * (1.0 - 2.0 * p))
    den = np.sqrt(np.sum((1.0 - 2.0 * p) ** 2 * p * (1.0 - p)))
    if den < EPS:
        return {"Z": np.nan, "p_value": np.nan, "note": "variance nulle"}
    z = num / den
    return {
        "Z": float(z),
        "p_value": float(2 * stats.norm.sf(abs(z))),
        "brier": float(np.mean((y - p) ** 2)),
    }


def hosmer_lemeshow_test(y, p, g: int = 10) -> dict:
    """Test de Hosmer-Lemeshow (1980), chi2 à g-2 degrés de liberté.

    Reporté par convention. Sa conclusion dépend du nombre de groupes
    (Bertolini et al., 2000) et il rejette H0 pour des écarts négligeables
    dès que l'effectif est grand (Nattino, Pennell & Lemeshow, 2020).
    À ne pas utiliser comme critère de décision sur un score d'octroi.
    """
    y, p = _check(y, p)
    tab = calibration_curve(y, p, n_bins=g, strategy="quantile")
    obs = tab["n_defauts"].to_numpy(dtype=float)
    n = tab["n"].to_numpy(dtype=float)
    att = (tab["pd_moyenne"] * tab["n"]).to_numpy(dtype=float)
    denom = att * (1.0 - att / n)
    ok = denom > EPS
    chi2 = float(np.sum((obs[ok] - att[ok]) ** 2 / denom[ok]))
    ddl = max(int(ok.sum()) - 2, 1)
    return {
        "chi2": chi2,
        "ddl": ddl,
        "p_value": float(stats.chi2.sf(chi2, ddl)),
        "n_groupes": int(ok.sum()),
    }


# --------------------------------------------------------------------------- #
# 4. Tests par classe de risque
# --------------------------------------------------------------------------- #

def jeffreys_test(n_defauts: int, n_obs: int, pd_estimee: float) -> float:
    """Test de Jeffreys, prescrit par la BCE pour le backtesting des PD.

    Prior de Jeffreys Beta(1/2, 1/2) -> posterior Beta(D + 1/2, N - D + 1/2).
    H0 unilatérale : la PD appliquée est supérieure à la vraie PD
    (le modèle est conservateur). La p-valeur est la fonction de répartition
    du posterior évaluée en la PD estimée. Une p-valeur faible signale une
    sous-estimation du risque.
    """
    if n_obs <= 0:
        return np.nan
    a = n_defauts + 0.5
    b = n_obs - n_defauts + 0.5
    return float(stats.beta.cdf(pd_estimee, a, b))


def binomial_test(n_defauts: int, n_obs: int, pd_estimee: float) -> float:
    """Test binomial exact unilatéral (sous-estimation du risque).

    Suppose l'indépendance des défauts, hypothèse fausse en crédit :
    à utiliser en complément, jamais seul (BCBS, 2005).
    """
    if n_obs <= 0:
        return np.nan
    return float(stats.binom.sf(n_defauts - 1, n_obs, pd_estimee))


def vasicek_critical_default_rate(pd_estimee: float, rho: float, q: float = 0.999) -> float:
    """Taux de défaut critique sous modèle à un facteur (Vasicek).

    DR_q = Phi( (Phi^-1(PD) + sqrt(rho) * Phi^-1(q)) / sqrt(1 - rho) )

    Intègre la corrélation d'actifs rho, et élargit donc considérablement le
    seuil de rejet par rapport au binomial (Hamerle, Liebig & Rösch, 2003 ;
    Tasche, 2005). Valeurs de rho typiques : 0.04 à 0.12 sur du retail /
    professionnels selon la formule CRR.
    """
    if not (0 <= rho < 1):
        raise ValueError("rho doit être dans [0, 1[.")
    num = stats.norm.ppf(_clip(np.array([pd_estimee]))[0]) + np.sqrt(rho) * stats.norm.ppf(q)
    return float(stats.norm.cdf(num / np.sqrt(1.0 - rho)))


def traffic_light(
    n_defauts: int,
    n_obs: int,
    pd_estimee: float,
    rho: float = 0.08,
    seuils: Sequence[float] = (0.95, 0.99, 0.999),
) -> str:
    """Feux tricolores étendus (Blochwitz, Hohl, Tasche & Wehn ; BCBS, 2005).

    Classe la classe de risque en VERT / JAUNE / ORANGE / ROUGE selon le
    quantile de Vasicek dépassé par le taux de défaut observé. Présentation
    plus lisible pour un comité qu'une p-valeur brute.
    """
    if n_obs <= 0:
        return "NA"
    dr = n_defauts / n_obs
    couleurs = ["VERT", "JAUNE", "ORANGE", "ROUGE"]
    zone = 0
    for i, q in enumerate(seuils):
        if dr > vasicek_critical_default_rate(pd_estimee, rho, q):
            zone = i + 1
    return couleurs[zone]


def grade_report(
    y,
    p,
    grade,
    rho: float = 0.08,
    methode_multiplicite: str = "holm",
) -> pd.DataFrame:
    """Tableau de backtesting par classe de risque.

    Colonnes : effectif, défauts, PD moyenne prédite, taux observé,
    intervalle de Wilson, p-valeur de Jeffreys (brute et corrigée de la
    multiplicité), p-valeur binomiale, feu tricolore.

    La correction de multiplicité (Holm, 1979) est indispensable dès que le
    système compte une dizaine de classes : sans elle, on rejette au moins une
    classe presque à coup sûr par le seul jeu du hasard.
    """
    y, p = _check(y, p)
    grade = np.asarray(grade).ravel()

    rows = []
    for g in pd.unique(grade):
        m = grade == g
        n = int(m.sum())
        d = int(y[m].sum())
        pd_hat = float(p[m].mean())
        lo, hi = _wilson_ci(d, n)
        rows.append(
            dict(
                classe=g,
                n=n,
                n_defauts=d,
                pd_predite=pd_hat,
                taux_observe=d / n if n else np.nan,
                ic_bas=lo,
                ic_haut=hi,
                p_jeffreys=jeffreys_test(d, n, pd_hat),
                p_binomial=binomial_test(d, n, pd_hat),
                dr_critique_95=vasicek_critical_default_rate(pd_hat, rho, 0.95),
                dr_critique_999=vasicek_critical_default_rate(pd_hat, rho, 0.999),
                feu=traffic_light(d, n, pd_hat, rho=rho),
            )
        )

    tab = pd.DataFrame(rows).sort_values("pd_predite").reset_index(drop=True)
    tab["p_jeffreys_ajustee"] = _adjust_pvalues(tab["p_jeffreys"].to_numpy(), methode_multiplicite)
    return tab


def _adjust_pvalues(pvals: np.ndarray, methode: str = "holm") -> np.ndarray:
    """Correction de multiplicité (Holm, 1979 ; Bonferroni)."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if methode == "bonferroni":
        return np.minimum(pvals * m, 1.0)
    if methode != "holm":
        raise ValueError("methode doit valoir 'holm' ou 'bonferroni'.")
    ordre = np.argsort(pvals)
    ajust = np.empty(m)
    courant = 0.0
    for rang, i in enumerate(ordre):
        val = (m - rang) * pvals[i]
        courant = max(courant, val)
        ajust[i] = min(courant, 1.0)
    return ajust


# --------------------------------------------------------------------------- #
# 5. Décomposition du score de Brier
# --------------------------------------------------------------------------- #

def brier_decomposition(y, p, n_bins: int = 20, strategy: str = "quantile") -> dict:
    """Décomposition de Murphy (1973) : Brier = fiabilité - résolution + incertitude.

    - fiabilité   : composante de CALIBRATION, à minimiser
    - résolution  : composante de DISCRIMINATION, à maximiser
    - incertitude : taux de défaut de base, indépendant du modèle

    Permet de montrer qu'un modèle parcimonieux perd peu en résolution tout en
    gagnant en fiabilité — argument central si l'on compare plusieurs grilles.
    """
    y, p = _check(y, p)
    tab = calibration_curve(y, p, n_bins=n_bins, strategy=strategy)
    n_tot = len(y)
    ybar = float(y.mean())

    w = tab["n"].to_numpy(dtype=float) / n_tot
    fiabilite = float(np.sum(w * (tab["pd_moyenne"] - tab["taux_observe"]) ** 2))
    resolution = float(np.sum(w * (tab["taux_observe"] - ybar) ** 2))
    incertitude = ybar * (1 - ybar)

    return {
        "brier": float(np.mean((y - p) ** 2)),
        "fiabilite_calibration": fiabilite,
        "resolution_discrimination": resolution,
        "incertitude": float(incertitude),
        "brier_reconstitue": fiabilite - resolution + incertitude,
    }


# --------------------------------------------------------------------------- #
# 6. Bootstrap par grappes (emprunteur)
# --------------------------------------------------------------------------- #

def cluster_bootstrap_ci(
    y,
    p,
    cluster,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> dict:
    """Intervalle de confiance par bootstrap par grappes.

    Ré-échantillonne les EMPRUNTEURS avec remise, pas les observations. C'est
    la correction adaptée à un empilement sur deux dates d'observation avec
    fenêtres de performance chevauchantes : les tests usuels supposent une
    indépendance qui n'est pas vérifiée et sous-estiment la variance.

    stat_fn : fonction (y, p) -> scalaire, par exemple
              lambda y, p: calibration_intercept_slope(y, p).slope
    """
    y, p = _check(y, p)
    cluster = np.asarray(cluster).ravel()

    ids, inverse = np.unique(cluster, return_inverse=True)
    positions = [np.flatnonzero(inverse == k) for k in range(len(ids))]

    rng = np.random.default_rng(seed)
    stats_boot = []
    for _ in range(n_boot):
        tirage = rng.integers(0, len(ids), size=len(ids))
        idx = np.concatenate([positions[k] for k in tirage])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stats_boot.append(float(stat_fn(y[idx], p[idx])))
        except Exception:
            continue

    stats_boot = np.asarray(stats_boot, dtype=float)
    stats_boot = stats_boot[np.isfinite(stats_boot)]
    if len(stats_boot) < 10:
        return {"estimation": float(stat_fn(y, p)), "ic_bas": np.nan, "ic_haut": np.nan}

    return {
        "estimation": float(stat_fn(y, p)),
        "ic_bas": float(np.quantile(stats_boot, alpha / 2)),
        "ic_haut": float(np.quantile(stats_boot, 1 - alpha / 2)),
        "n_boot_valides": int(len(stats_boot)),
    }


# --------------------------------------------------------------------------- #
# Graphique
# --------------------------------------------------------------------------- #

def plot_calibration_curve(
    y,
    p,
    n_bins: int = 20,
    lowess_frac: float = 0.6,
    titre: str = "Courbe de calibration",
    chemin: str | None = None,
    echelle_logit: bool = True,
):
    """Diagramme de fiabilité : lissage loess sur l'échelle logit + points binnés.

    Le lissage est estimé sur l'échelle logit, où la relation est
    approximativement linéaire pour des PD faibles, puis retransformé.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    y, p = _check(y, p)
    tab = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")

    lp = logit(p)
    # it=0 : les itérations robustes de lowess écrasent la courbe vers zéro
    # sur une cible binaire rare.
    lisse = lowess(y, lp, frac=lowess_frac, it=0, return_sorted=True)
    x_lisse = expit(lisse[:, 0]) if echelle_logit else lisse[:, 0]
    y_lisse = np.clip(lisse[:, 1], 0, 1)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    lim = max(float(np.max(p)), float(tab["taux_observe"].max())) * 1.15

    ax.plot([0, lim], [0, lim], ls="--", lw=1, color="0.5", label="calibration parfaite")
    ax.plot(x_lisse, y_lisse, lw=2, color="#1f4e79", label="lissage loess")
    err_bas = np.clip(tab["taux_observe"] - tab["ic_bas"], 0, None)
    err_haut = np.clip(tab["ic_haut"] - tab["taux_observe"], 0, None)
    ax.errorbar(
        tab["pd_moyenne"],
        tab["taux_observe"],
        yerr=[err_bas, err_haut],
        fmt="o", ms=4, lw=1, color="#c00000", capsize=2,
        label=f"{len(tab)} groupes d'effectif égal (IC 95 % de Wilson)",
    )

    ind = ece_mce(y, p, n_bins=n_bins)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Probabilité de défaut prédite")
    ax.set_ylabel("Taux de défaut observé à 12 mois")
    ax.set_title(f"{titre}\nECE = {ind['ECE']:.4f}   MCE = {ind['MCE']:.4f}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if chemin:
        fig.savefig(chemin, dpi=200)
    return fig, ax


# --------------------------------------------------------------------------- #
# Rapport complet
# --------------------------------------------------------------------------- #

def full_calibration_report(
    y,
    p,
    grade=None,
    cluster=None,
    rho: float = 0.08,
    n_bins: int = 20,
    n_boot: int = 500,
    seed: int | None = 0,
) -> dict:
    """Exécute le protocole complet et renvoie un dictionnaire de résultats.

    Si `cluster` est fourni (identifiant emprunteur), la pente et l'intercept
    de calibration sont accompagnés d'un IC bootstrap par grappes.
    Si `grade` est fourni, le tableau de backtesting par classe est ajouté.
    """
    y, p = _check(y, p)
    res: dict = {}

    res["etape_1_courbe"] = calibration_curve(y, p, n_bins=n_bins)
    res["etape_1_ece"] = ece_mce(y, p, n_bins=n_bins)

    reg = calibration_intercept_slope(y, p)
    res["etape_2_regression"] = reg.to_dict()
    res["etape_2_cox"] = cox_calibration_test(y, p)

    if cluster is not None:
        res["etape_2_pente_bootstrap"] = cluster_bootstrap_ci(
            y, p, cluster,
            lambda yy, pp: calibration_intercept_slope(yy, pp).slope,
            n_boot=n_boot, seed=seed,
        )
        res["etape_2_intercept_bootstrap"] = cluster_bootstrap_ci(
            y, p, cluster,
            lambda yy, pp: calibration_intercept_slope(yy, pp).intercept_large,
            n_boot=n_boot, seed=seed,
        )

    res["etape_3_spiegelhalter"] = spiegelhalter_test(y, p)
    res["etape_3_hosmer_lemeshow"] = hosmer_lemeshow_test(y, p, g=10)

    if grade is not None:
        res["etape_4_5_par_classe"] = grade_report(y, p, grade, rho=rho)

    res["etape_6_brier"] = brier_decomposition(y, p, n_bins=n_bins)
    return res


# --------------------------------------------------------------------------- #
# Démonstration reproductible
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # 6 000 emprunteurs observés à 2 dates -> 12 000 observations empilées
    n_emp, n_dates = 6000, 2
    emprunteur = np.repeat(np.arange(n_emp), n_dates)
    effet = rng.normal(0, 0.6, n_emp)[emprunteur]        # hétérogénéité intra-emprunteur
    score = rng.normal(0, 1, n_emp * n_dates) + effet

    pd_vraie = expit(-4.2 + 1.0 * score)
    y_obs = rng.binomial(1, pd_vraie)

    # Cas A : modèle bien calibré.
    # Cas B : modèle trop plat qui SOUS-ESTIME le risque des mauvaises classes
    #         (pente de calibration > 1) — le cas que le test de Jeffreys, unilatéral,
    #         est précisément conçu pour détecter.
    p_bon = pd_vraie
    p_mauvais = expit(-4.6 + 0.80 * score)

    grades = pd.qcut(p_bon, 8, labels=[f"C{i+1}" for i in range(8)])

    for nom, p_hat in [("BIEN CALIBRÉ", p_bon), ("SOUS-ESTIMATION DU RISQUE", p_mauvais)]:
        print("\n" + "=" * 68)
        print(f"  {nom}   (taux de défaut observé : {y_obs.mean():.3%})")
        print("=" * 68)

        r = full_calibration_report(
            y_obs, p_hat, grade=grades, cluster=emprunteur, n_boot=200
        )

        print("\n[1] ECE / MCE                :", {k: round(v, 5) for k, v in r["etape_1_ece"].items()})
        cox = r["etape_2_cox"]
        print(f"[2] Intercept in-the-large   : {r['etape_2_regression']['intercept_large']:+.4f}")
        print(f"    Pente de calibration     : {cox['pente']:.4f}  IC95 iid {tuple(round(v,3) for v in cox['pente_ic95'])}")
        print(f"    IC95 bootstrap grappes   : "
              f"({r['etape_2_pente_bootstrap']['ic_bas']:.3f}, {r['etape_2_pente_bootstrap']['ic_haut']:.3f})")
        print(f"    Test de Cox              : LR = {cox['statistique_LR']:.2f}, p = {cox['p_value']:.2e}")
        print(f"[3] Spiegelhalter            : Z = {r['etape_3_spiegelhalter']['Z']:+.3f}, "
              f"p = {r['etape_3_spiegelhalter']['p_value']:.3e}")
        print(f"    Hosmer-Lemeshow          : chi2 = {r['etape_3_hosmer_lemeshow']['chi2']:.2f}, "
              f"p = {r['etape_3_hosmer_lemeshow']['p_value']:.3e}")
        print("[4/5] Backtesting par classe :")
        cols = ["classe", "n", "n_defauts", "pd_predite", "taux_observe",
                "p_jeffreys", "p_jeffreys_ajustee", "dr_critique_95", "feu"]
        print(r["etape_4_5_par_classe"][cols].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))
        print("[6] Décomposition de Brier   :",
              {k: round(v, 6) for k, v in r["etape_6_brier"].items()})

    plot_calibration_curve(y_obs, p_mauvais, titre="Courbe de calibration — modèle sous-estimant le risque",
                           chemin="calibration_demo.png")
    print("\nGraphique écrit : calibration_demo.png")
