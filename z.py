from scipy import stats
import numpy as np

def test_calibration_globale(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Teste H0 : PD moyenne == taux de défaut observé.
    Test Z bilatéral.
    """
    n        = len(y_true)
    pd_moy   = y_prob.mean()
    taux_obs = y_true.mean()

    # Écart absolu
    écart = pd_moy - taux_obs

    # Erreur standard sous H0
    se = np.sqrt(pd_moy * (1 - pd_moy) / n)

    # Statistique Z
    z_stat = écart / se

    # p-value bilatérale
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    if p_value >= alpha:
        statut = "✅ PD moyenne == taux observé (H0 non rejetée)"
    elif écart > 0:
        statut = "🔴 Sur-estimation significative (H0 rejetée)"
    else:
        statut = "🟡 Sous-estimation significative (H0 rejetée)"

    return {
        "n":           n,
        "pd_moyenne":  round(pd_moy,   4),
        "taux_defaut": round(taux_obs, 4),
        "écart":       round(écart,    4),
        "z_stat":      round(z_stat,   3),
        "p_value":     round(p_value,  4),
        "statut":      statut,
    }
