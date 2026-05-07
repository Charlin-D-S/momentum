    import numpy as np
import pandas as pd

def construire_woe_df(
    df: pd.DataFrame,
    y: pd.Series,
    variables: list[str],
) -> pd.DataFrame:
    """
    Construit un DataFrame de WoE à partir d'un DataFrame de variables catégorielles.
    Pour chaque variable, calcule le WoE de chaque modalité puis mappe sur les individus.

    Paramètres
    ----------
    df        : DataFrame contenant les variables catégorielles (modalités = tranches).
    y         : Série binaire (0/1) — cible défaut.
    variables : Liste des variables à traiter.

    Retourne
    --------
    DataFrame (n_individus × n_variables) contenant les WoE.
    """
    n_def   = y.sum()
    n_sain  = len(y) - n_def
    eps     = 1e-10
    woe_df  = pd.DataFrame(index=df.index)

    for var in variables:
        stats = (
            pd.DataFrame({"modalite": df[var], "y": y})
            .groupby("modalite")["y"]
            .agg(n_def="sum", n_tot="count")
        )
        stats["n_sain"] = stats["n_tot"] - stats["n_def"]
        stats["p1"]     = (stats["n_def"]  / n_def ).clip(lower=eps)
        stats["p0"]     = (stats["n_sain"] / n_sain).clip(lower=eps)
        stats["woe"]    = np.log(stats["p1"] / stats["p0"])

        # Mapping modalité → WoE sur chaque individu
        woe_df[var] = df[var].map(stats["woe"])

    return woe_df
def iv_sur_deciles(scores: np.ndarray, y: np.ndarray, n_deciles: int = 10) -> float:
    """
    Calcule l'IV d'un vecteur de scores continus en le découpant en déciles.
    Gère les cas dégénérés (division par zéro, deciles vides).
    """
    n_def = y.sum()
    n_sain = len(y) - n_def

    if n_def == 0 or n_sain == 0:
        return 0.0

    # Découpage en déciles (quantiles) — duplicates="drop" évite les erreurs
    # si le score est peu dispersé
    deciles = pd.qcut(scores, q=n_deciles, duplicates="drop")

    df = pd.DataFrame({"decile": deciles, "y": y})
    stats = df.groupby("decile", observed=True)["y"].agg(
        n_def="sum",
        n_tot="count",
    )
    stats["n_sain"] = stats["n_tot"] - stats["n_def"]

    # Proportions — on clip pour éviter log(0)
    eps = 1e-10
    p1 = (stats["n_def"] / n_def).clip(lower=eps)
    p0 = (stats["n_sain"] / n_sain).clip(lower=eps)

    iv = ((p1 - p0) * np.log(p1 / p0)).sum()
    return float(iv)


def forward_miv(
    woe_df: pd.DataFrame,
    y: pd.Series,
    seuil_miv: float = 0.01,
    n_max: int = 20,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Sélection forward de variables par Marginal Information Value.

    Paramètres
    ----------
    woe_df   : DataFrame (n_individus × n_variables) contenant les WoE
               de chaque individu pour chaque variable candidate.
               Les colonnes sont les noms des variables.
    y        : Série binaire (0/1) — cible défaut.
    seuil_miv: MIV minimale pour qu'une variable soit ajoutée.
    n_max    : Nombre maximum de variables à sélectionner.
    n_deciles: Nombre de déciles pour le calcul de l'IV du score.

    Retourne
    --------
    DataFrame avec colonnes :
        etape, variable, iv_cumule, miv, variables_modele
    """
    y_arr = np.asarray(y)
    candidats = list(woe_df.columns)
    score_courant = np.zeros(len(y_arr))
    iv_courant = 0.0
    modele = []
    historique = []

    for etape in range(1, n_max + 1):
        meilleure_var = None
        meilleure_miv = -np.inf
        meilleur_iv = None

        for var in candidats:
            score_test = score_courant + woe_df[var].values
            iv_test = iv_sur_deciles(score_test, y_arr, n_deciles)
            miv = iv_test - iv_courant

            if miv > meilleure_miv:
                meilleure_miv = miv
                meilleure_var = var
                meilleur_iv = iv_test

        # Critère d'arrêt
        if meilleure_miv < seuil_miv:
            print(f"Arrêt étape {etape} — MIV maximale ({meilleure_miv:.4f}) < seuil ({seuil_miv})")
            break

        # Mise à jour
        modele.append(meilleure_var)
        candidats.remove(meilleure_var)
        score_courant = score_courant + woe_df[meilleure_var].values
        iv_courant = meilleur_iv

        historique.append({
            "etape": etape,
            "variable": meilleure_var,
            "iv_cumule": round(meilleur_iv, 4),
            "miv": round(meilleure_miv, 4),
            "variables_modele": modele.copy(),
        })

        print(f"Étape {etape:2d} | +{meilleure_var:<30s} | IV cumulé = {meilleur_iv:.4f} | MIV = {meilleure_miv:.4f}")

    return pd.DataFrame(historique)
    
    
    
    
    def summarize_model(self,logit_model): 
        params = logit_model.params
        var_set = { col.split('&')[0] for col in params.index if col!='const'}
        self.X_train['count']=1
        
        summary = {}
        for col in var_set : 
            summary[col] = {}
            
            coefs = params[[i for i in params.index if i.startswith(col)]]
            summary[col]['coef_max'] = coefs.max()

            taux_cibles = self.X_train.groupby(col)[self.cible_col].mean().sort_values()*100
            taux_pops = self.X_train.groupby(col)['count'].sum()/self.n_train*100

            for i in range(len(taux_cibles)) : 
                index = taux_cibles.index[i]
                summary[col][index] ={}
                cle = col +"&"+ str(index)
                if cle in params.index : 
                    coef = params[cle] #pvalues
                    pvalue = logit_model.pvalues[cle] 
                else:
                    coef = 0
                    pvalue = np.nan
                taux_cible = taux_cibles[index]
                if i!=len(taux_cibles)-1:
                    ecart_relatif = (taux_cibles[taux_cibles.index[i+1]]/taux_cible-1) *100
                else:
                    ecart_relatif = np.nan

                summary[col][index]['taux_cible'] = taux_cible
                summary[col][index]['taux_pop'] = taux_pops[index]
                summary[col][index]['coef'] = coef
                summary[col][index]['pvalue'] = pvalue
                summary[col][index]['ecart_relatif'] = ecart_relatif

        sum_max_coefs = sum([ summary[col]['coef_max']  for col in var_set ])#for index in summary[col] ])
        for col in var_set :
            max = summary[col]['coef_max']
            max_contrib =0
            for index in summary[col] :
                if index != 'coef_max':
                    coef = summary[col][index]['coef']
                    x = 1000*(max-coef)/sum_max_coefs
                    if x>=max_contrib:
                        max_contrib = x
                    summary[col][index]['points_1000'] = x
            summary[col]['contribution'] = max_contrib/10

    def retrieve_var(self,col):
        terme = '&'+col.split('&')[-1]
        index = col.rfind(terme)
        if index !=1:
            return col[:index] + col[index+len(terme):]
        return col
    def build_summary_dataframe(self,logit_model):
        target_col=self.cible_col
        params = logit_model.params
        pvals = logit_model.pvalues

        var_set = {self.retrieve_var(col) for col in params.index if col!='const'}

        X_train = self.X_train.copy()
        X_train["_count"] = 1

        summary_rows = []
        var_max_coefs = {}

        # ---- 1) EXTRACTION DES INFOS ----
        for var in var_set:

            coefs = params[[i for i in params.index if i.startswith(var)]]
            var_max_coefs[var] = coefs.max()

            taux_cibles = X_train.groupby(var)[target_col].mean().sort_values() * 100
            taux_pops = X_train.groupby(var)['_count'].sum() / self.n_train * 100

            modalités = taux_cibles.index.tolist()

            for i, modal in enumerate(modalités):

                key = f"{var}&{modal}"

                coef = params[key] if key in params.index else 0
                pvalue = pvals[key] if key in pvals.index else np.nan

                taux_cible = taux_cibles[modal]
                taux_pop = taux_pops[modal]

                if i < len(modalités) - 1:
                    taux_next = taux_cibles[modalités[i+1]]
                    ecart_rel = (taux_next / taux_cible - 1) * 100 if taux_cible > 0 else np.nan
                else:
                    ecart_rel = np.nan

                summary_rows.append({
                    "variable": var,
                    "modalite": modal,
                    "taux_pop (%)": taux_pop,
                    "taux_cible (%)": taux_cible,
                    "coef": coef,
                    "pvalue": pvalue,
                    "ecart_relatif (%)": ecart_rel
                })

        df = pd.DataFrame(summary_rows)

        # ---- 2) Points /1000 ----
        total_max = sum(var_max_coefs.values())

        df["points/1000"] = df.apply(
            lambda r: 1000 * (var_max_coefs[r["variable"]] - r["coef"]) / total_max,
            axis=1
        )

        # ---- 3) Contribution echelle ----
        contributions = (
            df.groupby("variable")["points/1000"]
            .max() / 10
        )

        df["contrib_echelle (%)"] = df["variable"].map(contributions)

        # ---- 3) Contribution score ----
        moy_pts = df.groupby('variable')['points/1000'].mean()

        df['var'] = df.apply(lambda r: (r['taux_pop (%)']/100) * (moy_pts[r["variable"]] - r["points/1000"])**2,
                                                axis=1)
        ecart_type = np.sqrt(df.groupby('variable')['var'].sum())
        ecart_type_sum = np.sum(ecart_type)
        df['contrib_score (%)'] = df.apply(lambda r: 100 *ecart_type[r["variable"]]/ ecart_type_sum,
                                        axis=1)

        df.drop('var',axis=1,inplace=True)

        # ---- 5) ARRONDIR TOUTES LES COLONNES NUMÉRIQUES ----
        # num_cols = df.select_dtypes(include=[np.number]).columns
        # df[num_cols] = df[num_cols].round(2)

        return df
