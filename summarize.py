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
