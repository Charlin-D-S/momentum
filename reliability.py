    def reliability_diagram(self, n_bins=20, CHR='', _cible='', _proba_theorique = ''):
        _df=self.X
        if CHR=='':
            prob_true, prob_pred = calibration_curve(self.y_true, self.y_score, n_bins=n_bins)
            
        else:
            if _df is None or _cible == '' or _proba_theorique == '':
                raise ValueError()
                
            calibration_df = _df.groupby(self.CHR_col).agg(
                y_true_mean = (_cible, 'mean'),
                pred_proba_mean=(_proba_theorique, 'mean'),
                n=('cible', 'count')
            ).reset_index()

            prob_true, prob_pred = calibration_df['y_true_mean'], calibration_df['pred_proba_mean']
            
            
        if CHR=='':
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7), sharex=True, constrained_layout=True)
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7), constrained_layout=True)

            
        ax1.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f"Model (Brier={brier_score_loss(self.y_true, self.y_score):.2%})")
        ax1.plot([0,1],[0,1], linestyle='--', color='gray', label='Perfect calibration')
        ax1.set_xlabel("Mean predicted probability")
        ax1.set_ylabel("Fraction of positives")
        ax1.set_title("Reliability diagram (calibration curve)")
        ax1.legend(loc="best")
        ax1.grid(alpha=0.3)
        
        if CHR=='':
            ax2.hist(self.y_score, bins=n_bins, color='C0', edgecolor='k', alpha=0.7)
        else:
            counts = _df[self.CHR_col].value_counts().sort_index()
            ax2.bar(counts.index.astype(str), counts.values, color='C0', edgecolor='k', alpha=0.7)
        ax2.set_xlabel("Predicted probability")
        ax2.set_ylabel("Count")
        ax2.set_title("Histogram of predicted probabilities")
        ax2.grid(alpha=0.2)

        N = len(self.y_score) if len(self.y_score) > 0 else 1
        ax_perc = ax2.twinx()

        primary_yticks = ax2.get_yticks()
        secondary_yticks = primary_yticks / N 

        ax_perc.set_yticks(primary_yticks)  # positionner aux mêmes valeurs numériques que l'axe gauche
        ax_perc.set_ylim(ax2.get_ylim())  # garder mêmes limites pour alignement visuel
        ax_perc.set_ylabel("Share of total (%)")
        ax_perc.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100.0 * (y / N):.0f}%"))
        
        plt.show()
