# challenger_benchmark

Benchmark de challengers non lineaires servant la borne de performance (section 5.1bis du memoire). Optimise, evalue et explique quatre familles sur le meme protocole, et produit les figures et l'Excel comparatifs.

## Modeles
- `logistic_regression` : logistique classique, sans binning ni stepwise (imputation mediane + indicateur de manquant + one-hot).
- `hist_gradient_boosting` : remplace la foret aleatoire, manquants et categorielles natifs.
- `xgboost` : `enable_categorical`, manquants natifs.
- `catboost` : `cat_features` natives, NaN categoriel remplace par `MISSING`.

## Installation
```bash
pip install polars fastexcel pyarrow xgboost catboost shap optuna \
            openpyxl scikit-learn matplotlib pandas joblib pyyaml
```

## Lancement
```bash
python -m challenger_benchmark challenger.yaml
```

## Entrees
- **Base** : un seul parquet, colonne `sample` (`train`/`val` -> apprentissage, `test` -> evaluation finale).
- **drivers.xlsx** : colonnes `Variable`, `Description`, `Type` (`q`/`c`), `Risk_drivers`. La liste des variables est reconstituee depuis ce fichier ; `variables: all` retient les lignes `Risk_drivers == RISK_DRIVER`.
- **mapping.json** : `{ "variable": { "code": "modalite" } }`, decode les codes en modalites reelles.

## Sorties (par `output_dir`)
- Racine : `_summary.xlsx` (performances + SHAP consolide + variables), `_comparison.png`, `_shap_consolidated.png`.
- Par modele : `roc.png`, `shap_importance.png`, `shap_impact.png` (beeswarm), `optuna_history.png`, `model.joblib`, `best_params.json`, `shap_importance.csv`.

## Decisions de conception
- **Protocole equitable** : chaque famille recoit son pretraitement propre (au meilleur de ce qu'elle permet), encapsule dans `prepare`, pour que la comparaison et les SHAP soient lisibles au niveau variable.
- **Pas de GroupKFold** (choix assume) : la CV sert au choix d'hyperparametres ; les chiffres rapportes le sont sur un test reellement externe.
- **Memoire** : scan Polars projete, un seul `collect`, liberation (`del` + `gc.collect`) entre modeles ; SHAP sur echantillon du test.
- **Limite SHAP categorielles** : sur le beeswarm, l'axe couleur encode un code sans ordre pour les nominales (XGBoost/CatBoost gardent les libelles, HistGB des codes). L'importance agregee par variable reste valide partout.

## Tests
```bash
python -m pytest tests/test_units.py -q      # unitaires
python tests/smoke_synthetic.py              # bout en bout, donnees synthetiques
```
