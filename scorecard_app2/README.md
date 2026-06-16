# BNP Paribas — Scorecard Explorer

Application Streamlit de visualisation interactive d'une scorecard de crédit et des prédictions sur portefeuille. Style et palette BNP Paribas.

## Structure

```
bnp_scorecard_app/
├── app.py                       # entrée Streamlit (page d'accueil)
├── pages/
│   ├── 1_Scorecard.py          # grille de score interactive
│   └── 2_Segments.py           # analyse par segment + zonage décisionnel
├── components/
│   ├── charts.py                # graphiques Plotly thèmés
│   ├── filters.py               # widgets de filtres
│   └── profile_cards.py         # cartes de profils à la frontière
├── utils/
│   ├── scorecard_engine.py      # parser, scorer, décomposition
│   ├── data_loader.py           # chargement Polars + cache Streamlit
│   └── theme.py                 # palette BNP + CSS
├── data/
│   ├── scorecard.parquet        # règles (Variables, Label, coef, points_1000)
│   └── dataset_predit.parquet   # individus à scorer (variables brutes)
├── tests/
│   ├── generate_synthetic_data.py
│   ├── test_engine.py
│   ├── test_integration.py
│   └── test_streamlit_app.py
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Données

Deux fichiers parquet dans `data/` :

**`scorecard.parquet`** — règles de score (une ligne par bin) :

| Variables | Label | coef | points_1000 |
|---|---|---|---|
| Intercept | `-` | -2.0 | 500 |
| anciennete | `[-inf, 1.0)` | 0.50 | -45 |
| anciennete | `[10.0, inf) + MISSING` | -0.45 | 40 |
| type_client | `['Profession libérale']` | -0.40 | 36 |

Conventions :
- `Label = "-"` : intercept (constante)
- `Label = "[a, b)"` ou `"[a, b) + MISSING"` : bin numérique
- `Label = "['A', 'B']"` ou avec `'MISSING'` : bin catégoriel
- `sigmoid(score_logit) = P(défaut)` → coef positif = bin plus risqué

**`dataset_predit.parquet`** — individus à scorer, valeurs brutes (pas de bins).

## Configuration

Éditer `utils/data_loader.py`, section `ZONE À AJUSTER` :

```python
ID_COL = "id_client"
TARGET_COL = "defaut_obs"          # None si pas de cible observée
FILTER_VARS = []                    # vide = toutes les variables scorecard
```

## Lancement

```bash
streamlit run app.py
```

## Tests

```bash
# 1) générer des données de test (à exécuter une fois)
PYTHONPATH=. python tests/generate_synthetic_data.py

# 2) tests unitaires du moteur
PYTHONPATH=. python tests/test_engine.py

# 3) tests d'intégration des composants
PYTHONPATH=. python tests/test_integration.py

# 4) tests Streamlit (exécution réelle des pages)
PYTHONPATH=. python tests/test_streamlit_app.py
```

## Architecture mémoire et cache

- **`st.cache_resource`** : scorecard et dataset enrichi (immuables, partagés entre sessions)
- **`st.cache_data`** : options de filtres (clé : nom de variable)
- **Polars LazyFrame** : le pipeline de scoring reste paresseux jusqu'au `collect()` final
- **Sélection projective** : on ne garde que les colonnes utiles (`meta + raw + _bin + _pts + scores`)

## Pages

### 1 — Scorecard

Tableau interactif de la grille, barres horizontales des points par bin (vert positif / rouge négatif), barres d'importance globale (étendue des points par variable). Filtres + recherche + export CSV.

### 2 — Segments

Filtres en haut (multiselect par variable, modalités = bins de la scorecard). Statistiques du segment, calibration (quantile, nombre de bins ajustable), évolution du taux de défaut par score, zonage décisionnel bi- ou tricolore avec seuils en probabilité, et cartes dépliantes des profils à la frontière (décomposition variable par variable).
