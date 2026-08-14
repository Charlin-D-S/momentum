# Contrôle de calibration dans Streamlit — mode d'emploi

Trois fichiers, deux dépendances à installer.

| Fichier | Rôle |
|---|---|
| `calibration_tests.py` | Les tests statistiques. Aucune dépendance Streamlit, réutilisable en batch ou en notebook. |
| `calibration_streamlit.py` | L'affichage. Une fonction : `afficher_calibration`. |
| `app_demo_calibration.py` | Exemple exécutable sur données simulées. |

```bash
pip install statsmodels streamlit      # scipy, pandas, numpy, altair viennent avec
streamlit run app_demo_calibration.py  # pour voir le rendu tout de suite
```

---

## L'appel

Un DataFrame, les noms de colonnes, et tout s'affiche.

```python
import polars as pl
from calibration_streamlit import afficher_calibration

df = pl.read_parquet("scores_production_2025.parquet")

afficher_calibration(
    df,
    y="defaut_12m",           # obligatoire — défaut observé, 0/1
    p="pd_predite",           # obligatoire — PD prédite, dans [0, 1]
    classe="classe_risque",   # facultatif
    emprunteur="id_client",   # facultatif
    segment="perimetre",      # facultatif
)
```

C'est tout. La page affiche la synthèse, la courbe de calibration, la pente et l'intercept,
les tests globaux, le tableau réglementaire par classe, la décomposition de Brier et les exports.

Le DataFrame peut être **polars** (`DataFrame` ou `LazyFrame`), **pandas**, ou un simple
dictionnaire de tableaux. La conversion est interne.

---

## Les deux colonnes obligatoires

| Argument | Contenu attendu | Contrôle effectué |
|---|---|---|
| `y` | Défaut bâlois observé sur les 12 mois suivants, codé 0 / 1 | Rejet si d'autres valeurs sont présentes |
| `p` | Probabilité de défaut prédite par le modèle, dans [0, 1] | Rejet si hors bornes — attention si votre grille sort des **points de score** et non une probabilité |

Si votre modèle produit une note en points, convertissez-la avant :
`p = 1 / (1 + exp(-(points - offset) / facteur))`.

## Les trois colonnes facultatives — ce que chacune débloque

| Argument | Ce que ça active | Sans elle |
|---|---|---|
| `classe` | Onglet 4-5 : test de Jeffreys par classe, correction de Holm, feux tricolores de Vasicek | L'onglet affiche un message d'invitation |
| `emprunteur` | Bootstrap par grappes pour l'intervalle de la pente | Les intervalles supposent l'indépendance, et un avertissement le signale |
| `segment` | Un filtre déroulant en haut de page (sous-périmètre Pros-ER / Associations / SCI, ou échantillon stock 2024 / production 2025) | Pas de filtre |

**`emprunteur` mérite d'être renseignée.** Avec un empilement sur deux dates et des fenêtres de
performance de 12 mois qui se chevauchent, les observations ne sont pas indépendantes. Sans
bootstrap par grappes, l'intervalle de la pente est trop étroit et vous conclurez au rejet plus
souvent qu'il ne faut.

---

## Comparer plusieurs modèles

Passez une liste à `p` : un sélecteur apparaît en haut de page, et un onglet
« Comparaison des modèles » tabule ECE, intercept, pente, Brier, fiabilité et résolution
côte à côte.

```python
afficher_calibration(
    df,
    y="defaut_12m",
    p=["grille_logistique", "grille_existante", "challenger_xgboost"],
    classe="classe_risque",
    emprunteur="id_client",
)
```

C'est la vue qui alimente le tableau 5.6.2 du mémoire : elle montre si un modèle parcimonieux
perd en résolution ou seulement en niveau.

---

## Intégration dans une application existante

`afficher_calibration` n'appelle jamais `st.set_page_config`. Vous pouvez donc la déposer
telle quelle dans une page d'une application multipage :

```python
# pages/4_Calibration.py
import streamlit as st
from calibration_streamlit import afficher_calibration

df = st.session_state["scores"]
afficher_calibration(df, y="defaut_12m", p="pd_predite", classe="classe_risque")
```

Options utiles :

- `titre=None` supprime le titre et le sous-titre, si votre page a déjà son en-tête.
- `panneau_parametres=False` supprime les curseurs de la barre latérale et fige les valeurs
  passées en argument (`rho`, `n_bins`, `n_boot`, `alpha`).
- `cle="calib_pros"` change le préfixe des clés de widgets. **Obligatoire** si vous appelez
  la fonction plusieurs fois dans la même page, sinon Streamlit lève une erreur de clé dupliquée.

Exemple, trois sous-périmètres côte à côte :

```python
for nom, sous_df in [("Pros-ER", pros), ("Associations", asso), ("SCI", sci)]:
    st.header(nom)
    afficher_calibration(sous_df, y="defaut_12m", p="pd_predite",
                         classe="classe_risque", titre=None,
                         panneau_parametres=False, cle=f"calib_{nom}")
```

---

## Sans interface

Pour produire les tableaux du mémoire en batch, sans lancer Streamlit :

```python
from calibration_streamlit import resultats_calibration

res = resultats_calibration(df, y="defaut_12m", p="pd_predite", classe="classe_risque")
print(res["cox"]["pente"], res["ece"]["ECE"])
res["par_classe"].to_csv("tableau_5_6_1.csv", sep=";", index=False)
```

Ou directement via `calibration_tests.full_calibration_report`, qui renvoie le même contenu
sans dépendance Streamlit du tout.

---

## Paramètres de réglage

| Paramètre | Défaut | Remarque |
|---|---|---|
| `rho` | 0.08 | Corrélation d'actifs des seuils de Vasicek. À aligner sur la formule CRR de votre périmètre. Mettre 0 revient au test binomial sous indépendance. |
| `n_bins` | 20 | Groupes d'effectif égal de la courbe de calibration. |
| `n_boot` | 300 | Réplications du bootstrap par grappes. 300 suffit pour un intervalle indicatif, montez à 1000 pour un chiffre publiable. |
| `alpha` | 0.05 | Seuil de significativité, utilisé pour la coloration et les verdicts. |
| `lissage` | 0.6 | Fenêtre du lissage local. Réduire si la courbe paraît trop lisse. |

---

## Performance

Tous les calculs passent par `st.cache_data` : changer d'onglet ou de curseur ne relance que
ce qui a changé. Le lissage local est désactivé au-delà de 200 000 observations, la courbe
binnée restant affichée. Le bootstrap est le seul poste réellement coûteux — mettez `n_boot=0`
pendant la mise au point.

---

## Points de lecture

- **Intercept ≠ 0, pente ≈ 1** : décalage de niveau seul. Corrigeable par un simple recalage
  de l'ordonnée à l'origine de la grille, sans toucher au classement.
- **Pente < 1** : prédictions trop dispersées, symptôme de surajustement.
- **Pente > 1** : prédictions trop plates, le risque des mauvaises classes est sous-estimé.
  C'est le cas que le test de Jeffreys, unilatéral, est fait pour attraper.
- **Jeffreys signale des classes mais tous les feux restent verts** : ce n'est pas une
  incohérence. Les seuils de Vasicek corrigés de la corrélation sont très larges — le Comité
  de Bâle le reconnaît explicitement. Faites varier `rho` pour mesurer la sensibilité du verdict.
  L'application affiche un avertissement dédié dans ce cas.
