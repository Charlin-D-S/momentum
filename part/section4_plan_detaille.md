# Section 4 — Inférence statistique : plan détaillé

**Titre** : « Une grille logistique à discrétisation pilotée par un boosting de souches »
Budget corps ~6 pages. `[DONNÉE]` = à renseigner avec les chiffres de la banque. `[à confirmer]` = hypothèse à valider.

---

## Introduction de section (~0,2 p.)

- Objectif : une grille de score d'octroi interprétable sur défaut bâlois à 12 mois.
- Fil chronologique : le modèle cible, le challenger profond qui guide la sélection, la réduction et la discrétisation par un boosting de souches, la sélection finale, l'estimation et l'inférence, la validation.
- Une phrase assumant l'ouverture sur le challenger : outil de construction, pas modèle livré ; la section se referme sur la grille.

---

## 4.1 Le modèle retenu : une grille logistique sur classes imbriquées (~1 p.)

**§1 — Forme et notations.** Échantillon $(x_i, y_i)_{i=1}^{n}$, $y_i = 1$ si défaut à 12 mois. $p_i = \mathbb{P}(Y_i=1\mid x_i)$. Modèle : $\operatorname{logit}(p_i) = \beta_0 + \sum_j \beta_j z_i^{(j)}$, avec $z^{(j)}$ des indicatrices imbriquées.

**§2 — Pourquoi discrétiser.** Performance : capter non-linéarités et effets de seuil. Robustesse : absorbe l'imprécision (valeur près d'une borne reste dans sa classe), tolère erreurs de codage et omissions (classe « manquante »), neutralise les valeurs extrêmes (classes bornées). Pré-répond à l'exigence de robustesse du gabarit.

**§3 — Pourquoi des classes imbriquées** (nested dummies, Scallan 2011). Coefficient = écart incrémental avec la classe voisine ; monotonie ↔ contrainte de signe ; test avec la voisine, pas une référence arbitraire ; évite les grilles à trous.

**§4 — Principe de la mise en points.** Extraction de la partie linéaire puis changement d'échelle affine vers [0, 1000]. La conversion concrète coefficient → point se fait après estimation (4.5). Règle exacte en annexe A.

**Encadré 1** — Codage en classes imbriquées et principe de la mise en points.
**Figure 4.1** — WoE par classe d'une variable fil rouge `[DONNÉE]`.
**Citations** : Scallan (2011), Siddiqi (2006).

---

## 4.2 Le challenger XGBoost de profondeur > 1 : guide de la sélection et borne de performance (~0,8 p.)

**§1 — Présentation.** XGBoost profond, boosting de gradient régularisé (Friedman 2001 ; Chen et Guestrin 2016), modèle le plus performant du scoring dans la littérature récente (Gunnarsson et al. 2021).

**§2 — Rôle 1 : guider la sélection.** Ses valeurs de SHAP (Lundberg et Lee 2017 ; TreeSHAP, Lundberg et al. 2020) donnent une importance globale cohérente, utilisée pour arbitrer le choix des variables dans les clusters de Spearman (4.3).

**§3 — Rôle 2 : borne de performance.** La profondeur > 1 capture les interactions et fixe le plafond. L'opposition à la version additive de profondeur 1 (4.3) quantifiera l'apport des interactions, exploitée en 5.1bis.

**Encadré 2** — Objectif régularisé de XGBoost et valeurs de SHAP.
**Citations** : Friedman (2001), Chen et Guestrin (2016), Gunnarsson et al. (2021), Lundberg et Lee (2017), Lundberg et al. (2020).

---

## 4.3 Réduction et discrétisation : d'un large ensemble à une vingtaine de variables (~1,5 p., cœur)

**§1 — Étape 1 : clusters de Spearman (seuil 80 %) et choix guidé.** Sur variables continues et ordinales ; **[Contribution]** encodage ordinal par sens du risque, uniquement ici, pour intégrer les qualitatives. Règle de choix dans chaque cluster : d'abord un pré-filtre métier retirant les variables jugées trop volatiles ; ensuite on retient la plus importante au sens SHAP du challenger profond ; puis la suivante seulement si sa corrélation avec une variable déjà retenue est sous le seuil fixé. **[Contribution]**.

**§2 — Étape 2 : discrétisation par un second XGBoost de profondeur 1 (GAM).** Un boosting de souches, additif et sans interaction (Nori et al. 2019), optimisé pour être monotone et stable : il découpe les variables quantitatives et regroupe les modalités des catégorielles. Critique rapide des méthodes de binning univariées, qui traitent chaque variable en isolation, au profit d'un **binning multivarié** où chaque coupure vise à améliorer ce que les précédentes ont déjà produit (logique de boosting sur les résidus). Extraction des coupures, fusion et regroupement sous contrainte de monotonie, contrôle de stabilité, significativité et sens du risque, validation métier des qualitatives. **[Contribution]** hyperparamètres réglés en amont ; **[Contribution]** implémentation dans le package interne AGBoost.

**§3 — Étape 3 : clusters de Cramér's V (seuil 0,4, méthodo interne).** Sur les variables discrétisées. Élimination des redondances résiduelles, sortie à environ 20 variables.

**Figure 4.2** — Courbe de contribution $h_j$ d'une variable (GAM) `[DONNÉE]`.
**Citations** : Nori et al. (2019) ; Navas-Palencia (2020) pour le binning univarié critiqué.

---

## 4.4 Sélection finale : forward par IV marginale (méthode de Scallan) et LASSO (~0,9 p.)

**§1 — Pourquoi pas le forward sur le Gini.** Le stepwise qui optimise le Gini ou la déviance sur-ajuste et surestime les coefficients (Harrell 2001 ; Scallan 2011).

**§2 — [Contribution] Forward par IV marginale du score.** Procédure exacte de Scallan (2011), sans régression dans la boucle. Sur chacun des 3 sous-échantillons : le score courant est augmenté du WoE de la variable candidate ($\text{score} = \text{score} + \operatorname{WoE}(\text{candidate})$) ; ce score est binné en quartiles ; on calcule son IV, puis son IV marginale (gain d'IV apporté par la variable). On ajoute la variable dont l'IV marginale moyenne sur les 3 sous-échantillons est la plus élevée, jusqu'à épuisement de l'IV marginale significative. Avantage : environ 5 fois plus rapide et plus robuste que le forward sur le Gini.

**§3 — LASSO en complément.** Pénalité $L^1$ (Tibshirani 1996) pour confirmer la parcimonie ; lecture du chemin de régularisation.

**Encadré 3** — WoE, score additif de WoE, IV et IV marginale.
**Figure 4.3** — Décroissance de l'IV marginale au fil des étapes `[DONNÉE]`.
**Citations** : Scallan (2011), Harrell (2001), Tibshirani (1996).

---

## 4.5 Estimation et inférence de la grille (~0,9 p.)

**§1 — Estimation finale par maximum de vraisemblance sans LASSO.** Sur les variables sélectionnées, ré-ajustement par MV : le LASSO sélectionne mais rétrécit les coefficients, le refit MV les rétablit et rend les tests valides. Écriture de la log-vraisemblance.

**§2 — Du coefficient au point.** On extrait le coefficient associé à chaque modalité (ce sont eux qui figurent sur les tableaux présentés), puis le point sur 1000 qui lui correspond par la mise à l'échelle de 4.1.

**§3 — Remarque : régresser directement sur les WoE.** On aurait pu, comme Siddiqi (2006), régresser directement sur les WoE plutôt que sur les indicatrices ; on préfère les indicatrices pour lire un coefficient et un point par modalité.

**§4 — Inférence.** Erreurs-types via l'information de Fisher, distribution asymptotique. Test de **Wald** (significativité d'une classe = écart de risque avec la voisine) et de **déviance** (apport d'une variable). Fonde la validation.

**Encadré 4** — Maximum de vraisemblance et tests de Wald et de déviance (résumé).
**Citations** : Siddiqi (2006), Scallan (2011).

---

## 4.6 Validation et lecture métier du modèle retenu (~0,8 p.)

**§1 — Protocole.** Apprentissage vs *out-of-time* (production 2025), aussi *out-of-population*.

**§2 — Discrimination.** Gini/AUC, KS. Pros-ER en fil rouge `[DONNÉE]`, Associations et SCI en tableau `[DONNÉE]`.

**§3 — Stabilité et significativité.** PSI `[DONNÉE]`. Significativité des coefficients (tests de 4.5) `[DONNÉE]`.

**§4 — Lecture métier.** Extrait de la grille : modalité, coefficient, points sur 1000, pour la variable fil rouge ; signes, monotonie, cohérence `[DONNÉE]`. Grille complète en annexe G.

**§5 — Robustesse succincte.** Imprécision, erreurs de codage et omissions déjà absorbées par la discrétisation (4.1) ; renvoi à 5.5. Calibration secondaire, seuil fixé par le métier.

**Table 4.1** — Gini, KS, PSI par sous-périmètre `[DONNÉE]`.
**Table 4.2** — Extrait de grille (modalité, coefficient, points/1000), variable fil rouge `[DONNÉE]`.

---

## Annexes de la section 4

**A. Codage imbriqué et mise en points.** Équivalence codage imbriqué / par attribut. Règle de mise en points : base, facteur, exemple chiffré `[DONNÉE]`.

**B. Challengers et SHAP.** Hyperparamètres des deux XGBoost, profondeur > 1 et profondeur 1 `[DONNÉE]`. TreeSHAP. Graphes d'importance SHAP `[DONNÉE]`.

**C. Discrétisation et AGBoost.** Dérivation au second ordre de l'objectif XGBoost (poids de feuille, gain). Hyperparamètres du boosting de souches et contraintes de monotonie `[DONNÉE]`. Rôle du package interne AGBoost, **[Contribution]** implémentation. Tables de binning par variable `[DONNÉE]`.

**D. Réduction de la redondance.** Matrices de Spearman et clusters (80 %) `[DONNÉE]`. Matrices de Cramér's V et clusters (0,4) `[DONNÉE]`.

**E. Sélection finale.** Procédure de Scallan détaillée (score additif de WoE, binning quartile, delta score, IV marginale). Historique des IV marginales `[DONNÉE]`. Chemin LASSO `[DONNÉE]`.

**F. Inférence.** Rappels maximum de vraisemblance et information de Fisher. Formules des tests de Wald et de déviance.

**G. Validation étendue.** Courbes ROC par sous-périmètre `[DONNÉE]`. Grilles complètes (modalité, coefficient, points) et tables de coefficients avec IC `[DONNÉE]`. PSI détaillé `[DONNÉE]`.

---

## Éléments à renseigner avec les données

- Taux de défaut et volumétrie par sous-périmètre.
- Nombre de variables candidates initial, après réduction (~20), et final.
- Grille finale : modalités, coefficients, points sur 1000, par variable.
- Gini/AUC, KS en apprentissage et *out-of-time*, par sous-périmètre ; PSI.
- Hyperparamètres des deux challengers et du boosting de souches.
- Tables de binning, matrices Spearman et Cramér, historique des IV marginales, chemin LASSO.

## Points à confirmer

- Nombre de variables de la grille finale (~10-12 ?).
- Le seuil de Spearman à 80 % relève-t-il de la méthodo interne, comme le Cramér à 0,4 ?
- Le seuil de corrélation « fixé » du choix intra-cluster (§4.3.1) est-il aussi 80 %, ou un autre ?
