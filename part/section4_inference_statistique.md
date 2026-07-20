# 4. Inférence statistique

## Une grille logistique à discrétisation pilotée par un boosting de souches

On cherche une grille de score d'octroi lisible, capable de trier les demandes selon leur risque de défaut bâlois à douze mois. La contrainte double, performance et interprétabilité, gouverne chaque choix. La présentation suit l'ordre réel de construction : on définit le modèle cible, on introduit le challenger qui guide la sélection, on réduit et on discrétise les variables par un boosting de souches, on arrête la grille finale, on l'estime et on l'éprouve. Le challenger est un outil de construction, non le modèle livré ; la section se referme sur la grille.

Le périmètre comprend trois sous-populations, qui donnent trois grilles. Le corps du texte détaille le périmètre standard, le Pros-ER, qui pèse l'essentiel de la production ; les Associations et les SCI sont résumés en tableaux, leurs résultats complets figurant en annexe et en section 5.

---

## 4.1 Le modèle retenu : une grille logistique sur classes

On dispose de l'échantillon $(x_i, y_i)_{i=1}^{n}$, où $y_i = 1$ marque un défaut observé dans les douze mois et $y_i = 0$ son absence. On suppose les observations indépendantes et $Y_i \mid x_i \sim \mathcal{B}(p_i)$, de sorte que $p_i = \mathbb{P}(Y_i = 1 \mid x_i)$. Le vecteur $x_i$ rassemble les variables candidates.

La probabilité de défaut est modélisée par une régression logistique sur variables discrétisées :

$$\operatorname{logit}(p_i) = \ln\!\left(\frac{p_i}{1 - p_i}\right) = \beta_0 + \sum_{m} \beta_m\, z_{im},$$

où les $z_{im}$ sont des indicatrices de classes. Pour chaque variable, la classe la plus risquée est placée en référence : les coefficients mesurent l'écart de risque des autres classes par rapport à elle.

Le passage aux classes répond à deux besoins. Le premier est la performance : découper une variable continue transforme une relation non linéaire avec le log-odds en une fonction en escalier, que la régression logistique estime sans hypothèse de forme, et capte les effets de seuil que le modèle brut lisserait. Le second est la robustesse. La discrétisation absorbe l'imprécision de mesure, puisqu'une valeur proche d'une borne reste dans sa classe. Elle tolère les erreurs de codage et les omissions, une valeur manquante devenant une classe à part entière. Elle neutralise les valeurs extrêmes, rangées dans une classe bornée. Cette étape répond donc déjà, par construction, à une part de l'exigence de robustesse du gabarit.

Le modèle se déploie en grille de points. On extrait la partie linéaire du modèle, puis on la met à l'échelle sur l'intervalle [0, 1000], choisi pour la lisibilité métier. La conversion concrète de chaque coefficient en points intervient après l'estimation (4.5).

> **Encadré 1 — Poids de preuve et Information Value**
>
> Pour une classe $c$ de bons $g_c$ et de mauvais $b_c$, avec $G$ et $B$ les totaux, le poids de preuve vaut $\operatorname{WoE}_c = \ln\!\big[(g_c/G)/(b_c/B)\big]$. Il place la classe sur l'échelle du log-odds. Le pouvoir discriminant d'une variable se résume par son Information Value, $\operatorname{IV} = \sum_c (g_c/G - b_c/B)\operatorname{WoE}_c$. Ces deux quantités servent à la discrétisation et à la sélection.

**Figure 4.1** — Poids de preuve par classe d'une variable fil rouge `[DONNÉE]`.

---

## 4.2 Le challenger XGBoost de profondeur supérieure à un

Un modèle non linéaire performant est entraîné en amont, non pour être déployé, mais pour guider la construction de la grille et en apprécier le plafond de performance.

Ce challenger est un XGBoost profond, un boosting de gradient régularisé qui ajuste séquentiellement des arbres sur les résidus des précédents (Friedman, 2001 ; Chen et Guestrin, 2016). La littérature récente le désigne comme le modèle le plus performant du scoring de crédit (Gunnarsson et al., 2021), ce qui en fait une référence de plafond crédible.

Son premier rôle est de guider la sélection. Les valeurs de SHAP, fondées sur la théorie des jeux et calculées efficacement sur les arbres par l'algorithme TreeSHAP (Lundberg et Lee, 2017 ; Lundberg et al., 2020), fournissent une importance globale cohérente de chaque variable. Cette importance sert à trancher entre variables corrélées lors de la réduction (4.3).

Son second rôle est de borner la performance. Sa profondeur supérieure à un lui permet de capturer les interactions entre variables et d'atteindre le plafond que la grille cherchera à approcher. La comparaison avec sa version additive de profondeur un, décrite plus loin, quantifie l'apport de ces interactions, analyse développée en section 5.

> **Encadré 2 — Objectif de XGBoost et valeurs de SHAP**
>
> À l'étape $t$, l'arbre $f_t$ minimise un objectif régularisé $\mathcal{L}^{(t)} = \sum_i l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$, avec $\Omega(f) = \gamma T + \tfrac{1}{2}\lambda \lVert w \rVert^2$, où $T$ est le nombre de feuilles et $w$ leurs poids. La valeur de SHAP d'une variable pour une prédiction est sa contribution marginale moyenne, au sens de Shapley ; sa moyenne en valeur absolue sur l'échantillon donne l'importance globale utilisée ici.

---

## 4.3 Réduction et discrétisation : d'un large ensemble à une vingtaine de variables

Le passage de l'ensemble des candidates à une vingtaine de variables enchaîne trois étapes : une première réduction guidée par le challenger, la discrétisation par un boosting de souches, puis une seconde réduction sur les variables discrétisées.

**Première réduction, par clusters de Spearman.** Les variables continues et ordinales sont regroupées en clusters selon leur corrélation de Spearman, sensible aux associations monotones, au seuil de 80 %. Les variables qualitatives sont d'abord encodées de façon ordinale selon leur sens du risque, c'est-à-dire l'ordre de leur poids de preuve, afin d'entrer dans cette analyse ; cet encodage ne sert qu'ici. **[Contribution]** Le choix dans chaque cluster suit trois temps : un pré-filtre métier retire les variables jugées trop volatiles ; on retient ensuite la plus importante au sens SHAP du challenger ; on ajoute la suivante seulement si sa corrélation avec une variable déjà retenue reste sous 80 %.

**Discrétisation, par un second XGBoost de profondeur un.** Les variables retenues sont découpées par un boosting de souches, un XGBoost dont la profondeur est fixée à un. Chaque arbre teste alors une seule variable, et la somme des souches ne contient aucune interaction. En regroupant les souches par variable, le modèle s'écrit :

$$F(x) = \beta_0 + \sum_{j} h_j(x_j),$$

où $F(x)$ est le log-odds de défaut prédit par le boosting et chaque $h_j$ une fonction constante par morceaux de la seule variable $x_j$. Cette forme additive appartient à la lignée des modèles additifs boostés (Nori et al., 2019). Le procédé découpe les variables quantitatives et regroupe aussi les modalités des variables qualitatives.

Ce binning multivarié se distingue des méthodes univariées. Ces dernières découpent chaque variable individuellement, sur un critère lié à la cible : découpage par arbre de type CART, fusion de classes de type ChiMerge, ou optimisation directe de l'Information Value (Siddiqi, 2006 ; Navas-Palencia, 2020). Ce sont les méthodes les plus répandues, mais comme chaque variable est traitée en isolation sur la même cible, les découpages obtenus sont généralement très corrélés entre eux à travers cette cible. Dans le boosting de souches, à l'inverse, chaque coupure est choisie pour améliorer ce que les précédentes ont déjà produit, sur toutes les variables simultanément.

On extrait les points de coupure, puis on fusionne et regroupe les tranches. Chaque classe finale doit respecter plusieurs conditions : un effectif minimal, un risque monotone le long des classes, une différence significative avec la classe voisine, un sens métier cohérent et une stabilité entre échantillons. **[Contribution]** Les hyperparamètres du boosting sont réglés en amont du binning, de sorte que les coupures ne soient pas un sous-produit d'un modèle par défaut. **[Contribution]** La méthodologie est implémentée dans le package interne AGBoost.

**Seconde réduction, par clusters de Cramér.** Les variables discrétisées sont regroupées selon le $V$ de Cramér, adapté aux variables catégorielles, au seuil de 0,4 fixé par la méthodologie interne. Cette étape retire les redondances résiduelles apparues après discrétisation. On en sort avec une vingtaine de variables.

**Figure 4.2** — Fonction de contribution $h_j$ d'une variable, issue du boosting de souches `[DONNÉE]`.

---

## 4.4 Sélection finale : forward par Information Value marginale et LASSO

De la vingtaine de variables restantes, la grille finale en retient douze. La sélection combine une sélection ascendante fondée sur l'Information Value marginale et un contrôle par pénalisation.

Le stepwise qui optimise directement le Gini ou la déviance est écarté. Il sur-ajuste, surestime les coefficients et rend le choix des variables instable en présence de colinéarité (Harrell, 2001 ; Scallan, 2011). On lui préfère un critère fondé sur l'Information Value marginale du score.

> **[Contribution]** La sélection reprend la procédure de Scallan (2011), sans régression dans la boucle. Sur chacun de trois sous-échantillons, le score courant est augmenté du poids de preuve de la variable candidate, $\text{score} \leftarrow \text{score} + \operatorname{WoE}(\text{candidate})$. Ce score est découpé en quartiles, et l'on calcule son Information Value, puis son Information Value marginale, soit le gain d'IV apporté par la variable. On ajoute la variable dont l'IV marginale, moyennée sur les trois sous-échantillons, est la plus élevée, jusqu'à ce qu'aucun gain marginal ne soit plus significatif. Le recours à trois sous-échantillons limite l'effet de l'échantillonnage sur le classement. La procédure est environ cinq fois plus rapide que le forward sur le Gini, et plus robuste.

Un LASSO complète la sélection. Sa pénalité $L^1$ annule les coefficients les moins informatifs et confirme la parcimonie retenue ; le chemin de régularisation en donne une lecture continue (Tibshirani, 1996).

> **Encadré 3 — Score additif de poids de preuve et IV marginale**
>
> Au pas $t$, le score d'un individu est la somme des poids de preuve de ses modalités : $s_i^{(t)} = s_i^{(t-1)} + \operatorname{WoE}(\text{candidate})$. Ce score est découpé en quartiles, sur lesquels on calcule $\operatorname{IV}(s^{(t)})$. L'IV marginale de la variable vaut $\operatorname{MIV} = \operatorname{IV}(s^{(t)}) - \operatorname{IV}(s^{(t-1)})$. La variable retenue maximise la moyenne de la MIV sur les trois sous-échantillons.

**Figure 4.3** — Décroissance de l'IV marginale au fil des étapes de sélection `[DONNÉE]`.

---

## 4.5 Estimation et inférence de la grille

Les douze variables retenues, la grille est estimée puis interprétée.

L'estimation finale se fait par maximum de vraisemblance, sans pénalisation. Le LASSO a servi à sélectionner, mais il rétrécit les coefficients ; un ré-ajustement par maximum de vraisemblance sur les seules variables retenues rétablit des coefficients non biaisés et rend les tests valides. On maximise la log-vraisemblance logistique

$$\ell(\beta) = \sum_{i=1}^{n} \big[ y_i \ln p_i + (1 - y_i)\ln(1 - p_i)\big],$$

qui n'admet pas de solution explicite et se résout numériquement par l'algorithme de Newton-Raphson, sous sa forme des moindres carrés repondérés itérés.

Avant d'estimer un coefficient par classe, un codage imbriqué a servi à vérifier que chaque classe diffère significativement de sa voisine, condition posée à la discrétisation. Le modèle final est ensuite estimé avec, pour chaque variable, la classe la plus risquée en référence, ce qui donne un coefficient interprétable par modalité.

La mise en points découle de ces coefficients. Les variables étant toutes binaires, on souhaite que le log-odds le plus élevé corresponde à 1000 points et le plus faible, associé au profil le plus risqué, à 0. Pour chaque variable $j$, on note $\beta_j^{\max}$ le coefficient de sa classe la moins risquée, et $S = \sum_j \beta_j^{\max}$ leur somme. Les points attribués à la classe $c$ de la variable $j$ valent :

$$\text{points}_{j,c} = 1000 \times \frac{\beta_j^{\max} - \beta_{j,c}}{S}.$$

Ce sont ces coefficients et ces points, par modalité, qui composent les tableaux présentés.

On aurait pu, comme Siddiqi (2006), préférer un encodage des classes en poids de preuve suivi d'une régression logistique. Le codage en indicatrices est retenu parce qu'il donne un coefficient et un point lisibles pour chaque modalité.

L'inférence s'appuie sur la théorie du maximum de vraisemblance. Les erreurs-types dérivent de l'information de Fisher, et la distribution asymptotique des estimateurs fonde deux tests. Le test de Wald juge la significativité d'une classe, soit l'écart de risque avec sa voisine. Le test de déviance juge l'apport global d'une variable. Ces tests fondent la validation de la section suivante.

> **Encadré 4 — Maximum de vraisemblance, Newton-Raphson et tests**
>
> L'estimateur $\hat{\beta}$ maximise $\ell(\beta)$ ; l'équation de score $\nabla \ell(\beta) = 0$ est résolue par itérations de Newton-Raphson, $\beta^{(k+1)} = \beta^{(k)} + \mathcal{I}(\beta^{(k)})^{-1}\nabla\ell(\beta^{(k)})$, où $\mathcal{I}$ est l'information de Fisher. Sa matrice de covariance est $\mathcal{I}(\hat\beta)^{-1}$, d'où les erreurs-types $\widehat{\operatorname{se}}(\hat\beta_m)$. Le test de Wald compare $z = \hat\beta_m / \widehat{\operatorname{se}}(\hat\beta_m)$ à une loi normale ; son carré suit un $\chi^2$ à un degré de liberté. Le test de déviance compare deux modèles emboîtés par $2\big[\ell(\text{complet}) - \ell(\text{réduit})\big]$, de loi $\chi^2$.

---

## 4.6 Validation et lecture métier du modèle retenu

La validation présentée ici reste succincte ; les comparaisons au score existant et les tests de représentativité relèvent de la section 5.

Le protocole partage les données en apprentissage, test et *out-of-time*, ce dernier étant la production 2025, qui est aussi *out-of-population* puisqu'on passe du stock au flux. Le Pros-ER sert de fil rouge ; les Associations et les SCI sont donnés en tableaux, avec renvoi à l'annexe pour le détail.

La discrimination se mesure par le Gini et l'AUC, ainsi que par la statistique de Kolmogorov-Smirnov. Sur le Pros-ER, la grille atteint un Gini de `[DONNÉE]` en apprentissage, `[DONNÉE]` en test et `[DONNÉE]` en *out-of-time*, pour un KS de `[DONNÉE]`. Les trois sous-périmètres figurent dans la Table 4.1.

Pour situer la grille sous son plafond, on présente la performance de trois modèles emboîtés en complexité, sur les trois échantillons : le XGBoost sur la totalité des variables, sa version simplifiée de profondeur un, puis la régression logistique finale. Cette progression montre au lecteur ce que coûte, en discrimination, chaque pas vers l'interprétabilité (Table 4.3). L'apport détaillé des interactions est analysé en section 5.

La stabilité dans le temps est suivie par l'indice de stabilité de population, dont la valeur *out-of-time* atteint `[DONNÉE]`. La significativité des coefficients, évaluée par les tests de 4.5, confirme que chaque classe retenue diffère de sa voisine `[DONNÉE]`.

La lecture métier ferme la validation. Un extrait de grille donne, pour la variable fil rouge, la modalité, son coefficient et ses points sur 1000 (Table 4.2). Le signe et l'ordre des coefficients, contraints à la monotonie, produisent une grille cohérente avec le sens du risque attendu et défendable devant un comité de crédit. La grille complète des douze variables figure en annexe G.

La robustesse est ici esquissée, son développement revenant à la section 5.5. L'imprécision des variables, les erreurs de codage et les omissions sont déjà absorbées par la discrétisation (4.1). Un point de vigilance demeure sur l'hypothèse d'indépendance des observations : un même individu est photographié à deux dates espacées de six mois, ce qui produit deux observations corrélées. La stabilité de la grille, entre échantillons et dans le temps, doit donc être contrôlée avec soin, faute de quoi la précision des estimations serait surestimée. La stabilité de chaque variable du modèle est documentée en annexe H. La calibration est laissée volontairement secondaire, le seuil de décision étant fixé par le métier à partir d'un taux de défaut cible chez les acceptés.

**Table 4.1** — Gini, KS et PSI par sous-périmètre, en apprentissage, test et *out-of-time* `[DONNÉE]`.
**Table 4.2** — Extrait de la grille du Pros-ER : modalité, coefficient, points sur 1000, pour la variable fil rouge `[DONNÉE]`.
**Table 4.3** — Gini et KS des trois modèles (XGBoost complet, boosting de souches, régression logistique), en apprentissage, test et *out-of-time* `[DONNÉE]`.

---

# Annexes de la section 4

Les annexes complètent la section sans entrer dans le quota de pages. Elles doivent permettre à un lecteur averti de reproduire l'étude. Les emplacements `[DONNÉE]` reçoivent les résultats chiffrés issus des données de la banque.

## Annexe A — Codage des classes et mise en points

**Codage de référence.** Pour chaque variable, la classe la plus risquée est prise comme référence ; le coefficient d'une autre classe mesure son écart de risque avec cette référence, sur l'échelle du log-odds.

**Codage imbriqué et test des classes voisines.** En amont de l'estimation finale, un codage imbriqué a servi à tester la significativité de l'écart entre classes adjacentes. Pour une variable à classes ordonnées $1, \dots, K$, on pose les indicatrices $P_k = \mathbf{1}\{\text{classe} \ge k\}$ ; le coefficient de $P_k$ mesure l'écart entre les classes $k$ et $k-1$, et son test de Wald juge si ces deux classes diffèrent. Ce diagnostic conditionne la fusion des classes non distinctes.

**Mise en points.** Toutes les variables étant binaires, le score est mis à l'échelle [0, 1000]. Pour chaque variable $j$, $\beta_j^{\max}$ note le coefficient de sa classe la moins risquée et $S = \sum_j \beta_j^{\max}$ leur somme. Les points de la classe $c$ valent $\text{points}_{j,c} = 1000\,(\beta_j^{\max} - \beta_{j,c})/S$.

**Table A.1** — Exemple de mise en points sur une variable : classe, coefficient, points `[DONNÉE]`.

## Annexe B — Challengers et valeurs de SHAP

**Hyperparamètres.** Réglages des deux XGBoost, le challenger de profondeur supérieure à un et le boosting de souches : profondeur, taux d'apprentissage, nombre d'arbres, régularisation, contraintes de monotonie.

**Table B.1** — Hyperparamètres des deux modèles `[DONNÉE]`.

**TreeSHAP.** Les valeurs de SHAP attribuent à chaque variable sa contribution marginale moyenne au sens de Shapley ; TreeSHAP les calcule exactement sur les arbres en temps polynomial. L'importance globale est la moyenne des valeurs absolues sur l'échantillon.

**Figure B.1** — Importance SHAP globale des variables du challenger `[DONNÉE]`.

## Annexe C — Discrétisation et package AGBoost

**Objectif de XGBoost.** Développement au second ordre : le poids optimal d'une feuille $I_j$ est $w_j^{\star} = -\big(\sum_{i \in I_j} g_i\big)/\big(\sum_{i \in I_j} h_i + \lambda\big)$, et le gain d'une coupure séparant $I_L$ et $I_R$ vaut $\tfrac{1}{2}\big[G_L^2/(H_L+\lambda) + G_R^2/(H_R+\lambda) - (G_L+G_R)^2/(H_L+H_R+\lambda)\big] - \gamma$, avec $g_i, h_i$ les gradient et hessien de la perte. La contrainte de monotonie n'autorise que les coupures respectant le signe imposé.

**Conditions des classes.** Chaque classe finale respecte : un effectif minimal, un risque monotone, un écart significatif avec la classe voisine (test de l'annexe A), un sens métier cohérent et une stabilité entre échantillons.

**Package AGBoost.** Rôle du package interne dans la chaîne de discrétisation et **[Contribution]** implémentation de la méthodologie : réglage des hyperparamètres en amont, extraction des points de coupure, fusion et regroupement sous contraintes.

**Table C.1** — Tables de binning par variable : variable, classe, bornes, effectif, taux de défaut, poids de preuve `[DONNÉE]`.

## Annexe D — Matrices et clusters de corrélation

Cette annexe présente les corrélations qui fondent la réduction de 4.3.

**Corrélation de Spearman.** Matrice sur les variables continues et ordinales, et clusters formés au seuil de 80 %. La lecture identifie les groupes de variables porteuses de la même information monotone.

**Figure D.1** — Matrice de Spearman, carte de chaleur `[DONNÉE]`.
**Table D.1** — Composition des clusters de Spearman : cluster, variables, variable retenue, motif du choix (pré-filtre métier, importance SHAP) `[DONNÉE]`.

**$V$ de Cramér.** Matrice sur les variables discrétisées, et clusters formés au seuil de 0,4 fixé par la méthodologie interne.

**Figure D.2** — Matrice du $V$ de Cramér `[DONNÉE]`.
**Table D.2** — Composition des clusters de Cramér et variables retenues `[DONNÉE]`.

## Annexe E — Sélection finale

**Procédure.** Sélection de Scallan détaillée : score additif de poids de preuve, découpage en quartiles, calcul de l'Information Value puis de l'IV marginale, moyenne sur les trois sous-échantillons.

**Table E.1** — Historique des étapes : variable ajoutée, IV marginale moyenne, significativité `[DONNÉE]`.
**Figure E.1** — Chemin de régularisation du LASSO `[DONNÉE]`.

## Annexe F — Inférence

Maximum de vraisemblance : l'équation de score $\nabla \ell(\beta) = 0$ est résolue par Newton-Raphson sous forme de moindres carrés repondérés itérés, $\beta^{(k+1)} = \beta^{(k)} + \mathcal{I}(\beta^{(k)})^{-1} \nabla \ell(\beta^{(k)})$. La covariance des estimateurs est $\mathcal{I}(\hat\beta)^{-1}$. Test de Wald : $z = \hat\beta_m / \widehat{\operatorname{se}}(\hat\beta_m)$, avec $z^2 \sim \chi^2_1$. Test de déviance : $2\big[\ell(\text{complet}) - \ell(\text{réduit})\big] \sim \chi^2$.

## Annexe G — Grilles complètes des trois modèles

Les trois grilles finales, une par sous-périmètre. Chaque grille donne, pour chaque variable et chaque classe, l'effectif, le taux de défaut, le poids de preuve, le coefficient estimé et les points sur 1000. C'est le livrable central de la modélisation.

**Table G.1** — Grille du Pros-ER : variable, classe, effectif, taux de défaut, poids de preuve, coefficient, points sur 1000 `[DONNÉE]`.
**Table G.2** — Grille des Associations, mêmes colonnes `[DONNÉE]`.
**Table G.3** — Grille des SCI, mêmes colonnes `[DONNÉE]`.

## Annexe H — Variables du modèle et stabilité temporelle

Cette annexe présente les douze variables du modèle et documente la stabilité de chacune dans le temps. Elle sera renseignée avec les données de la banque ; l'esquisse ci-dessous fixe le format.

Pour chaque variable, une fiche réunit son intitulé et sa définition, son sens métier, son découpage en classes, et un graphe de stabilité temporelle. Le graphe suit, entre les deux dates de photographie espacées de six mois, la répartition des effectifs par classe et le taux de défaut par classe, complétés par le PSI de la variable entre les deux dates. Une variable est jugée stable si sa distribution et son taux de défaut par classe varient peu, et si son PSI reste faible. Une dérive signalerait une variable à surveiller ou à retirer.

**Figures H.1 à H.12** — Une fiche de stabilité par variable du modèle `[DONNÉE, à compléter]`.
**Table H.1** — PSI par variable entre les deux dates `[DONNÉE]`.

## Annexe I — Validation étendue

Courbes ROC par sous-périmètre `[DONNÉE]`. Tables de coefficients avec erreurs-types et intervalles de confiance `[DONNÉE]`. Indice de stabilité de population détaillé `[DONNÉE]`. Progression de performance des trois modèles, XGBoost complet, boosting de souches et régression logistique, par sous-périmètre, en apprentissage, test et *out-of-time* `[DONNÉE]`.
