# 4. Une grille logistique à discrétisation pilotée par un boosting de souches

Le règlement impose de progresser du modèle le plus simple au plus complexe. On part donc de la régression logistique brute, dont les limites motivent la discrétisation, puis on construit le modèle retenu : une grille logistique sur variables codées en *Weight of Evidence*, dont le découpage est appris par un boosting d'arbres de profondeur un. Les challengers non linéaires ne servent que de borne de performance et sont posés en fin de section.

On note l'échantillon $(x_i, y_i)_{i=1}^{n}$, où $y_i = 1$ si l'emprunteur $i$ est en défaut au sens réglementaire dans les douze mois (section 3) et $y_i = 0$ sinon. Le vecteur $x_i = (x_i^{(1)}, \dots, x_i^{(p)})$ rassemble les $p$ variables candidates. On modélise la probabilité de défaut $p_i = \mathbb{P}(Y_i = 1 \mid x_i)$.

---

## 4.1 La régression logistique brute et ses limites

La régression logistique relie le logarithme de la cote de défaut à une combinaison linéaire des variables :

$$\operatorname{logit}(p_i) = \ln\!\left(\frac{p_i}{1 - p_i}\right) = \beta_0 + \sum_{j=1}^{p} \beta_j\, x_i^{(j)}.$$

Les coefficients sont estimés par maximum de vraisemblance. Ce modèle est le format historique du scoring d'octroi pour sa lisibilité et sa traçabilité (Siddiqi, 2006).

Appliqué aux variables brutes, il impose une hypothèse forte : l'effet de chaque variable sur le log-odds est linéaire et monotone. Cette hypothèse résiste mal aux données de crédit. Les relations risque/variable présentent des effets de seuil, des paliers et parfois des non-monotonies. La forme brute est de plus sensible aux valeurs extrêmes et ne traite pas nativement les valeurs manquantes, nombreuses et parfois informatives sur notre périmètre (section 3).

Ces limites motivent une transformation préalable des variables par discrétisation. Découper une variable en classes remplace l'effet linéaire imposé par une fonction en escalier, que la régression logistique estime sans hypothèse de forme. Le découpage absorbe les valeurs extrêmes dans les classes de bord et accueille le manquant comme une modalité à part entière.

---

## 4.2 La grille logistique sur variables codées en Weight of Evidence

Chaque variable $x^{(j)}$ est partitionnée en $K_j$ classes, puis chaque classe est remplacée par son *Weight of Evidence*. On note $z_i^{(j)} = \operatorname{WoE}_j\big(x_i^{(j)}\big)$ la valeur codée de la variable $j$ pour l'individu $i$. Le modèle retenu s'écrit :

$$\operatorname{logit}(p_i) = \beta_0 + \sum_{j=1}^{p} \beta_j\, z_i^{(j)}.$$

Ce codage présente deux propriétés utiles. La transformation aligne chaque classe sur le log-odds cible, ce qui rend la relation linéaire par construction et légitime la forme logistique. Les coefficients $\beta_j$ se rapprochent de l'unité lorsque le codage capte bien le risque, ce qui facilite le diagnostic : un coefficient très éloigné de un signale une variable mal découpée ou instable.

> **Encadré 1 — Weight of Evidence et Information Value**
>
> Pour une variable $j$ découpée en classes indexées par $k$, on note $n^{+}_{j,k}$ le nombre de non-défauts et $n^{-}_{j,k}$ le nombre de défauts de la classe $k$, avec $N^{+}$ et $N^{-}$ les totaux sur l'échantillon. Le *Weight of Evidence* de la classe est :
>
> $$\operatorname{WoE}_{j,k} = \ln\!\left(\frac{n^{+}_{j,k} / N^{+}}{n^{-}_{j,k} / N^{-}}\right).$$
>
> Il mesure l'écart, en log-odds, entre la composition de la classe et celle de la population. Le pouvoir discriminant de la variable se résume par son *Information Value* :
>
> $$\operatorname{IV}_{j} = \sum_{k=1}^{K_j} \left(\frac{n^{+}_{j,k}}{N^{+}} - \frac{n^{-}_{j,k}}{N^{-}}\right) \operatorname{WoE}_{j,k}.$$
>
> Les seuils d'usage (variable faible en deçà de 0,02, suspecte au-delà de 0,5) sont des repères empiriques de la pratique scorecard (Siddiqi, 2006), non des règles. Détails et tables de découpage complètes en annexe.

Le modèle final se déploie sous forme de grille de points. Les coefficients logistiques sont convertis en points entiers par une transformation affine du log-odds qui préserve le classement, mise à l'échelle sur 1000 pour la lisibilité métier (Siddiqi, 2006). La règle de conversion figure en annexe.

---

## 4.3 Une discrétisation pilotée par un boosting de souches

Le découpage des variables ne repose pas sur une analyse univariée mais sur un boosting d'arbres, dont on exploite la structure quand la profondeur est fixée à un.

Un boosting de gradient construit une somme additive d'arbres en ajustant chaque nouvel arbre sur les résidus des précédents (Friedman, 2001). Avec des arbres de profondeur un, chaque arbre est une souche : il ne teste qu'une seule variable par une coupure unique. Comme aucun arbre ne combine deux variables, la somme des souches ne contient aucune interaction. En regroupant les souches par variable, la fonction apprise s'écrit :

$$F(x) = \beta_0 + \sum_{j=1}^{p} h_j\big(x^{(j)}\big),$$

où chaque $h_j$ est une fonction constante par morceaux de la seule variable $x^{(j)}$. C'est la forme d'un modèle additif, interprétable variable par variable, dans la lignée des modèles additifs boostés et des *Explainable Boosting Machines* (Nori et al., 2019).

Cette forme additive se lit comme un découpage supervisé. Les seuils de coupure retenus par les souches de la variable $j$ partitionnent son domaine, et la valeur de $h_j$ sur chaque tranche joue le rôle d'un poids de preuve appris directement sur la cible, dans l'estimation multivariée du boosting. La gestion native des manquants par XGBoost, qui apprend une direction par défaut à chaque coupure (Chen et Guestrin, 2016), route les valeurs manquantes vers la tranche la plus cohérente et rejoint le traitement décrit en section 3.

> **Encadré 2 — Gradient boosting et objectif de XGBoost**
>
> À l'étape $t$, l'arbre $f_t$ est choisi pour minimiser un objectif régularisé (Chen et Guestrin, 2016) :
>
> $$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l\!\left(y_i,\, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t), \qquad \Omega(f) = \gamma T + \tfrac{1}{2}\lambda \lVert w \rVert^2,$$
>
> où $T$ est le nombre de feuilles et $w$ leurs poids. Un développement au second ordre, avec $g_i$ et $h_i$ les gradient et hessien de la perte, donne le poids optimal d'une feuille $I_j$ et le gain d'une coupure séparant $I_L$ et $I_R$ :
>
> $$w_j^{\star} = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}, \qquad \text{Gain} = \tfrac{1}{2}\!\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma.$$
>
> Une contrainte de monotonie par variable force $h_j$ à croître ou décroître avec $x^{(j)}$. Hyperparamètres complets en annexe.

La monotonie est l'articulation clé avec le scorecard. En imposant à XGBoost une contrainte de monotonie par variable, chaque $h_j$ devient monotone, et le découpage induit hérite d'un risque monotone. On obtient des classes valides au sens du scorecard sans étape de correction *a posteriori*.

Les points de coupure sont ensuite fusionnés et regroupés pour produire le découpage final. Le regroupement impose un effectif minimal par classe, garantit la significativité des écarts de risque et respecte le sens métier attendu, avec une validation dédiée pour les modalités des variables qualitatives.

> **[Contribution]** Les hyperparamètres du boosting (taux d'apprentissage, nombre d'arbres, effectif minimal par feuille, régularisation, contraintes de monotonie) sont optimisés en amont du binning, la profondeur restant fixée à un. Le découpage n'est donc pas un sous-produit d'un modèle réglé par défaut, mais le résultat d'un boosting additif calibré pour la qualité des coupures.

Cette démarche prolonge un courant qui nourrit une régression logistique des découpages appris par des arbres. Dumitrescu et al. (2022) proposent une *penalised logistic tree regression* où des arbres courts encodent seuils et interactions avant injection dans un logit pénalisé. Leur profondeur deux vise justement les interactions. Le choix d'une profondeur un assume ici l'additivité stricte : on renonce aux interactions pour garantir un modèle strictement additif, plus simple à convertir en grille et à faire valider.

---

## 4.4 Réduction de la redondance entre variables

Le découpage produit un jeu de variables codées encore redondant. Deux variables très corrélées déstabilisent les coefficients logistiques et brouillent la lecture métier.

La redondance est mesurée par des clusters de corrélation. On utilise le coefficient de Spearman pour les variables continues et ordinales, sensible aux associations monotones, et le $V$ de Cramér pour les couples de variables qualitatives. Chaque cluster regroupe des variables porteuses de la même information.

> **[Contribution]** Les modalités des variables qualitatives sont encodées de façon ordinale selon leur sens du risque, c'est-à-dire l'ordre de leur *Weight of Evidence*. Cet encodage rend l'association monotone et homogénéise le traitement par Spearman, ce qui permet de mêler variables quantitatives et qualitatives dans une même analyse de corrélation.

Au sein de chaque cluster, une seule variable est conservée. L'élimination combine un critère d'instabilité *a priori* et le jugement métier, de préférence à une sélection purement automatique, pour retenir la variable la plus interprétable et la plus stable dans le temps.

---

## 4.5 Sélection parcimonieuse : stepwise puis LASSO

La parcimonie est recherchée pour la lisibilité, la surveillance et la stabilité du score. On passe d'environ trente variables candidates à dix ou douze.

> **[Contribution]** La sélection procède par ajout progressif *(forward stepwise)* sur les variables codées en *Weight of Evidence*. À chaque étape, la variable retenue est celle qui maximise le gain marginal d'*Information Value*, évalué en validation croisée pour limiter le sur-apprentissage. Une pénalisation LASSO sert ensuite de contrôle : elle confirme la parcimonie et la stabilité des coefficients du modèle sélectionné.

> **Encadré 3 — Pénalisation LASSO**
>
> Le LASSO ajoute à la log-vraisemblance logistique $\ell(\beta)$ une pénalité $L^1$ sur les coefficients :
>
> $$\hat{\beta} = \arg\min_{\beta} \; \left\{ -\ell(\beta) + \lambda \sum_{j=1}^{p} \lvert \beta_j \rvert \right\}.$$
>
> La pénalité annule les coefficients les moins informatifs, ce qui réalise conjointement l'estimation et la sélection (Tibshirani, 1996). Le paramètre $\lambda$ arbitre entre ajustement et parcimonie ; il est réglé par validation croisée.

L'optimisme de la validation croisée du stepwise est connu et sera confronté à une estimation propre en *out-of-time* (section 5.5).

---

## 4.6 Les challengers non linéaires comme borne de performance

Le modèle retenu est confronté à des challengers non linéaires, dont le rôle n'est pas d'être déployés mais de fixer la performance atteignable. XGBoost tient ce rôle de borne haute, désigné comme le modèle le plus performant du scoring dans la littérature récente (Gunnarsson et al., 2021).

Deux configurations sont opposées. Le boosting de profondeur un, additif, mesure ce qu'un modèle sans interaction peut atteindre. Un boosting de profondeur supérieure, qui autorise les interactions, quantifie par différence leur apport sur ce périmètre. L'écart entre les deux configurations est l'indicateur direct de la valeur des interactions, exploité en section 5. Les familles de modèles et leurs hyperparamètres complets figurent en annexe.

---

## 4.7 Validation et lecture métier du modèle final

La validation est annoncée ici et détaillée en section 5. La discrimination est mesurée par le Gini et l'AUC ainsi que par la statistique de Kolmogorov-Smirnov. La stabilité dans le temps est suivie par l'indice de stabilité de population (PSI) sur l'échantillon *out-of-time*. La calibration reste secondaire : le métier fixe le seuil de décision à partir d'un taux de défaut cible chez les acceptés, si bien que le classement et le taux de défaut observé par tranche suffisent à l'usage.

La lecture métier ferme la section. Le signe et l'ordre de grandeur de chaque coefficient sont confrontés au sens du risque attendu, et la monotonie imposée au découpage garantit une grille cohérente, défendable devant un comité de crédit. Les tables de coefficients et les courbes complètes sont reportées en annexe.
