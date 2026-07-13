# Construction d'un score de crédit parcimonieux : combiner un modèle linéaire et le gradient boosting

## Revue de littérature

---

## 1. La cible et la tension centrale du scoring

Le scoring de crédit regroupe les méthodes statistiques qui classent les emprunteurs selon leur risque de défaut (Hand et Henley, 1997). Deux exigences le gouvernent depuis l'origine, et elles tirent dans des directions opposées. On veut un modèle qui discrimine bien les futurs défaillants. On veut aussi un modèle qu'un métier peut lire, qu'un comité de crédit peut contester et qu'un régulateur peut auditer. Toute la littérature appliquée se joue dans cet écart entre performance et interprétabilité, et ce mémoire y prend position : garder la forme d'un scorecard linéaire, mais laisser un modèle complexe en piloter la construction.

La variable cible traduit une notion de fragilité financière. La référence retenue est la définition bâloise du défaut, qui combine un retard de paiement de 90 jours et une probable non-recouvrabilité (Comité de Bâle, 2006). Cette définition admet plusieurs profondeurs temporelles : on peut observer le défaut à 3, 6 ou 12 mois. Nous fixons l'horizon à **douze mois**, ce qui aligne la cible sur la PD réglementaire à un an de l'approche IRB et détermine mécaniquement la fenêtre d'observation des variables ainsi que l'échantillon *out-of-time* de validation. Le score produit une probabilité de défaut $p_i = \mathbb{P}(Y_i = 1 \mid x_i)$ qui alimente in fine le calcul d'exigence en fonds propres.

---

## 2. Le modèle linéaire de référence

### 2.1 Pourquoi le linéaire reste la référence

Le choix de la régression logistique ne relève pas d'un attachement au passé. Il découle d'un arbitrage mesuré. Les grands travaux de comparaison montrent que l'écart de performance entre la logistique et les méthodes plus sophistiquées existe mais reste contenu sur données de crédit. Lessmann et al. (2015), en confrontant quarante et un classifieurs, confirment que les méthodes d'ensemble dominent, tout en reconnaissant que la régression logistique tient une place difficile à déloger nettement. Le gain des modèles complexes est réel ; il se paie en opacité.

Cette opacité a un coût réglementaire concret. Les régulateurs attendent des modèles internes qu'ils soient transparents et auditables, ce qui a longtemps écarté les approches boîte noire du terrain (Bücker et al., 2022). Un modèle additif offre un autre avantage, opérationnel celui-là : il se traduit directement en grille de points, se surveille avec des indicateurs simples et se recalibre sans peine (Siddiqi, 2006). Le format de déploiement dominant du scoring d'octroi reste donc le scorecard linéaire.

### 2.2 Forme du modèle

La régression logistique modélise le log-odds de défaut comme une combinaison linéaire des variables :

$$\ln\!\left(\frac{p_i}{1 - p_i}\right) = \beta_0 + \sum_{j=1}^{p} \beta_j\, x_{ij}.$$

En pratique, les variables ne sont pas injectées brutes mais discrétisées puis recodées, ce qui fait l'objet de la section suivante. Une fois les coefficients estimés, la conversion en points repose sur une transformation affine du log-odds qui préserve le classement :

$$\text{Score} = \text{Offset} + \text{Facteur} \times \ln(\text{odds}), \qquad \text{Facteur} = \frac{\text{PDO}}{\ln 2},$$

où PDO désigne le nombre de points doublant la cote (*points to double the odds*). Nous adoptons une **échelle sur 1000** : chaque variable contribue un nombre de points immédiatement lisible et le score total se lit sur un intervalle familier au métier (Thomas et al., 2002 ; Siddiqi, 2006).

---

## 3. La discrétisation et le codage WoE : le cœur méthodologique

C'est l'étape qui décide de la performance d'un scorecard linéaire, et celle où la littérature offre le plus de variantes.

### 3.1 Pourquoi discrétiser

La motivation est double. Sur le plan statistique, découper une variable continue en classes transforme une relation potentiellement non linéaire avec le log-odds en une fonction en escalier, que la régression logistique capte sans hypothèse de forme. La discrétisation absorbe au passage les valeurs extrêmes, qui se rangent simplement dans une classe de bord, et elle traite le manquant comme une classe à part entière. Sur le plan opérationnel, elle produit des tranches lisibles, traduisibles en points et défendables devant un métier. Cette lisibilité justifie qu'on la conserve malgré la perte d'information qu'elle entraîne.

### 3.2 Du binning univarié au binning supervisé

Les découpages les plus simples ignorent la cible : largeur égale ou fréquence égale (quantiles). Rapides, ils servent surtout de pré-découpage, mais ne garantissent aucun pouvoir discriminant des classes obtenues. Le binning supervisé corrige ce défaut en choisissant les seuils qui séparent le mieux bons et mauvais. Un arbre de décision fournit ce mécanisme naturellement : il sélectionne récursivement les coupures maximisant le gain sur la cible. Cette même logique, poussée jusqu'à une optimisation sous contraintes, structure les sections 4 et 6.

### 3.3 Weight of Evidence et Information Value

Une fois les classes formées, chaque classe $c$ reçoit un poids de preuve défini à partir de ses effectifs de bons $g_c$ et de mauvais $b_c$ (avec $G$ et $B$ les totaux) :

$$\mathrm{WoE}_c = \ln\!\left(\frac{g_c / G}{b_c / B}\right).$$

Le WoE place chaque classe sur une échelle monotone directement alignée sur le log-odds cible. Régresser sur les WoE plutôt que sur des indicatrices rend la relation linéaire par construction :

$$\ln\!\left(\frac{p_i}{1 - p_i}\right) = \beta_0 + \sum_{j=1}^{p} \beta_j\, \mathrm{WoE}_j(x_{ij}),$$

et les coefficients se rapprochent de l'unité quand le codage capte bien le risque, ce qui facilite le diagnostic. Le pouvoir prédictif d'une variable se résume par son *Information Value* :

$$\mathrm{IV} = \sum_{c} \left(\frac{g_c}{G} - \frac{b_c}{B}\right) \mathrm{WoE}_c.$$

Les seuils usuels (en dessous de 0{,}02 la variable est inutile, au-dessus de 0{,}5 elle est suspecte de fuite) sont des heuristiques de terrain issues de la tradition scorecard (Siddiqi, 2006), à manier comme telles.

### 3.4 La contrainte de monotonie

L'apport méthodologique le plus actif concerne l'imposition de contraintes au découpage. La monotonie du WoE le long des classes répond à une double exigence : statistique, parce qu'elle régularise et stabilise le modèle hors échantillon ; métier, parce qu'une relation risque/variable non monotone se défend mal. Mironchyk et Tchistiakov (2017) proposent un algorithme fusionnant les classes jusqu'à obtenir un WoE strictement monotone. Navas-Palencia (2020) généralise cette idée en formulant le binning comme un programme d'optimisation mathématique sous contraintes de forme (croissante, décroissante, convexe, concave), maximisant l'IV ou la statistique de Kolmogorov-Smirnov sous contrôle du nombre de classes et de l'effectif minimal. Cette formulation, disponible dans la bibliothèque *OptBinning*, donne un découpage à la fois performant et réglementairement présentable. La monotonie sera aussi le fil qui relie ce découpage au modèle boosté de la section 6.

---

## 4. XGBoost, un challenger très performant

Le gradient boosting occupe une place centrale, non comme modèle livré mais comme référence de performance et comme instrument d'analyse.

### 4.1 Ce que dit la littérature sur sa performance

Sur données de crédit, les résultats convergent. Gunnarsson et al. (2021), en confrontant apprentissage profond et méthodes établies, désignent **XGBoost comme le meilleur modèle** de leur panel et concluent que les réseaux profonds ne justifient pas leur surcoût. Le constat rejoint celui de Lessmann et al. (2015) sur la supériorité des ensembles d'arbres. Le message pratique est stable : quand seule compte la discrimination, le boosting fixe la borne haute que les modèles simples cherchent à approcher.

### 4.2 Mécanique du modèle

Le boosting construit une somme additive d'arbres en minimisant un objectif régularisé (Friedman, 2001 ; Chen et Guestrin, 2016) :

$$\mathcal{L} = \sum_{i} l(y_i, \hat{y}_i) + \sum_{k} \Omega(f_k), \qquad \Omega(f) = \gamma T + \tfrac{1}{2} \lambda \lVert w \rVert^2,$$

où $T$ est le nombre de feuilles d'un arbre et $w$ ses poids. Les arbres sont ajoutés séquentiellement, chacun corrigeant les erreurs du précédent :

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta\, f_t(x_i).$$

À chaque nœud, la coupure retenue maximise un gain fondé sur les gradients $G$ et hessiens $H$ des deux enfants :

$$\text{Gain} = \tfrac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma.$$

Deux propriétés comptent pour la suite. XGBoost gère nativement les valeurs manquantes en apprenant une direction par défaut à chaque nœud. Il accepte surtout des **contraintes de monotonie** par variable, qui forcent la réponse du modèle à croître ou décroître avec la variable concernée (Bücker et al., 2022). Cette dernière capacité est la clé de l'articulation avec le scorecard.

---

## 5. La sélection parcimonieuse guidée par le challenger

Un score reposant sur peu de variables se documente et se surveille plus facilement. La sélection classique filtre par IV pour écarter les variables sans pouvoir discriminant, puis contrôle les corrélations et le VIF pour retirer les redondances qui déstabilisent les coefficients (Siddiqi, 2006).

L'apport récent consiste à orienter cette sélection par le challenger. Plutôt que de s'appuyer sur des mesures univariées, on lit l'importance des variables dans le modèle boosté au moyen des valeurs de SHAP (Lundberg et Lee, 2017), fondées sur la théorie des jeux coopératifs et calculables efficacement sur les arbres par l'algorithme *TreeSHAP* (Lundberg et al., 2020). Retenir les variables à forte contribution SHAP intègre les effets non linéaires que l'IV univariée ignore, tout en restant lisible. La sélection finale croise trois filtres : contribution SHAP, contrôle des corrélations, recherche de parcimonie.

---

## 6. Combiner le linéaire et le boosting : XGBoost de profondeur 1

Le point d'aboutissement de la revue tient en une idée. Le XGBoost est puissant mais opaque. En ramenant la profondeur de ses arbres à **1**, on le rend additif, donc linéaire et interprétable, sans renoncer à sa manière de découper les variables.

### 6.1 Profondeur 1 égale modèle additif

Un arbre de profondeur 1 est une souche : il ne teste qu'une seule variable. Comme aucun arbre ne combine deux variables, la somme des souches ne contient aucune interaction. En regroupant les souches par variable, la prédiction du modèle s'écrit :

$$F(x) = \beta_0 + \sum_{j=1}^{p} h_j(x_j),$$

où chaque $h_j$ est une fonction constante par morceaux de la seule variable $x_j$. C'est exactement la forme d'un modèle additif généralisé (GAM), interprétable variable par variable. Cette équivalence entre boosting d'arbres peu profonds et modèle additif est le principe des modèles additifs intelligibles (Lou et al., 2013) et des *Explainable Boosting Machines* (Nori et al., 2019).

### 6.2 De la monotonie à des classes valides

Chaque $h_j$ définit un découpage : ses seuils de coupure partitionnent le domaine de $x_j$, et sa valeur sur chaque tranche joue le rôle d'un WoE appris directement sur la cible. En imposant une contrainte de monotonie à XGBoost, on force $h_j$ à être monotone. Le découpage induit hérite alors d'un risque monotone, ce qui produit des classes valides au sens du scorecard, sans étape de correction *a posteriori*. Le boosting de profondeur 1 devient un binning supervisé, multivarié dans son estimation mais additif dans sa lecture.

### 6.3 Deux façons d'exploiter le découpage

À partir de là, deux usages se présentent, du plus direct au plus classique.

Le premier garde le modèle additif tel quel. Chaque $h_j$ se lit comme une courbe de contribution, se convertit en points et donne un score interprétable qui récupère une large part de la performance du XGBoost complet.

Le second reconstruit un scorecard familier. On collecte les seuils de $h_j$, on **regroupe les découpages** en fusionnant les tranches voisines de contribution proche pour gagner en parcimonie, puis on calcule le WoE de chaque classe ainsi obtenue. Ces classes alimentent enfin une régression logistique, affinée par sélection *stepwise* (sur la performance ou sur l'IV marginale) et par pénalisation Lasso :

$$\hat{\beta} = \arg\min_{\beta} \; \sum_{i} l\!\left(y_i,\, \beta_0 + \sum_{j} \beta_j\, z_{ij}\right) + \lambda \sum_{j} \lvert \beta_j \rvert,$$

où les $z_{ij}$ sont les WoE des classes issues du boosting. Le Lasso réalise conjointement estimation et sélection en annulant les coefficients faibles (Tibshirani, 1996), et fournit un chemin de régularisation qui arbitre explicitement entre parcimonie et ajustement. Le modèle final reste un logit sur WoE, entièrement lisible, mais son découpage a été appris par un modèle bien plus performant que l'analyse univariée.

### 6.4 Positionnement dans la littérature

Cette démarche prolonge un courant clair. Dumitrescu et al. (2022) extraient d'arbres courts des règles binaires qu'ils injectent dans une régression logistique pénalisée ; leur profondeur 2 vise justement à capter des interactions. Le choix inverse d'une profondeur 1 assume l'additivité pure : on renonce aux interactions pour garantir un modèle strictement additif, plus simple à convertir en grille de points et à faire accepter. Les modèles additifs intelligibles (Lou et al., 2013 ; Nori et al., 2019) fournissent la justification théorique de ce compromis, où un modèle appris comme un boosting se lit comme une somme de contributions par variable.

---

## 7. En aval : calibration, validation, gouvernance

Le pipeline se referme sur trois étapes largement traitées par la littérature. La calibration ajuste le niveau des probabilités prédites au taux de défaut réel de la population, condition de leur usage pour la PD réglementaire. La validation mobilise la discrimination (Gini, AUC, KS), la qualité de calibration (test de Hosmer-Lemeshow) et la stabilité temporelle (*Population Stability Index*), avec un contrôle *out-of-time* impératif. La gouvernance documente la grille, trace les choix de modélisation et organise la surveillance, dimension d'autant plus critique que la conception mobilise un challenger complexe dont l'usage doit lui-même être justifié (Bücker et al., 2022).

---

## 8. Synthèse

Le fil relie trois moments. Un modèle linéaire de référence, dont la persistance s'explique par l'arbitrage performance/interprétabilité et par sa traduction immédiate en grille de points. Une discrétisation soignée, cœur du scorecard, des découpages univariés au binning optimal sous contrainte de monotonie. Un challenger XGBoost, très performant dans la littérature mais opaque, que l'on domestique en ramenant sa profondeur à 1 : le modèle devient additif, monotone et interprétable, ses découpages deviennent des classes valides, et l'on peut soit le lire directement comme un score, soit en tirer un logit régularisé par stepwise et Lasso. Le score livré reste simple. L'intelligence du pipeline tient à la manière dont le complexe informe le simple.

---

## Références

Bücker, M., Szepannek, G., Gosiewska, A., & Biecek, P. (2022). Transparency, auditability, and explainability of machine learning models in credit scoring. *Journal of the Operational Research Society*, 73(1), 70-90.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

Comité de Bâle sur le contrôle bancaire (2006). *International Convergence of Capital Measurement and Capital Standards (Bâle II)*. Banque des règlements internationaux.

Dumitrescu, E., Hué, S., Hurlin, C., & Tokpavi, S. (2022). Machine learning for credit scoring: Improving logistic regression with non-linear decision-tree effects. *European Journal of Operational Research*, 297(3), 1178-1192.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189-1232.

Gunnarsson, B. R., vanden Broucke, S., Baesens, B., Óskarsdóttir, M., & Lemahieu, W. (2021). Deep learning for credit scoring: Do or don't? *European Journal of Operational Research*, 295(1), 292-305.

Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: A review. *Journal of the Royal Statistical Society: Series A*, 160(3), 523-541.

Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research*, 247(1), 124-136.

Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2013). Accurate intelligible models with pairwise interactions. *Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 623-631.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56-67.

Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. *Working paper*.

Navas-Palencia, G. (2020). Optimal binning: Mathematical programming formulation. *arXiv:2001.08025*.

Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). InterpretML: A unified framework for machine learning interpretability. *arXiv:1909.09223*.

Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. John Wiley & Sons.

Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). *Credit Scoring and Its Applications*. SIAM.

Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.
