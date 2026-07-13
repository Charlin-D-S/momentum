# Construction d'un score de crédit parcimonieux : du modèle linéaire de référence au logit enrichi par le gradient boosting

## Revue de littérature

---

## 1. Cadrage : que cherche-t-on à modéliser ?

Le scoring de crédit désigne l'ensemble des méthodes statistiques formelles qui classent les emprunteurs selon leur risque de défaut (Hand et Henley, 1997). La littérature séminale pose déjà le cadre qui structure encore aujourd'hui la pratique : une variable cible binaire *bon/mauvais*, un jeu de caractéristiques hétérogènes, et une double exigence de performance discriminante et d'acceptabilité opérationnelle. Cette dernière contrainte n'est pas secondaire. Comme le rappellent Thomas, Edelman et Crook (2002) puis Siddiqi (2006), le score n'est pas seulement un prédicteur : c'est un objet réglementaire, décisionnel et auditable, ce qui explique la persistance remarquable de modèles simples dans un domaine où les méthodes plus puissantes sont disponibles depuis longtemps.

La cible retenue ici traduit une notion de *fragilité financière*. La référence naturelle est la définition bâloise du défaut, qui combine un critère objectif de retard de paiement — classiquement 90 jours d'impayé — et un critère de probable non-recouvrement (Comité de Bâle, 2006). Cette définition admet une profondeur temporelle variable : on peut observer la matérialisation du défaut à 3, 6, 12 mois ou davantage. Le choix de l'horizon n'est pas neutre statistiquement — il conditionne le taux de défaut observé, la stabilité de la cible et la comparabilité avec les modèles réglementaires. Nous retenons un **horizon de 12 mois**, cohérent avec la PD *point-in-time* à un an mobilisée sous IFRS 9 pour le provisionnement en *stage 1* et avec la maille annuelle dominante dans la littérature de scoring d'octroi. Fixer la fenêtre de performance à douze mois définit mécaniquement la fenêtre d'observation des variables explicatives et l'échantillonnage *out-of-time* qui servira à la validation.

---

## 2. Préparation des données : traitement informatif du manquant, absorption des valeurs extrêmes

Cette étape est traitée rapidement dans la littérature appliquée dès lors que le taux de manquant est faible, ce qui est notre cas. Deux points méritent néanmoins d'être ancrés bibliographiquement, parce qu'ils engagent des hypothèses de modélisation.

Le premier concerne les valeurs manquantes. La pratique du scorecard recommande de ne pas imputer aveuglément mais de traiter le manquant comme une information (Siddiqi, 2006 ; Thomas et al., 2002). Deux stratégies coexistent. On peut isoler une **modalité *Missing*** propre, laissée au modèle qui lui affectera son propre poids — approche naturelle dès lors que les variables sont ensuite discrétisées, la catégorie *Missing* devenant une classe à part entière dotée de son *Weight of Evidence*. On peut alternativement **imputer par une valeur spécifique** volontairement placée hors du support observé, de manière à pousser l'individu manquant vers un pôle favorable ou défavorable selon la sémantique métier de la variable. Le choix entre les deux dépend du mécanisme de manquance supposé : la modalité *Missing* domine quand le fait d'être renseigné est lui-même porteur d'information (MNAR), l'imputation informative se justifie quand une valeur de repli métier fait sens.

Le second point concerne les valeurs extrêmes. Plutôt qu'un écrêtage *ad hoc*, la discrétisation par intervalles absorbe naturellement les *outliers* : une valeur aberrante est simplement rattachée à la classe extrême, sans distordre l'estimation. Le binning agit ainsi comme un mécanisme de robustesse, ce qui reporte la question du traitement des extrêmes sur l'étape de discrétisation elle-même.

---

## 3. Discrétisation et binning : le cœur méthodologique

C'est ici que la revue prend toute son ampleur, car la discrétisation est à la fois l'opération la plus déterminante pour la performance d'un scorecard linéaire et la plus riche en variantes méthodologiques.

### 3.1 Pourquoi discrétiser ?

La motivation est double. Statistiquement, la discrétisation transforme une relation potentiellement non linéaire entre une variable continue et le log-odds du défaut en une fonction en escalier, que la régression logistique — linéaire dans ses paramètres — peut capturer sans hypothèse de forme fonctionnelle (Thomas et al., 2002). Elle absorbe les valeurs extrêmes, tolère les manquants comme une classe, et stabilise l'estimation en réduisant la variance au prix d'un biais contrôlé. Opérationnellement, elle produit des classes lisibles, traduisibles en grille de points et communicables à un comité de crédit ou à un régulateur. Cette lisibilité est précisément l'argument qui maintient la discrétisation au centre de la pratique bancaire malgré la perte d'information qu'elle occasionne.

La littérature générale sur la discrétisation supervisée fournit le socle théorique : García et al. (2013) en proposent une taxonomie complète, distinguant méthodes non supervisées (largeur ou fréquence égales) et supervisées, ces dernières exploitant la cible pour définir les coupures.

### 3.2 Discrétisation univariée non supervisée

Les approches les plus simples découpent la variable indépendamment de la cible : intervalles de largeur égale ou de fréquence égale (quantiles). Elles sont rapides et servent souvent de *pré-binning* avant une étape d'optimisation. Leur limite est connue : ne mobilisant pas la cible, elles ne garantissent aucun pouvoir discriminant des classes obtenues et peuvent scinder des zones homogènes en risque ou, à l'inverse, regrouper des populations hétérogènes.

### 3.3 Discrétisation supervisée par arbre de décision

La discrétisation par arbre constitue le pont naturel vers les méthodes supervisées. Un arbre CART ou un critère fondé sur le gain d'information sélectionne récursivement les seuils de coupure qui maximisent la séparation entre défaillants et non défaillants (Fayyad et Irani, 1993, pour la formulation MDL fondatrice). L'attrait est que les coupures sont directement optimisées pour la discrimination, et que la profondeur de l'arbre offre un levier de contrôle de la granularité. C'est cette logique que l'on retrouve, généralisée, dans les bibliothèques modernes de binning qui proposent un *pré-binning* par CART avant fusion des feuilles.

### 3.4 Weight of Evidence et Information Value

Une fois les classes formées, la transformation en *Weight of Evidence* (WoE) est l'opération canonique du scorecard. Le WoE d'une classe est le logarithme du rapport entre la proportion de bons et la proportion de mauvais qu'elle contient ; il exprime chaque classe sur une échelle monotone directement alignée sur le log-odds cible de la régression logistique (Siddiqi, 2006 ; Thomas et al., 2002). Régresser sur les WoE plutôt que sur des indicatrices présente deux avantages : la relation devient linéaire par construction, et les coefficients estimés se rapprochent de l'unité lorsque le codage capte bien le risque, ce qui facilite le diagnostic.

L'*Information Value* (IV) agrège les WoE pondérés par l'écart de distribution entre bons et mauvais pour fournir une mesure synthétique du pouvoir prédictif d'une variable. Les seuils empiriques usuels — en deçà de 0,02 la variable est inutile, au-delà de 0,5 elle est suspecte de sur-ajustement ou de fuite — sont des heuristiques de terrain issues de la tradition scorecard (Siddiqi, 2006) plus que des résultats théoriques, et doivent être maniés comme tels.

### 3.5 Contraintes : monotonie et binning optimal

La contribution méthodologique la plus active concerne l'imposition de contraintes au binning. La contrainte de **monotonie** du WoE le long des classes répond à une exigence à la fois statistique (régularisation, robustesse hors échantillon) et métier (une relation risque/variable non monotone est difficile à défendre devant un régulateur ou un métier). Mironchyk et Tchistiakov (2017) formalisent un algorithme de binning optimal monotone pour le risque de crédit, en fusionnant les classes jusqu'à obtenir un WoE strictement monotone. Navas-Palencia (2020) généralise cette logique en posant le binning optimal comme un **programme d'optimisation mathématique** sous contraintes — monotonie ascendante, descendante, convexe, concave, en pic ou en vallée — maximisant l'IV ou la statistique de Kolmogorov-Smirnov sous contrôle du nombre de classes et de leur effectif minimal. Cette formulation, implémentée dans la bibliothèque *OptBinning*, unifie pré-binning par arbre et optimisation combinatoire, et constitue aujourd'hui une référence d'implémentation pour un binning à la fois performant et réglementairement défendable.

---

## 4. Sélection des variables : filtrage classique et apport d'un challenger

La sélection vise la parcimonie — un score reposant sur un nombre restreint de variables se documente, se surveille et se conteste plus facilement. La littérature scorecard s'appuie traditionnellement sur un filtrage par IV pour écarter les variables sans pouvoir discriminant, complété par une analyse de corrélation (et de VIF) pour éliminer les redondances susceptibles de déstabiliser les coefficients logistiques (Siddiqi, 2006).

L'apport plus récent consiste à orienter la sélection par un **modèle challenger plus performant**. Plutôt que de s'en remettre aux seules mesures univariées, on ajuste un modèle non linéaire — typiquement un gradient boosting — et l'on exploite ses attributions d'importance pour hiérarchiser les variables. Les valeurs de **SHAP** (Lundberg et Lee, 2017), fondées sur la théorie des jeux coopératifs et dotées de propriétés d'additivité et de cohérence, fournissent une importance globale robuste, calculable efficacement sur les arbres via l'algorithme *TreeSHAP* en temps polynomial (Lundberg et al., 2020). Retenir les variables à forte contribution SHAP présente un double intérêt : la sélection intègre les effets non linéaires et les interactions que l'IV univariée ignore, tout en restant interprétable. Bussmann et al. (2021) illustrent en gestion du risque de crédit comment SHAP articule performance du boosting et explication au niveau de la décision individuelle. La sélection finale combine ainsi trois filtres : contribution SHAP du challenger, contrôle des corrélations, et recherche de parcimonie.

---

## 5. Le modèle linéaire de référence : justification et construction

### 5.1 Pourquoi un modèle linéaire ?

Le choix de la régression logistique comme modèle de référence ne relève pas d'un conservatisme méthodologique mais d'un arbitrage documenté. Trois arguments convergent.

Le premier est la **performance relative**. Les grands travaux de *benchmarking* montrent que, sur données de crédit, l'écart entre la régression logistique et les méthodes plus complexes est réel mais modéré. Baesens et al. (2003), sur huit jeux de données réels, concluent que la plupart des classifieurs se situent dans un mouchoir de poche et que la régression logistique demeure très compétitive. Lessmann et al. (2015), actualisant ce benchmark sur quarante et un classifieurs, confirment la supériorité des ensembles — forêts aléatoires et boosting — mais soulignent que la logistique reste une référence robuste et difficile à distancer nettement. Autrement dit, le gain de performance des méthodes complexes existe mais doit être mis en balance avec leur coût en interprétabilité.

Le deuxième argument est l'**interprétabilité et l'acceptabilité réglementaire**. Bücker et al. (2022) rappellent que la préférence bancaire pour la régression logistique et les arbres tient d'abord à l'exigence de transparence et d'auditabilité imposée par les régulateurs, exigence que les modèles boîte noire satisfont mal sans outillage additionnel. Le cadre IFRS 9 pour le provisionnement, les attentes prudentielles sur les modèles internes de PD (Comité de Bâle, 2006 ; lignes directrices de l'ABE) et, plus largement, le débat sur le « droit à l'explication » ouvert par le RGPD (Goodman et Flaxman, 2017) placent l'explicabilité au rang de contrainte de conception et non d'option.

Le troisième argument est **opérationnel** : un modèle linéaire additif se traduit directement en grille de points, se surveille par des indicateurs simples et se recalibre aisément, ce qui en fait le format de déploiement dominant du scoring d'octroi (Siddiqi, 2006).

### 5.2 Construction : logit sur WoE ou classes, sélection stepwise et Lasso

Le modèle s'ajuste soit sur les variables transformées en **WoE**, soit directement sur les **classes** issues du binning encodées en indicatrices. La première option produit un modèle compact aux coefficients diagnostiquables ; la seconde offre davantage de flexibilité au prix d'un espace de paramètres plus large.

Deux stratégies de sélection encadrent la recherche de parcimonie. La sélection **stepwise**, ascendante ou descendante, ajoute ou retire les variables selon un critère qui peut être la performance prédictive globale ou la **valeur d'information marginale** — le gain d'IV apporté par une variable conditionnellement à celles déjà présentes, qui corrige la myopie de l'IV univariée. La régularisation **Lasso** (Tibshirani, 1996) offre une alternative fondée sur la pénalisation L1, qui réalise conjointement estimation et sélection en annulant les coefficients des variables faibles ; elle est particulièrement adaptée lorsque les prédicteurs sont nombreux et corrélés, et fournit un chemin de régularisation exploitable pour arbitrer explicitement entre parcimonie et ajustement.

### 5.3 Mise en points

L'ultime transformation convertit les coefficients logistiques en une grille de points entière. Le passage log-odds → points repose sur une mise à l'échelle linéaire (paramètres de *score de base* et de *point to double the odds*) qui préserve le classement tout en produisant une échelle lisible (Siddiqi, 2006). Nous adoptons une **mise en points sur 1000**, choix d'échelle qui facilite l'interprétation — chaque variable contribue un nombre de points immédiatement intelligible et le score total se lit sur un intervalle familier au métier.

---

## 6. Le challenger XGBoost : élément central du pipeline

Le gradient boosting occupe une place centrale, non comme modèle de déploiement mais comme **borne de performance** et comme instrument d'analyse. Formalisé par Friedman (2001) comme une descente de gradient fonctionnelle ajustant séquentiellement des apprenants faibles sur les résidus, il connaît sa mise en œuvre de référence avec XGBoost (Chen et Guestrin, 2016), dont la régularisation explicite, la gestion native des manquants et l'efficacité computationnelle expliquent l'adoption massive.

En scoring de crédit, la littérature récente converge sur sa supériorité empirique. Gunnarsson et al. (2021), comparant apprentissage profond et méthodes établies, concluent que **XGBoost est le meilleur modèle** de leur panel et que les réseaux profonds ne justifient pas leur surcoût. Le tableau mérite toutefois d'être nuancé : sur un large échantillon de PME italiennes, Zedda (2024) trouve des capacités de sélection *comparables* entre XGBoost et régression logistique, l'avantage du boosting restant faible et sensible au réglage du seuil de décision. L'ampleur du gain dépend donc du jeu de données et de l'usage visé. Ce constat structure notre usage : le XGBoost n'est pas un concurrent à départager du linéaire, mais un **révélateur** de ce que le score parcimonieux laisse sur la table — l'écart de performance mesure la non-linéarité et les interactions non captées, et ses attributions SHAP en localisent l'origine. Il devient ainsi le pivot analytique du pipeline, alimentant à la fois la sélection de variables (section 4) et l'enrichissement du modèle final (section 7).

---

## 7. Le logit final qui exploite le XGBoost

La dernière étape ferme la boucle : construire une version finale du modèle linéaire qui **récupère une partie de la performance du challenger tout en conservant la forme d'un scorecard**. C'est le champ le plus dynamique de la littérature récente, structuré autour de l'idée que l'on peut transférer la connaissance d'un modèle complexe vers un modèle interprétable.

La contribution la plus directement applicable est celle de Dumitrescu, Hué, Hurlin et Tokpavi (2022). Leur *Penalised Logistic Tree Regression* (PLTR) extrait de courts arbres de décision — typiquement de profondeur deux, ajustés sur des paires de variables — des règles binaires qui encodent seuils et **interactions**, puis injecte ces règles comme prédicteurs dans une régression logistique pénalisée (Lasso adaptatif). Le modèle obtenu reste linéaire et interprétable, mais incorpore les effets non linéaires et croisés que les arbres ont détectés ; les auteurs montrent qu'il comble une part substantielle de l'écart de performance avec les méthodes boîte noire. Cette architecture est l'incarnation exacte de la logique visée : le linéaire tire parti du non-linéaire sans en payer le coût d'opacité.

Plusieurs voies convergentes existent dans la littérature. La première est la **sélection et l'ingénierie de variables guidées par SHAP** : les interactions saillantes révélées par les valeurs de SHAP d'interaction (Lundberg et al., 2020) sont matérialisées en variables croisées, puis rebinnées et intégrées au logit — le boosting sert de détecteur d'interactions, le linéaire de support de déploiement. La deuxième est l'imposition de **contraintes de monotonie** cohérentes entre le challenger et le scorecard : XGBoost supporte nativement des contraintes de monotonie par variable, ce qui aligne le modèle complexe sur les relations métier attendues et rend le transfert vers la grille de points plus légitime (Bücker et al., 2022). La troisième relève de la **distillation** et des modèles additifs intelligibles : les *Explainable Boosting Machines* et modèles additifs généralisés avec interactions (Lou et al., 2013 ; Nori et al., 2019) apprennent une décomposition additive de forme fonctionnelle libre par variable, dont les formes estimées peuvent informer le rebinning du scorecard — le boosting devient le professeur, le scorecard l'élève contraint.

Au-delà des techniques, cette étape cristallise l'**arbitrage central de la littérature** entre pouvoir prédictif et interprétabilité (Bücker et al., 2022 ; Bussmann et al., 2021 ; Bracke et al., 2019). La position retenue ici n'est pas de choisir un pôle mais de l'instrumenter : le XGBoost quantifie la performance atteignable et en diagnostique les sources ; le modèle linéaire final en réintègre autant que possible sous une forme parcimonieuse, monotone et auditable. Le score livré reste un scorecard ; sa conception, elle, aura été entièrement pilotée par le challenger.

---

## 8. En aval : calibration, validation, gouvernance

Pour mémoire, le pipeline se referme sur trois étapes que la littérature traite abondamment et qui encadrent tout déploiement. La **calibration** ajuste le niveau des probabilités prédites au taux de défaut réel de la population cible, condition de l'usage réglementaire de la PD sous IFRS 9. La **validation** mobilise la discrimination (Gini, AUC, KS), la qualité de calibration (test de Hosmer-Lemeshow) et la stabilité dans le temps (*Population Stability Index*), avec un contrôle *out-of-time* impératif. La **gouvernance** enfin documente la grille, trace les choix de modélisation et organise la surveillance — dimension d'autant plus critique que le pipeline mobilise un challenger complexe dont l'usage doit lui-même être justifié (Bücker et al., 2022).

---

## 9. Synthèse

Le fil de cette revue relie trois moments. Un modèle linéaire de référence, dont la persistance s'explique par un arbitrage performance/interprétabilité documenté depuis Hand et Henley (1997) jusqu'à Lessmann et al. (2015) et Bücker et al. (2022), et dont la construction repose sur une discrétisation soignée — véritable cœur méthodologique du scorecard, des approches par arbre au binning optimal sous contraintes de Navas-Palencia (2020). Un challenger XGBoost (Chen et Guestrin, 2016 ; Gunnarsson et al., 2021), non pour remplacer le linéaire mais pour en mesurer les limites et en éclairer les angles morts via SHAP (Lundberg et al., 2020). Une version finale du modèle linéaire qui exploite ce challenger, dans la lignée du PLTR de Dumitrescu et al. (2022), pour reconstituer sous forme de score parcimonieux et auditable une part de la performance du non-linéaire. Le score reste simple ; l'intelligence du pipeline réside dans la manière dont le complexe informe le simple.

---

## Références

Zedda, S. (2024). Credit scoring: Does XGBoost outperform logistic regression? A test on Italian SMEs. *Research in International Business and Finance*, 70, 102397.

Baesens, B., Van Gestel, T., Viaene, S., Stepanova, M., Suykens, J., & Vanthienen, J. (2003). Benchmarking state-of-the-art classification algorithms for credit scoring. *Journal of the Operational Research Society*, 54(6), 627-635.

Bracke, P., Datta, A., Jung, C., & Sen, S. (2019). Machine learning explainability in finance: An application to default risk analysis. *Bank of England Staff Working Paper* No. 816.

Bücker, M., Szepannek, G., Gosiewska, A., & Biecek, P. (2022). Transparency, auditability, and explainability of machine learning models in credit scoring. *Journal of the Operational Research Society*, 73(1), 70-90.

Bussmann, N., Giudici, P., Marinelli, D., & Papenbrock, J. (2021). Explainable machine learning in credit risk management. *Computational Economics*, 57, 203-216.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

Comité de Bâle sur le contrôle bancaire (2006). *International Convergence of Capital Measurement and Capital Standards (Bâle II)*. Banque des règlements internationaux.

Dumitrescu, E., Hué, S., Hurlin, C., & Tokpavi, S. (2022). Machine learning for credit scoring: Improving logistic regression with non-linear decision-tree effects. *European Journal of Operational Research*, 297(3), 1178-1192.

Fayyad, U. M., & Irani, K. B. (1993). Multi-interval discretization of continuous-valued attributes for classification learning. *Proceedings of the 13th International Joint Conference on Artificial Intelligence*, 1022-1027.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189-1232.

García, S., Luengo, J., Sáez, J. A., López, V., & Herrera, F. (2013). A survey of discretization techniques: Taxonomy and empirical analysis in supervised learning. *IEEE Transactions on Knowledge and Data Engineering*, 25(4), 734-750.

Goodman, B., & Flaxman, S. (2017). European Union regulations on algorithmic decision-making and a "right to explanation". *AI Magazine*, 38(3), 50-57.

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
