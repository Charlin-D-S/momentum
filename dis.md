# 5.4 Du modèle de stock au seuil d'octroi : justification de la règle interne et lecture prudente du taux de défaut

*Note. Les `[à compléter]` signalent des valeurs internes à renseigner. Le score construit dans ce mémoire est désigné par son nom interne, la probabilité de défaut à l'octroi (PDO).*

---

Le score ordonne les demandeurs ; le métier en déduit un seuil qui sépare les dossiers acceptés des refusés. Cette sous-section justifie la règle interne de fixation du seuil, expose l'asymétrie de population qu'elle induit, et montre pourquoi le taux de défaut des accordés doit être lu avec prudence.

## 5.4.1 La règle interne et sa justification

Le seuil est calé sur le flux d'octroi, avec un objectif de taux de défaut cible à douze mois chez les acceptés. Ce choix, plutôt qu'un calage sur le stock, repose sur trois arguments.

- **Cohérence décision/calibration.** Le seuil tranche des demandes de crédit. Le calibrer sur la population des demandeurs aligne la calibration sur l'usage ; un taux cible chez les acceptés n'a de sens que mesuré sur une population de type flux.
- **Conformité aux processus internes.** Les procédures validées imposent un seuil calé sur un flux d'octroi proche de la période de modélisation, gage de comparabilité et d'auditabilité.
- **Couverture calendaire et pertinence de population.** Le flux couvre tous les mois, ce qui lisse les effets calendaires. Il concentre l'analyse sur les clients susceptibles de demander un crédit : les dossiers accordés en 2024 ne représentent qu'environ 8 % du stock [à confirmer]. Caler le seuil sur le stock entier reviendrait à le fixer sur une population très majoritairement non demandeuse.

## 5.4.2 L'asymétrie qui en découle

Le modèle est estimé sur le stock, représentatif du risque global. Le seuil est calé sur le flux, population doublement filtrée : la PDO en place écarte les profils fragiles, puis les règles métier retirent les dossiers anormaux, filtres qui évoluent avec la politique d'octroi du groupe.

Cette asymétrie n'est pas un problème d'inférence des refusés : le défaut bâlois est observé sur tout le stock, la cible n'est pas censurée (section 3.8). Le problème est un décalage entre population d'apprentissage et population de décision.

Ce décalage n'est pas mesuré par un test formel. Le stock est photographié à deux dates espacées de six mois, le flux arrive en continu sur tous les mois. Un test opposant les deux échantillons confondrait cette différence de plan d'échantillonnage avec un écart de risque, et capterait surtout de la saisonnalité. À titre indicatif seulement, le taux de défaut moyen s'établit à 2,7 % sur la base de modélisation contre 3,0 % sur le flux [à confirmer], écart modéré possiblement calendaire.

## 5.4.3 Un taux de défaut des accordés difficile à interpréter

L'octroi d'un prêt introduit une rupture de comportement dont l'effet sur le moment du défaut joue dans deux sens opposés.

- La trésorerie injectée par le prêt peut repousser le défaut.
- Les mensualités créées par le prêt ajoutent une charge, dont le non-paiement peut le précipiter.

Le signe net est indéterminé a priori. Le taux de défaut observé sur les accordés est donc un estimateur biaisé du risque propre du demandeur, d'un biais de sens inconnu. Ce signal est de surcroît mal capté par la PDO : les clients ayant reçu un prêt sur douze mois sont très minoritaires, de l'ordre de 8 % du stock [à confirmer], donc noyés dans le bruit. Le seuil, calé sur un taux distordu, doit être lu avec prudence au voisinage de la frontière de décision.

## 5.4.4 Perspective : un taux de défaut contrefactuel par appariement pronostique

Une piste, non mise en œuvre ici, corrigerait cette distorsion. À chaque demandeur serait apparié un ou plusieurs clients du stock de même risque prédit et mêmes caractéristiques, n'ayant pas connu de prêt récent, dont le défaut observé fournit un taux de référence pour caler le seuil.

- **Vocabulaire.** L'appariement repose sur un score pronostique, qui résume l'association des covariables avec le défaut (Hansen, 2008, *Biometrika*, 95(2), p. 481-488), à distinguer du score de propension, qui prédit l'affectation au traitement, ici recevoir un prêt (Rosenbaum et Rubin, 1983, *Biometrika*, 70(1), p. 41-55). La PDO relève du pronostique.
- **Intérêt.** Éliminer l'effet du crédit sur le moment du défaut, sans présumer de son sens puisqu'il est ambigu.
- **Questions ouvertes.** Jeu de caractéristiques d'appariement ; appariements multiples à poids normalisés à 1 (PDO discrète à support fini) ; validation par l'équilibre des covariables. Extension reprise en conclusion.




# Ajout à 5.5 (perspective) et rédaction de 5.6

*Note. Les `[à compléter]` signalent des valeurs ou verdicts à renseigner une fois les chiffres produits.*

---

## 5.5.x Une variante du plan d'échantillonnage : étaler la photographie du stock sur douze mois (perspective)

Le plan d'échantillonnage retenu photographie le stock à deux dates fixes espacées de six mois (section 3.3). Une variante, non mise en œuvre dans ce mémoire, mérite d'être discutée car elle lèverait plusieurs limites identifiées plus haut.

Le principe est d'étaler les dates de photographie sur l'ensemble de l'année. Chaque client resterait observé au plus deux fois, à six mois d'intervalle, mais le couple de dates serait réparti aléatoirement sur les douze mois : un client observé en juin le serait de nouveau en décembre, un autre en février puis en août, et ainsi de suite. Collectivement, les dates d'observation couvriraient alors tous les mois.

Cette variante présenterait plusieurs avantages.

- **Neutralisation des effets calendaires.** En répartissant les observations sur toute l'année, le plan cesse de dépendre des conditions particulières de deux mois précis. Le signal appris ne reflète plus une conjoncture saisonnière isolée.
- **Comparabilité avec le flux d'octroi.** Le flux arrive de façon continue sur tous les mois. Un stock étalé sur douze mois épouse cette structure temporelle. L'obstacle soulevé en 5.4.2, qui interdit de comparer directement un stock à deux dates et un flux continu, disparaîtrait : les deux populations deviendraient comparables sur le plan du calendrier.
- **Facilitation de l'appariement contrefactuel.** La construction de clones décrite en 5.4.4 suppose d'apparier chaque demandeur à des clients du stock comparables. Disposer de clients du stock observés au même mois que chaque demandeur rendrait cet appariement plus naturel et mieux contrôlé sur le temps.

Cette variante ne résout pas tout, et ses limites doivent être énoncées.

- Chaque client reste observé deux fois à six mois d'intervalle. Les deux fenêtres de performance de douze mois continuent de se chevaucher, donc la corrélation intra-emprunteur et la structure de panel (section 3.3) subsistent. La variante traite le calendrier, pas la dépendance temporelle.
- Elle n'ajoute aucune information par client : elle redistribue les dates, sans augmenter le nombre d'observations individuelles.
- L'affectation d'un client à un couple de mois devrait être aléatoire et indépendante du risque, sous peine d'introduire un biais. Sa mise en œuvre suppose enfin une disponibilité homogène des données sur tous les mois.

En résumé, cette variante est une piste d'amélioration du plan d'échantillonnage, dont l'intérêt principal est de rendre le stock comparable au flux et d'outiller la fixation du seuil. Elle est reprise en perspective dans la conclusion.

---

## 5.6 Portée des résultats : pertinence, atouts et limites, robustesse

Cette dernière sous-section prend du recul sur l'ensemble de l'étude. Elle répond à trois questions : les résultats sont-ils pertinents, quels sont les atouts et les limites du travail, et quelle est la robustesse des conclusions.

### 5.6.1 Les résultats sont-ils pertinents ?

La pertinence se juge à l'aune de la question de départ : refondre un score d'octroi à la fois plus performant que l'existant et interprétable.

- **Gain sur l'existant.** La grille proposée améliore la différenciation du risque par rapport au score en place [gain de Gini à compléter], sur le même périmètre. La refonte est donc justifiée par un gain mesurable, et non par le seul renouvellement méthodologique.
- **Proximité de la borne non linéaire.** L'écart de performance entre la grille et les challengers non linéaires est [faible / modéré, à compléter]. Cet écart chiffre le prix de l'interprétabilité, et confirme empiriquement que les interactions apportent peu sur ce périmètre. Le choix d'une grille additive n'est donc pas un renoncement coûteux.
- **Réponse à la question des libellés.** Sur le Pros-ER, le test A/B tranche entre un modèle général et deux modèles spécialisés [verdict à compléter]. Ce verdict est directement exploitable par le métier, puisqu'il arbitre entre performance et coût de maintenance.

Ces résultats sont pertinents parce qu'ils débouchent sur une décision : quel modèle déployer, et à quel prix. Ils ne relèvent pas de la performance pour elle-même.

### 5.6.2 Atouts et limites de l'étude

**Atouts.**

- **Interprétabilité sans sacrifice majeur de performance.** La grille reste lisible variable par variable, tout en se situant près de la borne atteignable, ce qui satisfait la double exigence de départ.
- **Traçabilité de la construction.** Chaque étape (binning, réduction de redondance, sélection) est documentée et reproductible, condition d'auditabilité d'un score d'octroi.
- **Contributions méthodologiques.** L'optimisation des hyperparamètres en amont du binning, les clusters de corrélation et la sélection stepwise contrôlée par LASSO renforcent une méthodologie interne, au lieu de la subir.
- **Cadrage honnête de l'usage.** L'asymétrie stock/flux et l'effet ambigu du crédit sur le défaut sont explicités, ce qui borne correctement la portée du score.

**Limites.**

- **Structure de panel.** Le design à deux dates induit une corrélation intra-emprunteur et des fenêtres de performance chevauchantes, qui rendent optimistes les tests de significativité utilisés lors du binning et de la sélection (section 5.5).
- **Représentativité stock/flux.** Le modèle est appris sur le stock et appliqué au flux ; ce décalage n'est pas mesuré par un test formel, pour les raisons de plan d'échantillonnage exposées en 5.4.
- **Effet du crédit sur le défaut.** Le taux de défaut des accordés, qui sert à fixer le seuil, est biaisé d'un sens indéterminé, et ce biais n'est pas corrigé dans ce mémoire.
- **Écart au cadre IRB homologué.** Le score reprend la définition réglementaire du défaut mais n'est pas développé pour le calcul d'exigence en fonds propres ; il ne comporte ni calibration through-the-cycle ni marge de conservatisme réglementaire.
- **Calibration volontairement secondaire.** Le métier fixant le seuil par un taux de défaut cible chez les acceptés, seuls le classement et le taux observé par tranche sont nécessaires. Une PD globalement calibrée n'est pas produite, ce qui restreint les usages hors octroi.

### 5.6.3 Robustesse des conclusions

La robustesse a été éprouvée en confrontant chaque choix méthodologique à son alternative (section 5.5). Les conclusions principales tiennent selon les axes suivants.

- **Stabilité hors échantillon.** La performance se maintient en out-of-time sur la production 2025 [dégradation à compléter], et les indices de stabilité (PSI) restent [à compléter]. La grille ne surajuste pas la période d'apprentissage.
- **Convergence des méthodes de sélection.** Le stepwise et le LASSO retiennent des ensembles de variables voisins [à compléter], ce qui désamorce la critique classique du stepwise et conforte la parcimonie retenue.
- **Concordance avec le challenger.** Les variables importantes du modèle non linéaire (SHAP) recoupent celles de la grille [à compléter], signe que la sélection ne néglige pas de signal exploité par un modèle plus riche.
- **Sensibilité aux choix de traitement.** Le traitement des manquants (sentinelles vs modalité dédiée) et le mode de binning (boosté vs univarié monotone) modifient [marginalement / sensiblement, à compléter] la performance, ce qui situe le degré de dépendance des résultats à ces choix.

Ces vérifications ne prétendent pas à l'exhaustivité. Elles montrent que les conclusions centrales, à savoir le gain sur l'existant, la proximité de la borne non linéaire et le verdict du test A/B, ne reposent pas sur un réglage particulier, mais résistent au changement d'alternative méthodologique. Les limites subsistantes, structure de panel et représentativité stock/flux, sont identifiées et ouvrent les perspectives reprises en conclusion.
