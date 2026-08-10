# 5.4 Du modèle de stock au seuil d'octroi : justification de la règle interne et lecture prudente du taux de défaut

*Note de rédaction. Les `[à compléter]` signalent des valeurs internes à renseigner. Le score de différenciation du risque construit dans ce mémoire est désigné par son nom interne, la probabilité de défaut à l'octroi (PDO).*

---

Le score n'est pas une fin en soi. Il ordonne les demandeurs, et le métier en déduit un seuil qui sépare les dossiers acceptés des dossiers refusés. Cette sous-section justifie la règle interne de fixation du seuil, expose l'asymétrie de population qu'elle induit, et montre pourquoi le taux de défaut des dossiers accordés doit être lu avec prudence.

## 5.4.1 Fixer le seuil sur le flux d'octroi : la règle interne et sa justification

La règle interne fixe le seuil sur le flux d'octroi, avec un objectif de taux de défaut cible à douze mois chez les clients acceptés. Le score ordonne les demandeurs, un seuil sépare les profils acceptés des profils refusés, et ce seuil est choisi de sorte que le taux de défaut attendu parmi les acceptés atteigne la cible fixée par la politique de risque.

Ce choix de caler le seuil sur le flux, et non sur le stock de modélisation, se justifie par trois arguments.

Le premier est la cohérence entre l'objet de décision et l'objet de calibration. Le seuil sert à trancher des demandes de crédit. Le calibrer sur la population des demandeurs aligne la calibration sur l'usage. Un taux de défaut cible chez les acceptés n'a de sens que mesuré sur une population de type flux, puisque c'est sur ce flux que la décision s'applique.

Le deuxième est la conformité aux processus internes. Les procédures validées imposent un seuil calé sur un flux d'octroi d'une période proche de celle de la modélisation. Cette proximité garantit la comparabilité des populations et l'auditabilité de la décision.

Le troisième tient à la couverture temporelle et à la pertinence de la population. Le flux d'octroi couvre tous les mois de l'année, ce qui lisse les effets calendaires que deux dates de photographie du stock ne peuvent capter. Surtout, il concentre l'analyse sur les clients réellement susceptibles de demander un crédit. Les dossiers accordés en 2024 ne représentent qu'environ 8 % du stock de clients [valeur à confirmer]. Caler le seuil sur le stock entier reviendrait donc à le fixer sur une population très majoritairement non demandeuse, dont le comportement d'octroi n'est pas observé.

## 5.4.2 L'asymétrie qui en découle : modéliser sur le stock, décider sur le flux

Cette règle induit une asymétrie. Le modèle est estimé sur le stock, population représentative du risque global du portefeuille. Le seuil est calé sur le flux, population doublement filtrée. La PDO en place y écarte d'abord les profils les plus fragiles, puis les règles métier retirent les dossiers présentant des anomalies. Ces filtres évoluent au cours du temps, au rythme de la politique d'octroi du groupe et de son appétit au risque.

Cette asymétrie n'est pas un problème d'inférence des refusés. Le défaut bâlois est observé sur l'ensemble du stock, quel que soit le statut d'acceptation, comme rappelé en section 3.8. La cible n'est donc pas censurée. Le problème est d'une autre nature : la population d'apprentissage et la population de décision diffèrent, alors même que la cible est pleinement observée sur la première.

L'ampleur de cette différence ne fait pas l'objet d'un test formel de comparaison entre les deux populations, et ce pour une raison de construction. Le stock est photographié à deux dates espacées de six mois, tandis que le flux d'octroi arrive de façon continue, réparti sur tous les mois de l'année. Un test opposant directement les deux échantillons confondrait cette différence de plan d'échantillonnage avec un éventuel écart de composition du risque, et capterait pour l'essentiel de la saisonnalité et un artefact de conception. La représentativité est donc discutée par ses mécanismes, et non mesurée par un test. À titre indicatif seulement, le taux de défaut moyen s'établit à 2,7 % sur la base de modélisation contre 3,0 % sur le flux [valeurs à confirmer], écart modéré qui peut relever en partie d'effets calendaires.

## 5.4.3 Un taux de défaut des accordés difficile à interpréter : l'effet ambigu du nouveau crédit

Le taux de défaut observé sur les dossiers accordés, qui sert à fixer le seuil, est lui-même délicat à interpréter. Le modèle est appris sur le stock de clients existants. Or l'octroi d'un prêt introduit une rupture dans le comportement de l'emprunteur, dont l'effet sur le moment de survenue du défaut joue dans deux sens opposés.

D'un côté, le prêt injecte une trésorerie inattendue. Cet apport peut soulager temporairement la situation financière et repousser la survenue du défaut.

De l'autre, le prêt crée des mensualités. Cette charge nouvelle constitue une obligation de paiement supplémentaire, dont le non-respect peut au contraire précipiter le défaut.

Le signe net de ces deux effets n'est pas connu a priori. Dans les deux cas, le moment de survenue du défaut chez un client ayant reçu un prêt récent diffère de celui qu'aurait connu le même profil en l'absence de ce prêt. Le taux de défaut observé sur les accordés est donc un estimateur biaisé du risque propre du demandeur, d'un biais dont le sens lui-même est indéterminé.

Ce phénomène est de surcroît mal capté par la PDO, pour une raison de volumétrie. Au moment de la photographie du portefeuille, les clients ayant bénéficié d'un prêt au cours des douze derniers mois sont très minoritaires, de l'ordre de 8 % du stock [valeur à confirmer]. Le signal lié à l'impact d'un accord de crédit est ainsi noyé dans le bruit statistique, et quasi ignoré par le modèle. La conséquence pratique est que le seuil, calé sur un taux de défaut lui-même distordu, doit être lu avec prudence, en particulier au voisinage de la frontière de décision.

## 5.4.4 Perspective : un taux de défaut contrefactuel par appariement pronostique

Une piste méthodologique permettrait de fonder le seuil sur un taux de défaut affranchi de cette distorsion. Elle n'est pas mise en œuvre dans ce mémoire, et est présentée comme une extension.

L'idée est de reconstruire une population contrefactuelle. À chaque demandeur du flux serait apparié un ou plusieurs clients du stock partageant son niveau de risque prédit et ses caractéristiques observables, mais n'ayant pas connu de prêt récent. Ces clients servent de proxy du comportement qu'aurait eu le demandeur en l'absence de la rupture introduite par le crédit. Leur défaut, observé sans cette distorsion, fournit une estimation d'un taux de défaut de référence, sur lequel le seuil pourrait alors être calé.

La nature du score d'appariement doit être nommée avec précision, car elle détermine la validité de la démarche. L'appariement envisagé repose sur un score pronostique, c'est-à-dire un score qui résume l'association des covariables avec la réponse potentielle, ici le défaut (Hansen, 2008, *Biometrika*, 95(2), p. 481-488). Il se distingue du score de propension, qui résume l'association des covariables avec l'affectation au traitement, ici la réception d'un prêt (Rosenbaum et Rubin, 1983, *Biometrika*, 70(1), p. 41-55). La PDO étant un prédicteur du défaut, et non de la probabilité de recevoir un crédit, elle relève du score pronostique. La qualifier de score de propension serait une erreur de nature.

L'intérêt de cette construction est d'éliminer l'effet du crédit sur le moment du défaut, sans présumer de son sens. Puisque cet effet est ambigu, l'objectif n'est pas de corriger dans une direction connue, mais de rapporter le seuil à un taux de défaut qui ne soit ni repoussé ni rapproché par la présence d'un prêt récent. Le seuil serait ainsi calé sur une base mieux définie que le taux brut des accordés.

Plusieurs points resteraient à trancher pour une mise en œuvre. Le jeu de caractéristiques d'appariement devrait être arrêté, au-delà du seul risque prédit : segment, famille d'activité, ancienneté de la relation, intensité transactionnelle, particularités de risque. La PDO étant discrète et à support fini, plusieurs appariements par demandeur seraient possibles, avec des poids normalisés à l'unité. La qualité de l'appariement demanderait enfin une validation, par exemple par l'examen de l'équilibre des covariables entre demandeurs et clients appariés. Ces éléments constituent une extension naturelle du présent travail, reprise en perspective dans la conclusion.
