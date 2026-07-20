# 2. Environnement du stage

Le stage s'est déroulé chez BNP Paribas, au sein de la filière RISK de la Banque Commerciale en France. L'équipe d'accueil, Model Design (RISK BCEF Architecture), conçoit et maintient les modèles de risque de crédit utilisés sur le périmètre des particuliers et des professionnels : modèles d'octroi, de suivi et de provisionnement. Elle travaille en lien avec les métiers, qui exploitent les scores dans leurs décisions, et avec les fonctions de validation et de contrôle, qui en éprouvent la solidité.

Le maître de stage, `[à compléter : fonction]`, encadre `[à compléter : rôle dans l'équipe]` et a défini le périmètre de la mission. Celle-ci portait sur la refonte du score d'octroi des professionnels et entrepreneurs individuels : construire une nouvelle grille de notation, la comparer à l'existant, et éclairer plusieurs choix de conception. Le stage s'est inséré dans le cycle habituel de l'équipe, de la préparation des données jusqu'à la restitution des modèles aux parties prenantes.

---

# 3. Un score d'octroi sur défaut bâlois à douze mois : enjeux, données de stock et représentativité

## 3.1 Enjeux et difficultés de la refonte

Le score d'octroi en place sur le périmètre des professionnels doit être refondu. Quatre raisons le motivent : un gain de performance attendu, un périmètre à préciser, une méthode de construction à moderniser, et des données nouvelles à exploiter. La refonte vise une grille qui trie mieux les demandes selon leur risque, sans perdre en lisibilité.

La difficulté centrale tient à une double exigence. La grille doit être performante, pour distinguer nettement les futurs défaillants des bons payeurs. Elle doit aussi rester interprétable, parce qu'une décision d'octroi s'explique à un client, se défend devant un comité et se contrôle. Ces deux objectifs tirent en sens contraire : les modèles les plus performants sont souvent les plus opaques. La tradition du scoring bancaire privilégie pour cette raison des grilles simples (Siddiqi, 2006).

Le positionnement de la mission est opérationnel, hors du cadre réglementaire IRB homologué. La performance de la grille sera bornée par celle de modèles non linéaires de référence, qui fixent le plafond atteignable et servent à mesurer le prix de la simplicité. Cet écart, sur données de crédit, est en général modéré, un fait connu sous le nom de *flat maximum* (Hand et Henley, 1997), ce que confirment les comparaisons de méthodes (Lessmann et al., 2015).

## 3.2 La cible : le défaut réglementaire à douze mois

La variable à prédire est le défaut au sens réglementaire. L'article 178 du règlement CRR le définit comme la conjonction d'un arriéré de paiement de quatre-vingt-dix jours et d'une probable non-recouvrabilité (Union européenne, 2013). On observe sa matérialisation sur une fenêtre de douze mois, ce qui aligne la cible sur l'horizon usuel de l'octroi.

Cette cible et cet horizon commandent la structure des données. Observer un défaut à douze mois suppose un recul d'au moins un an, ce qui interdit de modéliser sur les données les plus récentes et oriente vers les millésimes antérieurs.

Sur le plan des méthodes, la littérature récente situe le débat. Les modèles d'ensemble à base d'arbres, en particulier le gradient boosting, dominent les comparaisons de performance en scoring de crédit (Gunnarsson et al., 2021), sans que l'écart avec une régression logistique bien construite soit toujours large (Lessmann et al., 2015). Ce constat justifie la démarche retenue : une grille logistique, mais guidée et bornée par un tel modèle.

## 3.3 Sources, structure et qualité des données

Les données proviennent des systèmes internes de la banque `[à compléter : sources précises]`, et leur usage respecte le cadre RGPD. Elles réunissent des variables `[DONNÉE : familles de variables — signalétiques, comptables, comportementales, transactionnelles]`, appariées à partir de plusieurs tables reliées par l'identifiant client. Le schéma relationnel de cet appariement figure en Figure 3.1.

La volumétrie s'élève à `[DONNÉE]` observations sur `[DONNÉE]` variables candidates. La qualité des données est globalement bonne, avec un taux de valeurs manquantes faible `[DONNÉE]`, dont le traitement est décrit en 3.6.

**Figure 3.1** — Schéma relationnel de la génération des données et de l'appariement des tables `[DONNÉE]`.

## 3.4 Un design sur le stock à deux dates, stock contre flux

Le modèle est construit sur le stock de clients, photographié à deux dates espacées de six mois. Un même individu peut donc apparaître deux fois, à six mois d'intervalle, ce qui fournit deux observations. Ce choix augmente la taille de l'échantillon, mais il introduit une corrélation entre les deux observations d'un même client, et les fenêtres de performance de douze mois se chevauchent partiellement. L'hypothèse d'indépendance des observations n'est donc qu'approchée ; sa portée est mesurée en robustesse (section 5.7).

La modélisation porte sur le stock 2024, la production 2025 servant d'échantillon de contrôle. Cet échantillon est *out-of-time*, puisque postérieur, et aussi *out-of-population*, puisqu'on passe du stock des clients établis au flux des nouvelles demandes. Modéliser sur 2024 malgré la préférence interne pour les données récentes se justifie par la nécessité d'un recul de douze mois pour observer le défaut. La représentativité du stock vis-à-vis du flux est éprouvée en section 5.6.

## 3.5 Trois sous-périmètres et le filtrage par les transactions

Le périmètre se divise en trois sous-populations, selon un découpage métier établi : le Pros-ER, cœur des professionnels et entrepreneurs individuels hors associations et SCI, qui représente environ 80 % de la production ; les Associations ; les SCI. Chacune donne lieu à une grille distincte, soit trois modèles. Le Pros-ER sert de fil rouge dans la suite.

Un enrichissement par les transactions et leurs libellés est étudié, mais sur le seul Pros-ER. Ce choix repose sur des tests montrant que les Associations et les SCI sont insensibles à ce filtrage `[DONNÉE]`, et non sur une simple commodité. Sur le Pros-ER, un seuil de cinq transactions mensuelles sépare deux sous-populations dotées de jeux de variables différents : au-delà, les libellés apportent une information de comportement ; en deçà, ils sont absents. Cette partition ouvre la question centrale du mémoire, celle de l'intérêt de spécialiser des modèles selon les libellés, traitée en section 5.5.

## 3.6 Valeurs manquantes et absence de reject inference

Les valeurs manquantes sont traitées comme une information, non comme un vide à combler. En règle générale, une modalité « manquant » est isolée, laissée au modèle qui lui affecte son propre poids. Pour les variables comptables, une valeur sentinelle hors du domaine observé (0, -1 ou 1 selon la variable) est utilisée pour forcer le regroupement par l'arbre de discrétisation.

> **[Contribution]** Ce traitement s'appuie sur l'hypothèse d'un manquant informatif : l'absence de valeur n'est pas aléatoire mais porte un sens de risque, ce qui justifie de la coder comme une classe hors domaine plutôt que de l'imputer.

La question du biais de sélection est écartée explicitement. Le défaut bâlois est observé sur l'ensemble du stock, qu'un client ait été accepté ou refusé par le passé, si bien qu'aucune population n'est censurée. Le recours à une inférence sur les refusés (*reject inference*) n'a donc pas lieu d'être, point que le jury attend de voir posé.

## 3.7 Statistiques descriptives

Quelques statistiques éclairent les enjeux et la structure des données, sans viser l'exhaustivité. La Table 3.1 donne, par sous-périmètre, la volumétrie et le taux de défaut à douze mois. La part des clients à au moins cinq transactions mensuelles dans le Pros-ER s'établit à `[DONNÉE]`, ce qui conditionne la portée du test des libellés. Les taux de valeurs manquantes par grande famille de variables figurent en Table 3.2.

**Table 3.1** — Volumétrie et taux de défaut à douze mois, par sous-périmètre `[DONNÉE]`.
**Table 3.2** — Taux de valeurs manquantes par famille de variables `[DONNÉE]`.
