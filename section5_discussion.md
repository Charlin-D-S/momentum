# 5. Discussion

## Performance, parcimonie et représentativité : ce que vaut la refonte

La section précédente a établi les trois grilles et leur validation de base. Cette discussion en défend la valeur, puis l'éprouve. Elle s'organise en trois blocs, alignés sur les questions du gabarit. Le premier montre que les résultats sont pertinents : les grilles sont performantes, elles battent l'existant, et elles tranchent les questions de spécialisation. Le second éprouve la robustesse de ces conclusions. Le troisième dégage les atouts, les limites et la portée. Le Pros-ER sert de fil rouge ; les résultats complets des autres périmètres figurent en annexe. Les emplacements `[DONNÉE]` reçoivent les chiffres de la banque.

---

# 5.1 Les résultats sont-ils pertinents ?

Ce bloc défend les trois affirmations de la thèse : la grille est performante et proche de la borne, elle bat le score existant, et elle éclaire s'il faut spécialiser les modèles.

## 5.1.1 Performance des trois grilles et proximité de la borne

Les trois grilles atteignent un pouvoir discriminant `[DONNÉE]`. Sur le Pros-ER, le Gini s'établit à `[DONNÉE]` en apprentissage, `[DONNÉE]` en test et `[DONNÉE]` en *out-of-time*, pour un KS de `[DONNÉE]`. La dégradation entre l'apprentissage et l'*out-of-time* reste contenue, et l'indice de stabilité de population demeure sous `[DONNÉE]`, ce qui témoigne d'une grille stable dans le temps. Les Associations et les SCI suivent le même profil (Table 5.1).

Pour situer ce niveau, on suit la performance de trois modèles emboîtés en complexité, sur les trois échantillons : le XGBoost sur la totalité des variables, sa version simplifiée de profondeur un, puis la régression logistique finale (Table 5.2). La progression montre ce que coûte, en discrimination, chaque pas vers l'interprétabilité. Le challenger profond fixe le plafond avec un Gini de `[DONNÉE]`. Le boosting de souches, additif, en retient l'essentiel à `[DONNÉE]`. La grille logistique clôt la chaîne à `[DONNÉE]`.

L'écart entre la grille et le challenger mesure le prix de l'interprétabilité. Sa faiblesse rejoint un fait ancien du scoring de crédit : sur ce type de données, les méthodes se tiennent dans un intervalle étroit, le *flat maximum* décrit par Hand et Henley (1997). Renoncer à la boîte noire coûte donc peu, ici `[DONNÉE]` points de Gini, au regard du gain de transparence.

La comparaison entre le boosting additif et sa version de profondeur supérieure éclaire un autre point. L'écart entre les deux quantifie l'apport des interactions entre variables. Il ressort à `[DONNÉE]`, ce qui indique que les interactions apportent peu sur ce périmètre et qu'une grille additive n'abandonne presque rien en s'en passant.

> **[Contribution]** On confronte enfin l'importance SHAP du challenger aux variables retenues dans la grille (Lundberg et Lee, 2017). Les moteurs de risque que le modèle complexe juge déterminants coïncident avec les variables sélectionnées `[DONNÉE]`. Cette coïncidence conforte la sélection : la grille ne s'appuie pas sur des variables commodes, mais sur celles que le meilleur modèle disponible identifie comme les plus prédictives.

**Table 5.1** — Gini, KS et PSI des trois grilles, en apprentissage, test et *out-of-time* `[DONNÉE]`.
**Table 5.2** — Gini et KS des trois modèles, du XGBoost complet à la grille logistique `[DONNÉE]`.
**Figure 5.1** — Importance SHAP du challenger et variables retenues dans la grille `[DONNÉE]`.

## 5.1.2 La refonte bat-elle le score existant ?

La refonte ne se justifie que si elle améliore le score en place. La comparaison se fait à périmètre et données identiques, sur les mêmes emprunteurs.

Sur le Pros-ER, la nouvelle grille porte le Gini de `[DONNÉE]`, celui du score existant, à `[DONNÉE]`, soit un gain de `[DONNÉE]` points. Les courbes ROC superposées montrent une dominance `[DONNÉE]` de la nouvelle grille (Figure 5.2), et les gains se lisent tranche de score par tranche de score (Table 5.3).

Le gain ne se répartit pas uniformément. Il vient surtout de `[DONNÉE : populations concernées]`, de l'apport de variables nouvelles `[DONNÉE]`, et de l'effet du binning multivarié, qui capte des seuils que le découpage précédent ignorait. Cette décomposition importe autant que le chiffre global, car elle explique où et pourquoi la refonte améliore la décision d'octroi.

Le verdict est chiffré : la refonte apporte `[DONNÉE]` points de Gini, ce qui `[DONNÉE : justifie / ne justifie pas]` son adoption.

**Figure 5.2** — Courbes ROC de la nouvelle grille et du score existant, Pros-ER `[DONNÉE]`.
**Table 5.3** — Gains par tranche de score, nouvelle grille contre score existant `[DONNÉE]`.

## 5.1.3 Faut-il trois grilles ou une seule ?

Trois grilles ont été construites, une par sous-périmètre. Leur maintenance a un coût, et la grille standard affiche des performances proches de celles des autres sur leur propre terrain. La question de la parcimonie des modèles se pose donc : une grille unique suffirait-elle.

Pour y répondre, la grille du Pros-ER est appliquée aux Associations et aux SCI, et sa performance comparée à celle des grilles dédiées (Table 5.4). L'écart ressort à `[DONNÉE]` points de Gini sur les Associations et `[DONNÉE]` sur les SCI.

Si cet écart est faible, la parcimonie plaide pour une grille unique, plus simple à surveiller et à documenter. S'il est marqué, la spécialisation se justifie par le gain de performance. L'arbitrage se tranche `[à confirmer]` en faveur de `[DONNÉE : une / trois grilles]`, en pesant la performance contre le coût de développer et de maintenir trois modèles distincts.

**Table 5.4** — Performance de la grille standard sur chaque périmètre, comparée à la grille dédiée `[DONNÉE]`.

## 5.1.4 Un modèle général suffit-il sur le Pros-ER ? Le test A/B des libellés

C'est la question centrale du mémoire. Sur le Pros-ER, on peut soit garder un modèle général unique, soit spécialiser deux modèles : l'un sur les clients à au moins 5 transactions mensuelles, enrichi des libellés de transaction, l'autre sur les moins de 5 transactions, sans libellés. Les libellés portent une information de comportement, mais les exploiter suppose de maintenir deux modèles et un pipeline de traitement.

Comparer directement le modèle général au modèle spécialisé serait trompeur. Un gain brut mêlerait deux effets : celui des libellés eux-mêmes, et celui de la spécialisation sur une population plus homogène. Attribuer aux libellés un gain qui vient de la population conduirait à une conclusion fausse.

> **[Contribution]** Pour isoler l'apport des libellés, la comparaison est menée à population constante. Sur les seuls clients à au moins 5 transactions, on oppose le modèle général sans libellés au modèle spécialisé avec libellés (Figure 5.3). La population étant identique, la différence de performance ne peut venir que des libellés.

Le gain attribuable aux libellés ressort à `[DONNÉE]` points de Gini, à situer par rapport à la borne du challenger (Table 5.5). Ce gain `[DONNÉE : dépasse / ne dépasse pas]` le coût de maintenir deux modèles et un pipeline de libellés.

Le verdict tranche la question centrale : sur le Pros-ER, `[DONNÉE : un modèle général suffit / deux modèles spécialisés se justifient]`. Cette conclusion oriente directement la préconisation de déploiement.

**Figure 5.3** — Design du test à population constante, général sans libellés contre spécialisé avec libellés `[DONNÉE]`.
**Table 5.5** — Performance des options et gain attribuable aux libellés, situé sous la borne du challenger `[DONNÉE]`.

---

# 5.2 Les conclusions sont-elles robustes ?

Ce bloc éprouve la solidité des résultats du bloc précédent : les grilles sont-elles homogènes et calibrées, s'appliquent-elles au flux d'octroi, et les conclusions résistent-elles aux choix de modélisation.

## 5.2.1 Une performance homogène et bien calibrée

Une bonne performance moyenne peut masquer des poches de sous-performance. On vérifie donc que le pouvoir discriminant reste stable à l'intérieur de chaque grille. Pour chaque sous-périmètre, le Gini et le KS sont mesurés sur des segments métier : les clients à moins de 5 transactions, les tranches d'ancienneté de la relation, les clients porteurs de particularités risque, et les types d'activité (Table 5.6). Les écarts entre segments restent faibles `[DONNÉE]`, ce qui indique une grille homogène, sans population mal servie.

La calibration est ensuite examinée, même si elle n'est pas l'objectif premier. Le seuil de décision étant fixé par le métier à partir d'un taux de défaut cible, seul le classement importe pour l'octroi. On contrôle néanmoins que la probabilité prédite coïncide avec le taux de défaut observé, tranche de score par tranche de score (Figure 5.4). Les deux courbes se suivent `[DONNÉE]` ; un écart systématique, s'il apparaissait, appellerait un simple recalage sans remettre en cause la grille.

Une grille homogène et calibrée s'utilise sur toute la population d'octroi avec la même confiance.

**Table 5.6** — Gini et KS par segment métier, pour chaque grille `[DONNÉE]`.
**Figure 5.4** — Courbes de calibration : probabilité prédite contre taux de défaut observé, par grille `[DONNÉE]`.

## 5.2.2 Le modèle est-il représentatif de la population d'octroi ?

Le modèle est appris sur le stock, mais il sert à décider sur le flux d'octroi. Si le stock et le flux diffèrent trop, la grille apprise pourrait mal se transporter. Le risque est double, puisque la production sert à la fois d'échantillon *out-of-time* et *out-of-population*.

> **[Contribution]** Pour mesurer cet écart de population, on recourt à un test à deux échantillons par classifieur. Un XGBoost tente de distinguer le stock 2024 de la production 2024, à date constante afin d'isoler l'effet population de l'effet temps. Si les deux populations sont indiscernables, le classifieur ne fait pas mieux que le hasard et son AUC avoisine 0,5 (Lopez-Paz et Oquab, 2017). L'AUC obtenue ressort à `[DONNÉE]` (Figure 5.5).

Les variables qui permettent au classifieur de séparer stock et flux sont identifiées par SHAP `[DONNÉE]`. Leur lecture métier indique `[DONNÉE : nature de l'écart]`, et la conséquence pour l'usage de la grille est `[DONNÉE]`. Une AUC proche de 0,5 conforterait le transport de la grille du stock vers le flux ; une AUC élevée signalerait des variables à surveiller.

**Figure 5.5** — AUC du classifieur stock contre flux et importance SHAP des variables discriminantes `[DONNÉE]`.

## 5.2.3 Robustesse des choix méthodologiques

La solidité des conclusions se juge à leur sensibilité aux choix de modélisation. Pour chacun, on présente l'alternative et son effet sur la grille et sa performance (Table 5.7).

Le binning multivarié par boosting de souches est comparé à un binning univarié monotone. Le premier tient compte de la structure conjointe des variables ; l'écart de performance `[DONNÉE]` mesure ce que la prise en compte de la corrélation apporte.

L'estimation sur variables indicatrices est comparée à un encodage des classes en poids de preuve puis à une régression, à la manière de Siddiqi (2006). Les deux approches convergent `[DONNÉE]`, ce qui montre que le choix du codage n'affecte pas les conclusions.

La sélection par Information Value marginale est comparée au forward sur le Gini et au LASSO seul. Les trois méthodes retiennent des ensembles proches `[DONNÉE]`, la sélection par IV marginale se distinguant par sa rapidité et sa stabilité.

L'hypothèse d'indépendance des observations mérite une attention particulière, car un même individu est photographié à deux dates espacées de six mois et produit deux observations corrélées. On mesure l'effet de la déduplication des identifiants clients dans la base d'entraînement, en ne conservant qu'une observation par client `[DONNÉE]`. Un écart faible confirmerait que la corrélation intra-emprunteur ne biaise pas les conclusions.

L'optimisme de la validation croisée de la sélection est enfin confronté à l'estimation propre en *out-of-time* `[DONNÉE]`, qui donne la mesure honnête de la performance attendue en production.

Un argument transversal soutient l'ensemble : les variables importantes du challenger, au sens de SHAP, coïncident avec celles de la grille. Quand le modèle le plus performant et la grille interprétable s'accordent sur les moteurs de risque, la sélection est robuste.

**Table 5.7** — Sensibilité de la grille et de sa performance à chaque choix méthodologique `[DONNÉE]`.

---

# 5.3 Atouts, limites et portée

La refonte présente plusieurs atouts. La grille reste interprétable et traçable, condition d'un usage en octroi et d'un contrôle par le métier. Sa performance approche la borne des modèles non linéaires, ce qui montre qu'elle ne sacrifie presque rien à la transparence. La méthodologie, pilotée par un challenger et implémentée dans un package interne, est reproductible.

Les limites doivent être posées avec la même clarté. Le modèle se tient hors du cadre IRB homologué, dans un usage opérationnel. La calibration est laissée secondaire, ce qui suffit tant que le seuil vient du métier, mais limiterait un usage probabiliste direct. L'hypothèse d'indépendance des observations reste une approximation, dont le bloc précédent a mesuré la portée. Le périmètre couvert, enfin, borne la généralité des conclusions.

La portée de l'étude est opérationnelle. Les grilles sont prêtes à être déployées par sous-périmètre, et le verdict du test des libellés oriente le choix entre un modèle général et deux modèles spécialisés. Les préconisations concrètes sont développées en conclusion.

---

# Annexes de la section 5

Numérotation continuant la section 4 (annexes A à I).

## Annexe J — Performance détaillée et homogénéité

Gini et KS par sous-segment, pour chaque grille : moins de 5 transactions, ancienneté de la relation, clients à particularités risque, type d'activité `[DONNÉE]`. Progression des trois modèles par sous-périmètre `[DONNÉE]`.

## Annexe K — Calibration

Courbes de calibration, probabilité prédite contre taux de défaut observé, par tranche de score et par grille `[DONNÉE]`. Table des écarts de calibration `[DONNÉE]`.

## Annexe L — Comparaison au score existant

Courbes ROC superposées par sous-périmètre `[DONNÉE]`. Table des gains par tranche de score `[DONNÉE]`. Décomposition détaillée du gain `[DONNÉE]`.

## Annexe M — Faut-il trois grilles

Performance croisée : la grille standard appliquée à chaque périmètre, comparée à la grille dédiée `[DONNÉE]`. Écarts de Gini et KS `[DONNÉE]`.

## Annexe N — Test A/B des libellés

Design détaillé du test à population constante `[DONNÉE]`. Tables de performance des options : général unique, général sans libellés, spécialisé avec libellés `[DONNÉE]`. Situation par rapport à la borne du challenger `[DONNÉE]`.

## Annexe O — Test de représentativité (C2ST)

AUC du classifieur stock contre flux, à date constante `[DONNÉE]`. Importance SHAP des variables discriminantes et lecture métier `[DONNÉE]`.

## Annexe P — Robustesse

Tables de sensibilité par choix méthodologique : binning multivarié vs univarié, indicatrices vs poids de preuve, IV marginale vs forward Gini vs LASSO, déduplication des identifiants, validation croisée vs *out-of-time* `[DONNÉE]`.

---

## Référence à ajouter à la bibliographie

Lopez-Paz, D., & Oquab, M. (2017). Revisiting classifier two-sample tests. *International Conference on Learning Representations (ICLR)*.
