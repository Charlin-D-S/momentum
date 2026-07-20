# Section 5 — Discussion : plan détaillé

**Titre** : « Performance, parcimonie et représentativité : ce que vaut la refonte »
Budget corps ~15 pages, cœur du mémoire. `[DONNÉE]` = à renseigner avec les chiffres de la banque. `[à confirmer]` = hypothèse à valider.

Ordre retenu : performance et plafond, qualité des grilles, comparaison à l'existant, parcimonie des modèles (nombre de grilles puis libellés), représentativité, robustesse, portée. Le test A/B des libellés (5.5) est le point culminant.

---

## Introduction de section (~0,3 p.)

- Annonce des questions : les grilles sont-elles performantes et proches du plafond, homogènes et calibrées, meilleures que l'existant, faut-il trois grilles, un modèle général suffit-il sur le Pros-ER, le modèle est-il représentatif, quelle robustesse.
- Fil rouge Pros-ER ; renvoi aux résultats de base établis en 4.6.

---

## 5.1 Performance des trois grilles et proximité de la borne (~2,5 p.)

**§1 — Tableau synthétique.** Les trois grilles, en apprentissage, test et *out-of-time* : Gini, KS, PSI. Commentaire sur le niveau atteint, la dégradation contrôlée en *out-of-time* et la stabilité. Table 5.1 `[DONNÉE]`.

**§2 — Progression des trois modèles.** XGBoost complet, boosting de souches de profondeur un, régression logistique, sur les trois échantillons. Lecture : le challenger fixe le plafond, l'écart entre la version additive et la logistique est faible. Table 5.2 `[DONNÉE]`.

**§3 — Le prix de l'interprétabilité.** L'écart de Gini entre la grille et le challenger, interprété via le *flat maximum* (Hand et Henley, 1997) : sur données de crédit, les méthodes se tiennent dans un intervalle étroit, ce qui rend le coût de l'interprétabilité modéré `[DONNÉE]`.

**§4 — Valeur des interactions.** L'écart entre le boosting additif (profondeur un) et la profondeur supérieure mesure l'apport des interactions. Conclusion attendue : apport faible sur ce périmètre `[à confirmer]`.

**§5 — [Contribution] Coïncidence des *risk drivers*.** Comparaison de l'importance SHAP du challenger aux variables retenues dans la grille (Lundberg et Lee, 2017). Leur coïncidence indique que la sélection capte les vrais moteurs de risque. Figure 5.1 `[DONNÉE]`.

**Citations** : Hand et Henley (1997), Gunnarsson et al. (2021), Lundberg et Lee (2017).

---

## 5.2 Une performance homogène et bien calibrée (~2,5 p.)

**§1 — Homogénéité par sous-segments.** Pour chaque grille, Gini et KS par segment, afin de montrer une performance stable et l'absence de poche de sous-performance : moins de 5 transactions, ancienneté de la relation, clients à particularités risque, type d'activité. Table 5.3 `[DONNÉE]`.

**§2 — Calibration.** Probabilité de défaut prédite comparée au taux de défaut observé, par tranche de score et par grille. Une grille est bien calibrée si les deux coïncident. Figure 5.2, courbes de calibration `[DONNÉE]`. La calibration reste secondaire pour la décision, le seuil étant fixé par le métier, mais elle est vérifiée ; un écart systématique justifierait un recalage.

**§3 — Interprétation.** Une grille homogène et calibrée est utilisable sur l'ensemble de la population d'octroi, sans segment mal servi.

**Question ENSAI** : robustesse des conclusions.

---

## 5.3 La refonte bat-elle le score existant ? (~2 p.)

**§1 — Protocole.** Comparaison directe au score en place, même périmètre et mêmes données.

**§2 — Résultats.** Courbes ROC superposées (Figure 5.3), Gini et KS comparés, gains par tranche de score. Table 5.4 `[DONNÉE]`.

**§3 — Décomposition du gain.** Où la refonte améliore : quelles populations, quelles variables nouvelles, quel rôle du binning multivarié.

**§4 — Verdict.** Justification chiffrée de la refonte.

**Question ENSAI** : résultats pertinents.

---

## 5.4 Faut-il trois grilles ou une seule ? (~2 p.)

**§1 — Question.** La spécialisation par sous-périmètre est-elle nécessaire, alors que la grille standard a des performances proches de celles des autres sur leur propre périmètre.

**§2 — Test.** Appliquer la grille standard du Pros-ER aux Associations et aux SCI, et comparer à leurs grilles dédiées. Table 5.5 `[DONNÉE]`.

**§3 — Lecture.** Si l'écart de performance est faible, la parcimonie des modèles plaide pour une grille unique ; sinon, la spécialisation se justifie `[à confirmer]`.

**§4 — Arbitrage.** Performance contre coût de développer et de maintenir trois grilles.

**Question ENSAI** : pertinence et parcimonie.

---

## 5.5 Un modèle général suffit-il sur le Pros-ER ? Le test A/B des libellés (~3 p., point culminant)

**§1 — Enjeu.** Modèle général unique sur tout le Pros-ER contre deux modèles spécialisés, l'un sur les clients à au moins 5 transactions avec libellés, l'autre sur les moins de 5 sans libellés.

**§2 — Le piège méthodologique.** Un gain brut mêle l'effet des libellés et l'effet de la spécialisation de population. Il faut les séparer.

**§3 — [Contribution] Design à population constante.** À population d'au moins 5 transactions fixée, comparer le modèle général sans libellés au modèle spécialisé avec libellés, ce qui isole l'apport des libellés. Schéma du design, Figure 5.4.

**§4 — Résultats.** Gain attribuable aux libellés, situé par rapport à la borne du challenger. Table 5.6 `[DONNÉE]`.

**§5 — Arbitrage.** Gain de performance contre coût de maintenir deux modèles et un pipeline de libellés.

**§6 — Verdict.** La chute du mémoire, formulée en une phrase claire.

**Question ENSAI** : résultats pertinents, question centrale.

---

## 5.6 Le modèle est-il représentatif de la population d'octroi ? (~1,5 p.)

**§1 — Problème.** Le stock sert à modéliser, le flux à décider ; la grille apprise sur stock peut mal représenter le flux d'octroi.

**§2 — [Contribution] Test à deux échantillons.** Un classifieur (XGBoost) tente de distinguer le stock 2024 de la production 2024, à temps constant pour isoler l'effet population de l'effet temps. Une AUC proche de 0,5 signale deux populations indiscernables (Lopez-Paz et Oquab, 2017). Figure 5.5 `[DONNÉE]`.

**§3 — Variables responsables.** Identifiées par SHAP ; lecture métier de ce qui distingue stock et flux ; conséquence pour l'usage de la grille.

**Question ENSAI** : robustesse des conclusions.

---

## 5.7 Robustesse des choix méthodologiques (~2 p.)

Pour chaque choix, on présente l'alternative et son effet sur la grille et sa performance. Table 5.7 `[DONNÉE]`.

- Binning multivarié par boosting de souches contre binning univarié monotone : sensibilité à la corrélation entre variables.
- Estimation sur variables indicatrices contre encodage en poids de preuve puis régression, à la manière de Siddiqi (2006).
- Sélection par IV marginale contre forward sur le Gini et contre LASSO seul : convergence sur les douze variables.
- Hypothèse d'indépendance : effet de la déduplication des identifiants clients dans la base d'entraînement, un même individu apparaissant à deux dates espacées de six mois.
- Optimisme de la validation croisée de la sélection contre l'estimation propre en *out-of-time*.

Argument transversal : la coïncidence des variables importantes du challenger (SHAP) avec celles de la grille conforte la robustesse de la sélection.

**Question ENSAI** : robustesse des conclusions.

---

## 5.8 Atouts, limites et portée (~1,5 p.)

**§1 — Atouts.** Interprétabilité, proximité de la borne de performance, traçabilité, reproductibilité, méthodologie pilotée par un challenger.

**§2 — Limites.** Écart au cadre IRB homologué, calibration laissée secondaire, hypothèse d'indépendance des observations, périmètre couvert.

**§3 — Portée.** Usage opérationnel de la grille et amorce des préconisations, développées en conclusion.

**Question ENSAI** : atouts et limites.

---

# Annexes de la section 5

Numérotation continuant la section 4 (annexes A à I), pour une intégration directe dans les annexes du mémoire.

## Annexe J — Performance détaillée et homogénéité

Tables de Gini et KS par sous-segment, pour chaque grille : moins de 5 transactions, ancienneté de la relation, clients à particularités risque, type d'activité `[DONNÉE]`. Progression des trois modèles par sous-périmètre `[DONNÉE]`.

## Annexe K — Calibration

Courbes de calibration, probabilité prédite contre taux de défaut observé, par tranche de score et par grille `[DONNÉE]`. Table des écarts de calibration `[DONNÉE]`.

## Annexe L — Comparaison au score existant

Courbes ROC superposées par sous-périmètre `[DONNÉE]`. Table des gains par tranche de score `[DONNÉE]`. Détail de la décomposition du gain `[DONNÉE]`.

## Annexe M — Faut-il trois grilles

Performance croisée : la grille standard appliquée à chaque périmètre, comparée à la grille dédiée `[DONNÉE]`. Écarts de Gini et KS `[DONNÉE]`.

## Annexe N — Test A/B des libellés

Design détaillé du test à population constante `[DONNÉE]`. Tables de performance des options : général unique, général sans libellés, spécialisé avec libellés `[DONNÉE]`. Situation par rapport à la borne du challenger `[DONNÉE]`.

## Annexe O — Test de représentativité (C2ST)

AUC du classifieur stock contre flux, à temps constant `[DONNÉE]`. Importance SHAP des variables discriminantes et lecture métier `[DONNÉE]`.

## Annexe P — Robustesse

Tables de sensibilité par choix méthodologique : binning multivarié vs univarié, indicatrices vs poids de preuve, IV marginale vs forward Gini vs LASSO, déduplication des identifiants, validation croisée vs *out-of-time* `[DONNÉE]`.

---

## Éléments à renseigner avec les données

- Gini, KS, PSI des trois grilles en apprentissage, test et *out-of-time*.
- Progression des trois modèles (XGBoost complet, boosting de souches, logistique).
- Gini/KS par sous-segment et par grille.
- Courbes de calibration par grille.
- Comparaison au score existant : ROC, gains par tranche.
- Performance croisée des grilles (grille standard sur chaque périmètre).
- Résultats du test A/B des libellés à population constante.
- AUC et SHAP du C2ST stock vs flux.
- Tables de sensibilité de la robustesse.

## Points à confirmer

- Ordre des sous-sections, en particulier 5.4 (nombre de grilles) placé avant le test A/B des libellés.
- Calibration groupée avec l'homogénéité en 5.2, ou en sous-section distincte.
- Référence à ajouter à la bibliographie pour le C2ST : Lopez-Paz et Oquab (2017), *Revisiting Classifier Two-Sample Tests*.
