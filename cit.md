# Références bibliographiques : où et comment les citer dans le mémoire

Document de travail. Rappel du cadre ENSAI : pas de chapitre « revue de littérature », bibliographie limitée à une page, citations tissées dans les sections 3, 4 et 5 au moment où elles justifient un choix. Système auteur-date (APA), version publiée référencée, DOI dans la liste finale.

---

## Cartographie des références sur le plan

| Référence | Où la placer | Rôle |
|---|---|---|
| CRR art. 178 (Règl. UE 575/2013) | 3 — Cible | Définition du défaut (90 j + *unlikeliness to pay*). À citer à la place de Bâle II |
| Hand & Henley (1997) | 3 et 5.1bis | Le *flat maximum*, borne de performance |
| Siddiqi (2006) | 3 et 4 | Scorecard, WoE/IV, mise en points (citer les chapitres précis) |
| Lessmann et al. (2015) | 3 | Benchmark, la logistique reste compétitive |
| Gunnarsson et al. (2021) | 3, 4, 5.1bis | XGBoost meilleur modèle, borne non linéaire |
| Chen & Guestrin (2016) | 4 | XGBoost |
| Friedman (2001) | 4 | Gradient boosting (encadré ou annexe) |
| Nori et al. (2019) | 4 | Lignée GAM/EBM, profondeur 1 = additif |
| Dumitrescu et al. (2022) | 4 et 5 | Effets d'arbres dans un logit pénalisé, positionnement |
| Navas-Palencia (2020) | 4 et 5.5 | Binning optimal monotone, alternative univariée en robustesse |
| Lundberg & Lee (2017) | 5.1bis, 5.4, 5.5 | SHAP (C2ST, *risk drivers* du challenger) |
| Tibshirani (1996) | 4 | LASSO de contrôle |

Écartées pour tenir dans la page : Thomas et al. (redondant avec Siddiqi), Lou et al. (couvert par Nori), Mironchyk & Tchistiakov (couvert par Navas-Palencia), Lundberg et al. 2020 (garder le SHAP de 2017). Bücker et al. (2022) optionnel, pour appuyer la double exigence performance/interprétabilité.

---

## Phrases prêtes à adapter, par partie

Gabarits à ajuster aux chiffres et au périmètre réels.

### Section 3 — Problématique et données

La cible est le défaut au sens réglementaire, défini à l'article 178 du règlement CRR comme la conjonction d'un arriéré de quatre-vingt-dix jours et d'une probable non-recouvrabilité (Union européenne, 2013), observé sur une fenêtre de douze mois.

Sur données de crédit, les performances des méthodes de classification se tiennent dans un intervalle étroit, un phénomène connu sous le nom de flat maximum (Hand et Henley, 1997) ; c'est lui qui rend une grille logistique compétitive face à des modèles plus complexes.

Le benchmark de référence sur quarante et un classifieurs confirme l'avantage des méthodes d'ensemble, tout en reconnaissant que la régression logistique reste un point de comparaison difficile à distancer (Lessmann et al., 2015).

Dans une comparaison récente des méthodes d'apprentissage pour le scoring, XGBoost ressort comme le modèle le plus performant, sans que les réseaux profonds justifient leur surcoût (Gunnarsson et al., 2021).

La pratique du scorecard repose sur la discrétisation des variables et leur codage en Weight of Evidence, format lisible et directement traduisible en grille de points (Siddiqi, 2006).

*(Optionnel)* La refonte doit satisfaire une double exigence de performance et d'interprétabilité, cette dernière restant une contrainte de conception pour un modèle d'octroi (Bücker et al., 2022).

### Section 4 — Inférence statistique

La régression logistique brute suppose une relation linéaire entre les variables et le log-odds de défaut, hypothèse mise en défaut par les effets de seuil observés, ce qui motive une discrétisation préalable.

Chaque variable est recodée en Weight of Evidence, dont la valeur s'aligne sur le log-odds cible, et son pouvoir discriminant se résume par l'Information Value (Siddiqi, 2006).

Le boosting de gradient construit une somme additive d'arbres en ajustant chaque apprenant sur les résidus du précédent (Friedman, 2001), mis en œuvre ici avec XGBoost pour sa régularisation et sa gestion native des valeurs manquantes (Chen et Guestrin, 2016).

En restreignant la profondeur des arbres à un, le modèle ne contient plus aucune interaction et s'écrit comme une somme de contributions par variable, la forme d'un modèle additif interprétable dans la lignée des Explainable Boosting Machines (Nori et al., 2019).

Les points de coupure issus des souches sont regroupés sous contrainte de monotonie, dans l'esprit du binning optimal formulé comme un programme d'optimisation (Navas-Palencia, 2020).

Nourrir une régression logistique des découpages appris par des arbres a déjà été proposé sous le nom de penalised logistic tree regression, où des arbres courts encodent seuils et interactions avant injection dans un logit pénalisé (Dumitrescu et al., 2022) ; le choix d'une profondeur un assume ici l'additivité stricte et renonce volontairement aux interactions.

La sélection est resserrée par une pénalisation LASSO, qui réalise conjointement l'estimation et le choix des variables en annulant les coefficients les moins informatifs (Tibshirani, 1996).

### Section 5 — Discussion

Le challenger XGBoost, identifié comme meilleur modèle dans la littérature récente (Gunnarsson et al., 2021), fixe la borne haute contre laquelle la grille est évaluée.

L'écart de Gini entre la grille et les challengers non linéaires mesure le prix de l'interprétabilité ; sa faiblesse, cohérente avec le flat maximum (Hand et Henley, 1997), indique que les interactions apportent peu sur ce périmètre.

Les contributions SHAP du challenger (Lundberg et Lee, 2017) servent à vérifier que ses variables déterminantes coïncident avec celles retenues dans la grille, ce qui appuie la robustesse de la sélection.

La représentativité du modèle est testée par un classifieur à deux échantillons, dont les variables responsables sont identifiées par SHAP (Lundberg et Lee, 2017), afin d'isoler l'effet population de l'effet temps.

La sensibilité du binning boosté multivarié est appréciée en le comparant à un binning univarié monotone (Navas-Palencia, 2020), pour mesurer l'effet de la corrélation entre variables.

---

## Deux rappels d'usage

Les phrases de la section 5 restent sobres sur la référence : c'est ton analyse qui doit dominer, pas la littérature.

Là où un bloc est marqué [Contribution] dans le plan, la citation encadre l'acquis, puis tu enchaînes sur ton apport sans le rattacher à une source.
