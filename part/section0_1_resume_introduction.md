# 0. Résumé

BNP Paribas attribue à chaque demande de crédit d'un professionnel une note qui mesure son risque de ne pas rembourser et oriente la décision d'accorder ou de refuser. Ce mémoire porte sur la refonte de cette note, appelée score d'octroi.

On construit une nouvelle grille de notation, simple à lire et à justifier, dont la construction est guidée par un modèle de référence plus puissant mais opaque. La grille est d'abord comparée au score en place. Elle est ensuite située par rapport à la performance la plus élevée que ce modèle de référence permet d'atteindre, pour mesurer ce que coûte sa simplicité. On teste enfin si les données de transaction des clients améliorent la prédiction, et si un seul modèle suffit ou s'il faut en spécialiser plusieurs.

La nouvelle grille améliore le score existant de `[DONNÉE]` et se tient à `[DONNÉE]` de la performance maximale, ce qui montre que sa lisibilité se paie peu. Sur le périmètre principal, `[DONNÉE : un modèle général suffit / deux modèles spécialisés se justifient]`. On préconise `[DONNÉE : recommandation de déploiement]`.

---

# 1. Introduction

Accorder ou refuser un crédit à un professionnel engage la banque sur la durée. Pour éclairer cette décision, un score attribue à chaque demandeur une note qui reflète son risque de défaut. Plus cette note sépare nettement les futurs défaillants des bons payeurs, mieux la banque arbitre entre le développement de son activité et la maîtrise de ses pertes.

BNP Paribas dispose déjà d'un tel score sur son périmètre de professionnels et d'entrepreneurs individuels. Ce score doit être refondu, ce qui fait l'objet du mémoire. La cible retenue est le défaut observé dans les douze mois, au sens de la définition réglementaire. La question est double : bâtir une grille plus performante, et le faire sans renoncer à la lisibilité qu'exigent le métier et le contrôle, car une note d'octroi doit pouvoir être expliquée et défendue.

La thèse défendue tient en une phrase. Une grille de notation simple, dont le découpage des variables est appris par un modèle plus puissant, peut battre le score existant tout en restant transparente, s'approcher de la performance des modèles les plus complexes, et éclairer s'il faut, sur le périmètre principal, spécialiser plusieurs modèles ou s'en tenir à un seul.

La démarche est directe. La grille est construite en s'appuyant sur un modèle de référence plus puissant, qui guide le choix des variables et fixe la performance à viser. Elle est ensuite comparée au score existant et à ce plafond. On teste l'apport des données de transaction des clients, puis le besoin de spécialiser plusieurs modèles selon les sous-populations. On vérifie enfin que le modèle appris sur les clients déjà présents s'applique aux nouvelles demandes, et on éprouve la solidité de chaque choix. Les sections qui suivent présentent l'environnement du stage, la problématique et les données, la construction et l'inférence du modèle, puis la discussion des résultats.
