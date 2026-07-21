Encadré — Comment se lisent les mesures de performance

Le BIC arbitre entre qualité d'ajustement et parcimonie. Il vaut 
BIC
=
−
2
 
ℓ
(
𝛽
^
)
+
𝑘
ln
⁡
𝑛
BIC=−2ℓ(
β
^
	​

)+klnn, où 
ℓ
ℓ est la log-vraisemblance, 
𝑘
k le nombre de paramètres et 
𝑛
n le nombre d'observations. Le second terme est un péage : chaque variable ajoutée doit gagner sa place en améliorant suffisamment l'ajustement. Entre deux modèles, on retient celui dont le BIC est le plus faible.

L'AUC mesure la capacité du score à ordonner la population. Elle vaut la probabilité qu'un défaillant tiré au hasard reçoive un score plus mauvais qu'un non-défaillant tiré au hasard. Une AUC de 0,5 correspond à un tri au hasard, une AUC de 1 à un tri parfait.

Le Gini est la même information sur une échelle plus parlante, 
Gini
=
2
×
AUC
−
1
Gini=2×AUC−1. Il va de 0 pour un score sans pouvoir discriminant à 1 pour un score parfait. C'est l'unité d'usage en risque de crédit : un Gini de 0,45 signifie que le score capte 45 % du pouvoir de séparation maximal.

Le ratio de performance dit combien de fois le score fait mieux qu'un tri au hasard. Il rapporte l'aire sous la courbe précision-rappel à la prévalence du défaut, 
ratio
=
AUC-PR
/
𝜋
ratio=AUC-PR/π. Un tri aléatoire produit une courbe précision-rappel plate au niveau de la prévalence, donc une AUC-PR égale à 
𝜋
π et un ratio de 1. Un ratio de 5 signifie que le score concentre cinq fois plus de défaillants qu'une sélection au hasard. Le défaut étant rare, cette mesure est plus sensible que l'AUC classique à ce qui se passe sur la population risquée, celle qui décide de l'octroi.
