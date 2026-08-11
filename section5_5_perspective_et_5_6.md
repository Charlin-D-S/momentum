# Ajout à 5.5 (perspective) et rédaction de 5.6

*Note. Les `[à compléter]` signalent des valeurs ou verdicts à renseigner une fois les chiffres produits.*

---

## 5.5.x Une variante du plan d'échantillonnage : étaler la photographie du stock sur douze mois (perspective)

Le plan d'échantillonnage retenu photographie le stock à deux dates fixes espacées de six mois (section 3.3). Une variante, non mise en œuvre dans ce mémoire, mérite d'être discutée car elle lèverait plusieurs limites identifiées plus haut.

Le principe est d'étaler les dates de photographie sur l'ensemble de l'année. Chaque client resterait observé au plus deux fois, à six mois d'intervalle, mais le couple de dates serait réparti aléatoirement sur les douze mois : un client observé en juin le serait de nouveau en décembre, un autre en février puis en août, et ainsi de suite. Collectivement, les dates d'observation couvriraient alors tous les mois.

Cette variante présenterait plusieurs avantages.

- **Neutralisation des effets calendaires.** En répartissant les observations sur toute l'année, le plan cesse de dépendre des conditions particulières de deux mois précis. Le signal appris ne reflète plus une conjoncture saisonnière isolée.
- **Comparabilité avec le flux d'octroi.** Le flux arrive de façon continue sur tous les mois. Un stock étalé sur douze mois épouse cette structure temporelle. L'obstacle soulevé en 5.4.2, qui interdit de comparer directement un stock à deux dates et un flux continu, disparaîtrait : les deux populations deviendraient comparables sur le plan du calendrier.
- **Facilitation de l'appariement contrefactuel.** La construction de contrefactuels décrite en 5.4.4 suppose d'apparier chaque demandeur à des clients du stock comparables. Disposer de clients du stock observés au même mois que chaque demandeur rendrait cet appariement plus naturel et mieux contrôlé sur le temps.

Cette variante ne résout pas tout, et ses limites doivent être énoncées.

- Chaque client reste observé deux fois à six mois d'intervalle. Les deux fenêtres de performance de douze mois continuent de se chevaucher, donc la corrélation intra-emprunteur et la structure de panel (section 3.3) subsistent. La variante traite le calendrier, pas la dépendance temporelle.
- Elle n'ajoute aucune information par client : elle redistribue les dates, sans augmenter le nombre d'observations individuelles.
- L'affectation d'un client à un couple de mois devrait être aléatoire et indépendante du risque, sous peine d'introduire un biais. Sa mise en œuvre suppose enfin une disponibilité homogène des données sur tous les mois.

En résumé, cette variante est une piste d'amélioration du plan d'échantillonnage, dont l'intérêt principal est de rendre le stock comparable au flux et d'outiller la fixation du seuil. Elle est reprise en perspective dans la conclusion.

---

## 5.6 Portée des résultats : pertinence, atouts et limites, robustesse

Cette dernière sous-section prend du recul sur l'ensemble de l'étude. Elle répond à trois questions : les résultats sont-ils pertinents, quels sont les atouts et les limites du travail, et quelle est la robustesse des conclusions.

### 5.6.1 Les résultats sont-ils pertinents ?

La pertinence des résultats se juge à l'aune de la question de départ, qui était de refondre un score d'octroi à la fois plus performant que l'existant et interprétable. Sur ce point, la grille proposée améliore la différenciation du risque par rapport au score en place, avec un gain de [à compléter] points de Gini sur le même périmètre. La refonte se justifie donc par un gain mesurable, et non par le seul renouvellement de la méthode.

La comparaison aux challengers non linéaires confirme la valeur de ce choix. L'écart de performance entre la grille et le meilleur modèle non linéaire est [à compléter], ce qui chiffre le prix de l'interprétabilité et montre empiriquement que les interactions apportent peu sur ce périmètre. Le recours à une grille additive n'est donc pas un renoncement coûteux, mais un compromis favorable entre lisibilité et pouvoir discriminant.

Enfin, le test conduit sur le Pros-ER tranche la question centrale du mémoire, celle de savoir si un modèle général suffit face à deux modèles spécialisés selon la disponibilité des libellés de transaction. Le verdict, [à compléter], est directement exploitable par le métier, puisqu'il arbitre entre un gain de performance et le coût de maintenance de deux modèles. Les résultats sont ainsi pertinents parce qu'ils débouchent sur une décision opérationnelle, et non sur une performance considérée pour elle-même.

### 5.6.2 Quels sont les atouts et les limites de l'étude ?

Le premier atout du travail est de concilier interprétabilité et performance. La grille reste lisible variable par variable, tout en se situant près de la borne atteignable par un modèle non linéaire, ce qui satisfait la double exigence posée au départ. Cette lisibilité s'accompagne d'une traçabilité complète de la construction, chaque étape étant documentée et reproductible, condition indispensable à l'auditabilité d'un score d'octroi. L'étude apporte en outre des contributions méthodologiques à la démarche interne, à savoir l'optimisation des hyperparamètres en amont du binning, la réduction de redondance par clusters de corrélation et la sélection stepwise contrôlée par une pénalisation. Elle borne enfin honnêtement sa propre portée, en explicitant l'asymétrie entre la population de modélisation et la population de décision, ainsi que l'effet ambigu du crédit sur la survenue du défaut.

Les limites sont de plusieurs ordres. La structure de panel, induite par l'observation du stock à deux dates rapprochées, engendre une corrélation intra-emprunteur et des fenêtres de performance chevauchantes, qui rendent optimistes les tests de significativité mobilisés lors du binning et de la sélection. La représentativité du modèle appris sur le stock, puis appliqué au flux d'octroi, n'est pas mesurée par un test formel, faute de pouvoir comparer directement deux populations aux plans d'échantillonnage différents. Le taux de défaut des dossiers accordés, qui sert à fixer le seuil, demeure biaisé d'un sens indéterminé, et ce biais n'est pas corrigé dans le présent travail. Le score s'écarte par ailleurs du cadre IRB homologué, puisqu'il reprend la définition réglementaire du défaut sans viser le calcul d'exigence en fonds propres, et ne comporte donc ni calibration through-the-cycle ni marge de conservatisme réglementaire. La calibration au taux de défaut est enfin traitée comme secondaire, ce qui suffit à l'usage d'octroi mais restreint la réutilisation du score pour d'autres finalités.

### 5.6.3 Quelle est la robustesse des conclusions ?

La robustesse des conclusions a été éprouvée en confrontant chaque choix méthodologique à son alternative, comme exposé en section 5.5. Les conclusions centrales résistent à ces variations selon plusieurs axes.

La performance se maintient hors de la période d'apprentissage. Sur la production 2025, la dégradation reste contenue à [à compléter], et les indices de stabilité de population demeurent [à compléter], ce qui indique que la grille ne surajuste pas la période d'estimation. Les deux logiques de sélection convergent également : le stepwise et la pénalisation LASSO retiennent des ensembles de variables voisins, ce qui désamorce la critique classiquement adressée au stepwise et conforte la parcimonie du modèle. La concordance avec le challenger va dans le même sens, puisque les variables les plus contributives du modèle non linéaire, lues au moyen des valeurs de SHAP, recoupent celles de la grille, signe que la sélection ne néglige pas un signal qu'un modèle plus riche exploiterait. L'examen de la sensibilité aux choix de traitement montre enfin que le codage des valeurs manquantes et le mode de binning modifient [marginalement / sensiblement, à compléter] la performance, ce qui situe le degré de dépendance des résultats à ces décisions.

Ces vérifications ne prétendent pas à l'exhaustivité. Elles établissent que les conclusions principales, à savoir le gain sur le score existant, la proximité de la borne non linéaire et le verdict du test sur les libellés, ne tiennent pas à un réglage particulier, mais résistent au changement d'alternative méthodologique. Les limites qui subsistent, en premier lieu la structure de panel et la représentativité du stock vis-à-vis du flux, sont identifiées et ouvrent les perspectives reprises en conclusion.
