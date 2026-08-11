# Perspective : estimation d'un taux de défaut contrefactuel par appariement pronostique

*Perspective méthodologique non mise en œuvre dans ce mémoire. Elle prolonge la discussion de la section 5.4 sur la lecture du taux de défaut des accordés, et s'articule avec la variante d'échantillonnage de la section 5.5. Les schémas ASCII ci-dessous sont des esquisses à redessiner en figures autonomes.*

---

## Objectif

Le taux de défaut observé sur les dossiers accordés est biaisé, d'un sens indéterminé, par l'effet du crédit sur le moment du défaut (section 5.4.3). L'objectif est de lui substituer un taux de défaut de référence, estimé sur des clients dont le comportement n'est pas perturbé par un prêt récent, pour caler le seuil sur une base mieux définie.

## Étape 1 : restreindre la population de référence

La population de référence est constituée des clients du stock n'ayant reçu aucun prêt dans les douze mois précédant ni dans les douze mois suivant la date de photographie. Cette double exclusion écarte deux sources de perturbation : un prêt antérieur dont l'effet se prolongerait jusqu'à la photo, et un prêt postérieur qui viendrait modifier le comportement pendant la fenêtre d'observation du défaut.

```
        Exclusion des clients ayant reçu un prêt sur la fenêtre [t0 - 12 mois, t0 + 12 mois]

   |<----------------- 12 mois ----------------->|<----------------- 12 mois ----------------->|
 t0-12                                          t0                                           t0+12
                                              (photo)
                                                |<======= fenêtre d'observation du défaut =======>|
```

> **[ZONE — Figure : fenêtre d'exclusion autour de la photographie]**
> *Type : schéma chronologique.* Représenter l'axe du temps, la date de photographie t0, la fenêtre d'exclusion symétrique de douze mois de part et d'autre, et la fenêtre d'observation du défaut de douze mois après t0. Annoter que seuls les clients sans prêt sur toute la fenêtre d'exclusion sont retenus comme référence.

## Étape 2 : apparier chaque demandeur sur la PDO, au même mois

À chaque demandeur $i$ du flux, de score $s_i = \mathrm{PDO}(i)$, sont associés les clients de la population de référence partageant la même valeur de PDO et le même mois de photographie. Cet ensemble apparié est noté $M_i$, de taille $K_i = |M_i|$.

```
   Flux (demandeurs)                 Stock de référence (sans prêt sur la fenêtre d'exclusion)

     i , PDO = v      ───────────►   { j : PDO_j = v , même mois }  ──►  défauts observés y_j
```

> **[ZONE — Figure : appariement d'un demandeur à ses clients de référence]**
> *Type : schéma d'appariement.* À gauche, un demandeur du flux avec sa valeur de PDO. À droite, le groupe de clients du stock de référence de même PDO et même mois. Une flèche relie le demandeur au groupe, dont on extrait le taux de défaut.

## Étape 3 : estimer le taux de défaut contrefactuel

En notant $y_j \in \{0,1\}$ le défaut observé du client de référence $j$ sur sa fenêtre de douze mois, le taux de défaut contrefactuel s'estime par

$$\hat{\tau}_{cf} = \frac{1}{n}\sum_{i=1}^{n}\frac{1}{K_i}\sum_{j\in M_i} y_j .$$

La moyenne interne estime la probabilité de défaut associée au niveau de risque du demandeur, mesurée sur des clients non perturbés par un prêt. La moyenne externe agrège sur les $n$ demandeurs du flux, chacun comptant pour un.

## Une écriture équivalente par strates

La PDO étant discrète et à support fini, l'estimateur admet une forme équivalente plus simple. En notant $V$ l'ensemble des valeurs possibles de la PDO, $R_v$ les clients de référence de PDO égale à $v$, $r_v$ leur taux de défaut, et $w_v$ la part du flux de PDO égale à $v$ :

$$r_v = \frac{1}{|R_v|}\sum_{j\in R_v} y_j , \qquad w_v = \frac{1}{n}\,\#\{\,i : s_i = v\,\}, \qquad \hat{\tau}_{cf} = \sum_{v\in V} w_v\, r_v .$$

Cette écriture est une standardisation directe : le taux de défaut de référence de chaque niveau de risque, pondéré par la fréquence de ce niveau dans le flux. Elle coïncide avec la double moyenne lorsque l'appariement retient, pour chaque demandeur, l'ensemble des clients de référence de même PDO. Elle est préférable en pratique, car elle mutualise tous les clients de référence d'un niveau donné, ce qui réduit la variance d'estimation.

## Étape 4 : le cas sans appariement exact, pondération par noyau

Lorsque aucun client de référence ne partage exactement la PDO d'un demandeur, l'appariement exact échoue ($K_i = 0$) et l'estimateur n'est pas défini. Une généralisation consiste à pondérer tous les clients de référence par un poids décroissant avec l'écart de PDO :

$$\hat{\tau}_{cf} = \frac{1}{n}\sum_{i=1}^{n}\frac{\displaystyle\sum_{j\in R} K_h\!\left(s_i - s_j\right) y_j}{\displaystyle\sum_{j\in R} K_h\!\left(s_i - s_j\right)} , \qquad K_h(u) = K\!\left(\frac{u}{h}\right),$$

où $R$ est la population de référence, $K$ un noyau positif et symétrique, par exemple gaussien ou triangulaire, et $h$ une fenêtre. Un client de référence dont la PDO est proche de celle du demandeur pèse davantage qu'un client éloigné. L'appariement exact de l'étape 3 en est le cas limite, obtenu avec un noyau indicateur et une fenêtre tendant vers zéro. Le choix de la fenêtre $h$ arbitre entre biais et variance : une fenêtre étroite réduit le biais mais accroît la variance, une fenêtre large fait l'inverse.

## Hypothèse d'identification et limites

L'estimateur repose sur une hypothèse à énoncer explicitement. Conditionnellement au niveau de PDO, et le cas échéant aux caractéristiques d'appariement retenues, les clients de référence sans prêt récent sont supposés représentatifs du comportement qu'aurait eu le demandeur en l'absence de prêt. C'est une hypothèse de suffisance du score pronostique, analogue à une condition d'ignorabilité. Elle est d'autant plus crédible que la PDO résume bien le risque, et peut être renforcée en appariant aussi sur d'autres caractéristiques : segment, famille d'activité, ancienneté, intensité transactionnelle.

Deux limites accompagnent cette hypothèse. La première est le support commun : chaque niveau de risque présent dans le flux doit exister dans la population de référence, faute de quoi une part du flux reste sans contrefactuel. La seconde est la qualité de l'appariement, qui demande une validation, par exemple l'examen de l'équilibre des caractéristiques entre demandeurs et clients appariés. Ces éléments constituent une extension du présent travail, reprise en perspective dans la conclusion.
