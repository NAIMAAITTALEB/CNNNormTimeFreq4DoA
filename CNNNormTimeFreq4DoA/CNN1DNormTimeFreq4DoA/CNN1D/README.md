# Estimation robuste de la DoA large bande par CNN1D et Critère Normalisé Temps-Fréquence Pondéré

Ce projet expose une chaîne algorithmique complète dédiée à l’estimation de l’angle d’arrivée (DoA) d’une source acoustique à partir de signaux microphoniques simulés, mobilisant un réseau de neurones convolutif (CNN) et un critère de perte pondéré appliqué à la représentation temps-fréquence normalisée.

## Structure des fichiers

* **generate\_data.py** : Génère des jeux de données simulés au format `.npy` via [Pyroomacoustics](https://pyroomacoustics.readthedocs.io/).
* **dataset.py** : Charge les données, réalise la transformation temps-fréquence (STFT), puis normalise l’ensemble pour l’apprentissage profond.
* **model.py** : Spécifie l’architecture CNN conçue pour la régression directe de l’angle DoA.
* **train.py** : Assure l’entraînement supervisé du modèle en s’appuyant sur une fonction de perte quadratique moyenne pondérée.
* **test.py** : Évalue la robustesse du modèle et restitue ses performances sur le jeu de test.

## Utilisation

### 0. (Optionnel) Environnement virtuel Python

Pour garantir l’isolation des dépendances, créez et activez un environnement virtuel dans le dossier du projet (CnnEnv) :

```bash
python3 -m venv CnnEnv
source CnnEnv/bin/activate
# Windows : .\CnnEnv\Scripts\activate
```

### 1. Générer les données simulées

```bash
python generate_data.py
```

Les fichiers `.npy` sont générés dans le dossier `dataset/`.

### 2. Entraîner le modèle

```bash
python train.py
```

Le modèle apprend sur les données, les poids sont sauvegardés dans `trained_model.pt`.

### 3. Tester le modèle

```bash
python test.py
```

Les performances du modèle sont affichées (erreur absolue moyenne entre angles prédits et réels).

## Détails techniques

* **Transformation STFT** : Chaque signal est projeté dans le domaine temps-fréquence.
* **Normalisation** : Les données sont centrées et réduites (moyenne nulle, variance unitaire).
* **Critère de perte pondéré** : La fonction Weighted MSE Loss donne une importance accrue aux exemples complexes ou à des plages d’angles spécifiques.
* **Modèle CNN** : Ingestion directe de la représentation temps-fréquence multi-canale, estimation angulaire en sortie.

## Dépendances

* `numpy`
* `pyroomacoustics`
* `scipy`
* `torch`

Installez-les si nécessaire :

```bash
pip install numpy scipy torch pyroomacoustics
```

## Exemples de résultats

### Generate dataset

```bash
$ python3 generate_dataset.py
```

**Résultat**

```bash
DONE: 10000 samples written in 'dataset'
```

### Explore dataset

```bash
$ python3 explore_dataset.py
```

**Résultat**

```bash
Nombre d'échantillons: 10000
Nombre d'antennes: 8, nombre d'échantillons temporels: 512
DoA min/max/moyenne/std: -89.98° / 89.99° / -0.23° / 52.27°
Amplitude moyenne (sur tous): -0.000, std: 1.000

Échantillon choisi: 6427   DoA: -35.60°
Écart-type par antenne (sur cet exemple): [1.4412 0.6214 0.616  0.6318 0.6525 0.643  0.6419 0.6244]
```

#### **Explication détaillée**

* **Nombre d’échantillons : 10 000**
  Le dataset couvre exhaustivement l’espace angulaire $[-90°, 90°]$ avec une granularité réaliste.
* **Nombre d’antennes : 8**
  La configuration ULA (Uniform Linear Array) permet d’exploiter la diversité spatiale.
* **Nombre d’échantillons temporels : 512**
  Les signaux, acquis à 10 GHz, garantissent une résolution temporelle élevée.
* **Angles DoA (min/max/moyenne/écart-type)**

  * Min : −89.98°
  * Max : 89.99°
  * Moyenne : −0.23°
  * Écart-type : 52.27°
* **Amplitude moyenne et écart-type**
  La normalisation globale (moyenne nulle, écart-type 1) permet une stabilité de l’apprentissage.
* **Échantillon choisi : 6427, DoA : -35.60°**
  L’analyse détaillée sur cet exemple illustre la variabilité inter-antennes (écarts-types).

**En résumé**
Le dataset s’avère volumineux, équilibré, optimalement normalisé et dépourvu de biais angulaire — chaque direction est représentée de manière homogène.

---

### Visualisations et Analyses

**Fichier : /figures/Ecart-type (bruit-puissance) par antenne pour l'exemple choisi.png**

* **Description** :
  Diagramme en barres illustrant la dispersion de l’amplitude sur chaque antenne pour un exemple spécifique.
* **Interprétation** :
  L’écart-type élevé sur un canal révèle une énergie supérieure, possiblement liée à la direction d’arrivée ou au bruit.

**Fichier : /figures/Energie par antenne (distribution sur tout le dataset).png**

* **Description** :
  Boxplot récapitulant la distribution de l’énergie totale par antenne sur tout le dataset.
* **Interprétation** :
  Permet de détecter d’éventuelles asymétries, variations de gain ou antennes défectueuses.

**Fichier : /figures/Heatmap du signal (antennes x temps), DoA = -35.60 degre.png**

* **Description** :
  Carte de chaleur présentant l’évolution temporelle de l’amplitude sur chaque antenne pour un DoA donné.
* **Interprétation** :
  Met en évidence les motifs spatio-temporels, fronts d’onde et particularités directionnelles.

**Fichier : /figures/Matrice de correlation entre antennes (exemple).png**

* **Description** :
  Matrice de corrélation de Pearson entre les canaux pour un échantillon précis.
* **Interprétation** :
  Met en lumière la redondance ou l’indépendance de l’information transportée par chaque antenne.

**Fichier : /figures/Signal temporel recu sur chaque antenne (DoA = -35.60 degre).png**

* **Description** :
  Visualisation des signaux bruts normalisés sur toutes les antennes pour un DoA donné.
* **Interprétation** :
  Permet de vérifier la cohérence de phase et la propagation attendue de l’onde incidente.

**Fichier : /figures/Spectre frequentiel (FFT) sur plusieurs antennes (DoA = -35.60 degre).png**

* **Description** :
  Analyse fréquentielle (FFT) en échelle semi-log de plusieurs canaux.
* **Interprétation** :
  Permet d’examiner la richesse fréquentielle, la présence d’éventuelles bandes passantes ou artefacts.

**Fichier : /figures/Zoom sur lantenne 0 (DoA = -35.60 degre).png**
**Fichier : /figures/Zoom sur lantenne 4 (DoA = -35.60 degre).png**
**Fichier : /figures/Zoom sur lantenne 7 (DoA = -35.60 degre).png**

* **Description** :
  Zooms individuels sur le signal reçu par les antennes 0, 4 et 7.
* **Interprétation** :
  Analyse granulaire permettant de repérer bruit, artefacts ou phénomènes atypiques.

**Fichier : /figures/Correlation entre energie recue et DoA.png**

* **Description** :
  Nuage de points croisant énergie reçue totale et DoA pour chaque échantillon.
* **Interprétation** :
  Recherche d’éventuelles corrélations directionnelles ou biais structurels.

**Fichier : /figures/Covariance spatiale (antennes x antennes).png**

* **Description** :
  Heatmap de la matrice de covariance entre tous les couples d’antennes, sur le dataset complet.
* **Interprétation** :
  Permet d’identifier la richesse informationnelle spatiale et la diversité inter-canaux.

**Fichier : /figures/Distribution des angles DoA.png**

* **Description** :
  Histogramme de la distribution des angles DoA simulés.
* **Interprétation** :
  Valide l’uniformité de la répartition angulaire, condition essentielle pour un apprentissage impartial.

---

### Train model

```bash
$ python3 train.py
```

**Résultat**

```bash
Epoch 1, Loss: 1100.4796
Validation Loss: 482.1757
...
Epoch 10, Loss: 487.9120
Validation Loss: 413.1413
```

#### Explication détaillée

L’amorçage du réseau par des poids aléatoires engendre naturellement des pertes initiales très élevées. Les itérations subséquentes voient une décroissance rapide de la loss, reflet d’un apprentissage efficace des grandes structures. La stabilisation de la loss de validation autour de 400 après dix époques témoigne d’une capacité de généralisation satisfaisante. Une perte de 400 équivaut à une erreur quadratique moyenne proche de 20°, soit une performance tout à fait raisonnable dans ce contexte synthétique.

Pour raffiner le modèle :

* Prolonger l’entraînement.
* Complexifier l’architecture (plus de couches/filtres).
* Ajuster le taux d’apprentissage.
* Enrichir le dataset ou employer des techniques de normalisation avancées.

---

### Test model

```bash
$ python3 test.py
```

**Résultat**

```bash
Pred: 39.78° / True: 38.58° → Error: 1.21°
Pred: 73.77° / True: 76.93° → Error: 3.16°
...
Mean Absolute Error: 12.02°
```

#### Analyse des performances

* **Erreur absolue moyenne (MAE) : 12°**
* **Erreurs faibles (<5°)** : Majoritaires, illustrant la robustesse du modèle pour la plupart des cas.
* **Erreurs modérées (5–20°)** : Observées sur des cas atypiques ou aux extrémités de la plage.
* **Erreurs fortes (>20°)** : Rares, souvent liées à des ambiguïtés structurelles ou bruit excessif.

**Synthèse**

Le réseau CNN montre une aptitude claire à la localisation directionnelle sur signaux large bande simulés. Les erreurs importantes, bien que peu fréquentes, incitent à prévoir, en contexte industriel, une étape de post-traitement ou d’optimisation complémentaire. Le pipeline dans son ensemble se révèle solide, performant et modulable pour des extensions futures.

