# Estimation robuste de la DoA large bande par CNN2D et Critère Normalisé Temps-Fréquence Pondéré

Ce projet implémente une chaîne complète d’estimation de la direction d’arrivée (DoA) d’un signal acoustique à partir de signaux multi-antennes simulés. L’approche repose sur un réseau de neurones convolutionnel profond (CNN2D) enrichi par des blocs résiduels et Squeeze-and-Excitation (SE), et optimisé via un critère de perte pondéré appliqué à une représentation temps-fréquence normalisée.

## Structure des fichiers

* **generate\_data.py** : Génère des signaux simulés avec bruit et DoA aléatoire.
* **dataset.py** : Applique la STFT à chaque signal d’antenne, normalise les tenseurs.
* **model.py** : Définit le modèle CNN2D avec blocs résiduels et SE.
* **train.py** : Entraîne le modèle et sauvegarde les poids les plus performants.
* **test.py** : Évalue le modèle sur le jeu de test et calcule l’erreur absolue moyenne.

## Instructions d’utilisation

### 1. Génération des données

```bash
python generate_data.py
```

* Génère 10 000 paires (signal multi-antennes + DoA) dans le dossier `dataset/`.

### 2. Entraînement du modèle

```bash
python train.py
```

#### 10 Epoch
```bash
Epoch 001,Train Loss: 1761.0637, Validation Loss: 887.4044 
Epoch 002,Train Loss: 997.9656, Validation Loss: 849.4402 
Epoch 003,Train Loss: 911.5387, Validation Loss: 802.2366 
...
...
Epoch 009,Train Loss: 516.8367, Validation Loss: 499.3397 
Epoch 010,Train Loss: 519.3103, Validation Loss: 535.3936
```

#### 100 Epoch

```bash
Epoch 001,Train Loss: 1750.0687, Validation Loss: 897.4074 
...
...
Epoch 094, Train Loss: 388.9873, Validation Loss: 394.6313 
Epoch 095, Train Loss: 387.5716, Validation Loss: 393.4338 
Epoch 096, Train Loss: 399.0940, Validation Loss: 392.9662 
Epoch 097, Train Loss: 393.8169, Validation Loss: 403.2364 
Epoch 098, Train Loss: 391.1696, Validation Loss: 393.6171 
Epoch 099, Train Loss: 390.2861, Validation Loss: 393.5433 
Epoch 100, Train Loss: 388.3509, Validation Loss: 393.1766 
```

**MAE ≈ √(MSE ou Loss), mais ceci n’est qu’une estimation approximative. Calculez directement le MAE pour plus de précision !
MAE = 10 signifie que vos prédictions s’écartent en moyenne de 10 degrés par rapport à la valeur réelle.**

* Sauvegarde le modèle dans `cnn2d_wideband.pt`.

### 3. Test du modèle

```bash
python test.py
```
#### 10 Epoch
```bash
#### 100 Epoch
Pred: -52.76° / True: -35.39° → Error: 17.37°
Pred: -53.03° / True: -51.17° → Error: 1.87°
...
...
Pred: 29.67° / True: 32.01° → Error: 2.34°
Pred: 65.16° / True: 68.61° → Error: 3.45°
Pred: 66.31° / True: 65.70° → Error: 0.60°
Mean Absolute Error: 12.81
```


#### 100 Epoch
```text
Pred: 30.01° / True: 25.99° → Error: 4.02°
Pred: 30.47° / True: 44.99° → Error: 14.53°
...
Pred: 67.31° / True: 65.70° → Error: 1.60°
Mean Absolute Error: 10.96°
```

* **Erreurs faibles (<5°)** : Fréquentes
* **Erreurs modérées (5–20°)** : Présentes en zone intermédiaire
* **Erreurs fortes (>20°)** : Cas extrêmes (angles aux limites ou bruit élevé)


* Affiche la MAE et les erreurs DoA prédites vs vraies.


## Architecture du Modèle CNN2D\_Doa

* 4 blocs résiduels convolutionnels (8 → 32 → 64 → 128 → 256 canaux)
* Blocs SE adaptatifs intégrés dans chaque bloc résiduel
* Pooling global (AdaptiveAvgPool2D) suivi de 2 couches fully-connected
* Activations LeakyReLU (slope = 0.1), Dropout (p = 0.3) dans convolutions et dense
* Perte : **Mean Squared Error pondérée** avec pondération basée sur la cible


## Résultats de l’entraînement

* Meilleure validation atteinte : **392.61** à l'époque 82
* Comportement typique d’une convergence progressive avec stabilisation


## Analyse visuelle des données (figures)


### **Carte de chaleur moyenne temps-antennes**

**Fichier** : `Heatmap_Temps_Antennes.png`
**Description** :
Cette carte de chaleur montre la moyenne de l’intensité du signal reçue par chaque antenne au cours du temps. Elle permet d’identifier des schémas de propagation ou de réception dominants, ainsi que des variations de phase ou de synchronisation entre les canaux. Elle est utile pour repérer visuellement la présence d’une onde plane incidente ou d’un signal cohérent.


### **Écart-type (bruit) par antenne**

**Fichier** : `STD_Bruit_Antennes_Exemple_1297.png`
**Description** :
Ce graphique présente la distribution de l’écart-type (σ) du bruit mesuré sur chaque antenne pour un exemple donné. Il met en évidence la stabilité ou la variabilité du bruit entre les capteurs, permettant d’identifier d’éventuelles antennes défaillantes ou des artefacts matériels.


### **Signal temporel multi-antennes (DoA ≈ 13.96°)**

**Fichier** : `Signal_Antennes_DoA_13.96.png`
**Description** :
Visualisation synchronisée des signaux bruts issus de toutes les antennes lors d’un scénario où la direction d’arrivée estimée est d’environ 13.96°. Ce graphique permet d’analyser le décalage temporel relatif entre les canaux, reflet direct de l’angle d’arrivée du front d’onde.


### 🧭 **Vue globale des signaux temporels**

**Fichier** : `Signal_Temporel_Antennes.png`
**Description** :
Affichage simultané des signaux temporels pour toutes les antennes sur une fenêtre complète. Cette vue fournit un aperçu macro de la dynamique du signal, de sa périodicité, et des niveaux d’énergie.

### **Moyenne temporelle multi-antennes**

**Fichier** : `Signal_Temporel_Moyen_Antennes.png`
**Description** :
Courbe représentant la moyenne temporelle des signaux de toutes les antennes. Elle atténue les composantes spécifiques à un capteur et met en avant la structure commune du signal incident, utile pour l’analyse énergétique globale.

### **Spectre STFT individuel – Antenne 0**

**Fichier** : `Spectre_STFT_Antenne_0.png`
**Description** :
Représentation temps-fréquence du signal reçu par l’antenne 0 via transformée de Fourier à court terme (STFT). Elle révèle les composantes fréquentielles dominantes et leur évolution dans le temps, souvent liée au mouvement ou au changement de source.

### **STFT moyen – Antenne 0**

**Fichier** : `Spectre_STFT_Moyen_Antenne_0.png`
**Description** :
Spectrogramme résultant de la moyenne temporelle des STFT appliquées au signal de l’antenne 0. Il permet de détecter des signatures spectrales persistantes, telles que les fréquences de modulation ou les bruits stationnaires.

### **Zoom temporel – Antenne 0**

**Fichier** : `Zoom_Antenne_0.png`
**Description** :
Zoom sur un segment temporel spécifique du signal reçu par l’antenne 0. Cette vue granulaire permet d’analyser finement les transitions, impulsions, ou réponses transitoires présentes dans le signal.

### **Zoom temps-fréquence – Antenne 0**

**Fichier** : `Zoom_TF_Antenne_0.png`
**Description** :
Extrait localisé dans le domaine temps-fréquence centré sur une zone d’intérêt du spectrogramme de l’antenne 0. Cette analyse fine permet d’examiner la présence de composantes intermittentes, interférences ou micro-modulations.


## Conclusion

Le modèle CNN2D exploitant la représentation STFT normalisée par canal et enrichi par des blocs résiduels et SE, montre une capacité robuste d’estimation du DoA sur des données simulées bruitées. Les performances obtenues indiquent une erreur moyenne raisonnable à \~11°, et une généralisation stable sur l'ensemble de test. Le pipeline est modulaire et facilement extensible à d'autres configurations d'antennes ou de réseaux profonds.

