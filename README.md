# Pipeline Data Cleaning & ML - Projet Voitures

Projet personnel d'auto-formation (réalisé en parallèle de ma L2 Informatique).
L'objectif est de monter en compétence sur la manipulation de données en construisant un pipeline complet de nettoyage et de préparation pour le Machine Learning.
L'objectif : prendre un dataset "sale" (erreurs de saisie, formats bizarres) et automatiser tout le nettoyage et la préparation pour du Machine Learning.

# C'est quoi ?
Un script Python qui transforme des données brutes en données exploitables pour une IA.
Les données brutes sont dispos dans le dossier `data/`.

# Structure
* `data/` : Contient `voitures_sales.csv` (le dataset avec les erreurs).
* `src/` : Contient le code (`projet2.py`) et le dashboard.
* `requirements.txt` : Les librairies à installer.

## Ce que fait le script (Pipeline)
Le code est divisé en classes pour que ce soit propre :

1. **Nettoyage (DataCleaner)** :
   - Réparation des prix (gestion des "k€", espaces, typos).
   - Nettoyage des unités (km, ch).
   - Création de la variable `Age` à partir de l'année.
   - Suppression des doublons.

2. **Analyse (DataAnalyzer)** :
   - Check rapide des valeurs manquantes et des types.

3. **Préparation ML (MLPipeline)** :
   - Split Train/Test pour ne pas biaiser le modèle.
   - Imputation : On remplace les trous (Médiane pour les chiffres, Plus fréquent pour le texte).
   - Encodage : Target Encoding pour les marques/modèles.
   - Scaling : RobustScaler pour gérer les outliers.

# Observations & Limites

En analysant les résultats (notamment via le Target Encoding), j'ai relevé des incohérences sémantiques dues à la génération aléatoire du dataset :
* Exemple : On retrouve le même modèle ("Sandero") associé à des marques différentes ("Dacia" vs "BMW").
* **Impact** : Le modèle mathématique traite ces cas correctement , mais dans un contexte réel, une étape de validation  serait nécessaire en amont comme vérifier le modèle en cohérence avec la marque.
## Comment lancer le projet

1. Installer les dépendances :
pip install -r requirements.txt

2. Lancer le script :
python src/projet2.py