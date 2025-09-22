################
# page 13 : alocohol vs praline
################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# Charger le jeu de données Wine
wine = load_wine()
X = wine.data
feature_names = wine.feature_names

# On choisit 2 colonnes : alcohol et proline
X_selected = X[:, [0, 10]]  # Alcohol (0) et Proline (10)

# Appliquer K-means SANS normalisation
kmeans_original = KMeans(n_clusters=3, random_state=42)
y_pred_original = kmeans_original.fit_predict(X_selected)

# Appliquer une normalisation sur les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Appliquer K-means APRÈS normalisation
kmeans_scaled = KMeans(n_clusters=3, random_state=42)
y_pred_scaled = kmeans_scaled.fit_predict(X_scaled)

# Afficher les deux résultats
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Graphe AVANT normalisation
ax[0].scatter(X_selected[:, 0], X_selected[:, 1], c=y_pred_original, cmap='viridis', edgecolors='k')
ax[0].scatter(kmeans_original.cluster_centers_[:, 0], kmeans_original.cluster_centers_[:, 1],
              s=200, c='red', marker='X', label="Centroids")
ax[0].set_xlabel(feature_names[0])
ax[0].set_ylabel(feature_names[10])
ax[0].set_title("K-Means AVANT StandardScaler")

# Graphe APRÈS normalisation
ax[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred_scaled, cmap='viridis', edgecolors='k')
ax[1].scatter(kmeans_scaled.cluster_centers_[:, 0], kmeans_scaled.cluster_centers_[:, 1],
              s=200, c='red', marker='X', label="Centroids")
ax[1].set_xlabel(feature_names[0])
ax[1].set_ylabel(feature_names[10])
ax[1].set_title("K-Means APRÈS StandardScaler")

plt.show()


################
# page 15 : voitures
################

import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# Initialiser l'outil de mise à l’échelle
scale = StandardScaler()

# Charger le fichier CSV (doit être dans le même dossier que ton .py)
df = pandas.read_csv("data.csv")

# Afficher le contenu du fichier
print("Données originales :")
print(df)

# Sélectionner les colonnes qu’on veut normaliser
X = df[['Weight', 'Volume']]

# Appliquer StandardScaler : centrer + réduire
scaledX = scale.fit_transform(X)

# Afficher les valeurs normalisées
print("\nDonnées après normalisation :")
print(scaledX)

################
# page 17
################

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score

# Charger le dataset Iris
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])

# Préparer les données pour KMeans
X = iris_df.drop('target', axis=1)  # Données sans la colonne "target"
y_true = iris_df['target']          # Les vraies classes (pour comparaison)

# Normalisation des données (important pour KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Appliquer KMeans avec k=3 (car on sait qu'il y a 3 espèces d'iris)
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
kmeans.fit(X_scaled)

# Obtenir les étiquettes de cluster
y_kmeans = kmeans.labels_

# Évaluer la qualité du clustering avec ARI
ari = adjusted_rand_score(y_true, y_kmeans)
print(f"Adjusted Rand Index: {ari:.3f}")

# Visualisation en 2D (avec les 2 premières colonnes)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=100, c='red', label='Centroids')
plt.title('KMeans Clustering of Iris Dataset')
plt.xlabel('Sepal length (standardized)')
plt.ylabel('Sepal width (standardized)')
plt.legend()
plt.show()
