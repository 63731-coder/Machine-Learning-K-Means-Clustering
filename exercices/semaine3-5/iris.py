import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Étape 1 : Chargement du dataset
data = load_iris()
X = data.data  # les 4 features
y = data.target  # 0=setosa, 1=versicolor, 2=virginica
feature_names = data.feature_names

# Étape 2 : Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Étape 3 : Clustering pour K = 2 et K = 3
for k in [2, 3,6,7]:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    labels = model.labels_

    # Évaluation
    ari = adjusted_rand_score(y, labels)
    inertia = model.inertia_

    print(f"\nK = {k}")
    print(f"Adjusted Rand Index : {ari:.3f}")
    print(f"Inertie (WCSS) : {inertia:.2f}")

    # Étape 4 : Visualisation avec les 2 premières features
    plt.figure(figsize=(6, 5))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='plasma', s=50)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                s=200, c='black', marker='X', label='Centroïdes')
    plt.title(f"KMeans - Iris Dataset (K={k})")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    plt.tight_layout()
    plt.show()
