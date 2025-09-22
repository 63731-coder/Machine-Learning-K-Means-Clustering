import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Étape 1 : Chargement du dataset
data = load_breast_cancer()
X = data.data
y = data.target  # étiquettes vraies : 0 = Malin, 1 = Bénin
feature_names = data.feature_names

# Étape 2 : Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Étape 3 : Appliquer K-means pour K=2 et K=3
for k in [2, 3,8,9,4]:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    labels = model.labels_

    # Évaluation
    ari = adjusted_rand_score(y, labels)
    inertia = model.inertia_

    print(f"\nK = {k}")
    print(f"Adjusted Rand Index : {ari:.3f}")
    print(f"Inertie (WCSS) : {inertia:.2f}")

    # Étape 4 : Visualisation
    plt.figure(figsize=(6, 5))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='coolwarm', s=40)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                c='black', marker='X', s=200, label='Centroïdes')
    plt.title(f"Clustering du cancer du sein (K={k})")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    plt.tight_layout()
    plt.show()
