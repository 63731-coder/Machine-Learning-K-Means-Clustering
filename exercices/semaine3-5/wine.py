import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# Étape 1 : Charger le dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
df = pd.read_csv(url, header=None)

# La première colonne est le label (le cépage)
y = df[0]
# Les colonnes restantes sont les caractéristiques
X = df.iloc[:, 1:]

# Étape 2 : Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Étape 3 : Clustering pour deux valeurs de K (2 et 3)
for k in [2, 3]:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    labels = model.labels_

    # Évaluation
    ari = adjusted_rand_score(y, labels)
    inertia = model.inertia_

    print(f"\nK = {k}")
    print(f"Adjusted Rand Index : {ari:.3f}")
    print(f"Inertie (WCSS) : {inertia:.2f}")

    # Étape 4 : Visualisation (avec deux premières features)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                s=200, c='red', marker='X', label='Centroïdes')
    plt.title(f"KMeans Clustering (K={k})")
    plt.xlabel("Feature 1 (normalisée)")
    plt.ylabel("Feature 2 (normalisée)")
    plt.legend()
    plt.tight_layout()
    plt.show()
