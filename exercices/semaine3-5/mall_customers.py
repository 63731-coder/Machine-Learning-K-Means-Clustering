import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Étape 1 : Charger le dataset
df = pd.read_csv("Mall_Customers.csv")

# Étape 2 : Introduire des valeurs manquantes
df.loc[5:10, "Age"] = np.nan
df.loc[15:20, "Annual Income (k$)"] = np.nan

# Étape 3 : Imputation (remplissage des valeurs manquantes par la moyenne)
imputer = SimpleImputer(strategy="mean")
df[["Age", "Annual Income (k$)"]] = imputer.fit_transform(df[["Age", "Annual Income (k$)"]])

# Étape 4 : Sélection des colonnes numériques pour le clustering
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Étape 5 : Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Étape 6 : Clustering pour K=2 et K=3
# Il n'y a pas de labels ici, donc pas de ARI réel (ou on en génère fictivement)
for k in [2, 3]:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    labels = model.labels_
    inertia = model.inertia_

    print(f"\nK = {k}")
    print(f"Inertie (WCSS) : {inertia:.2f}")

    # Étape 7 : Visualisation
    plt.figure(figsize=(6, 5))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                s=200, c='red', marker='X', label='Centroïdes')
    plt.title(f"KMeans - Mall_Customers (K={k})")
    plt.xlabel("Âge (normalisé)")
    plt.ylabel("Revenu annuel (normalisé)")
    plt.legend()
    plt.tight_layout()
    plt.show()
