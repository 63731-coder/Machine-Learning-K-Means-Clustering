import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA

# Chargement et préparation des données
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names

wine_df = pd.DataFrame(X, columns=feature_names)
wine_df['target'] = y

print("Aperçu des données Wine :")
print(wine_df.head())

# Étape 1 : Vérification des valeurs manquantes
print("\nValeurs manquantes dans le dataset :")
print(wine_df.isnull().sum())  # doit être tout à 0

# Étape 2 : Détection des outliers avec la méthode IQR
Q1 = wine_df[feature_names].quantile(0.25)
Q3 = wine_df[feature_names].quantile(0.75)
IQR = Q3 - Q1

out_low = Q1 - 1.5 * IQR
out_high = Q3 + 1.5 * IQR

is_outlier = (wine_df[feature_names] < out_low) | (wine_df[feature_names] > out_high)
outlier_rows = is_outlier.any(axis=1).sum()

print(f"\nNombre de lignes contenant au moins un outlier : {outlier_rows}")

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Évaluation de plusieurs k
K = range(2, 11)
silhouettes = []
db_scores = []

for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouettes.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

# Affichage des graphes
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(K, silhouettes, marker='o')
plt.title("Score Silhouette")
plt.xlabel("k")

plt.subplot(1, 2, 2)
plt.plot(K, db_scores, marker='o')
plt.title("Indice Davies-Bouldin")
plt.xlabel("k")

plt.tight_layout()
plt.show()

# Comparaison Lloyd vs KMeans++ avec k=3
k_final = 3

# Lloyd (init random)
kmeans_lloyd = KMeans(n_clusters=k_final, init='random', n_init=10, random_state=42)
labels_lloyd = kmeans_lloyd.fit_predict(X_scaled)
inertia_lloyd = kmeans_lloyd.inertia_
silhouette_lloyd = silhouette_score(X_scaled, labels_lloyd)
db_lloyd = davies_bouldin_score(X_scaled, labels_lloyd)
ari_lloyd = adjusted_rand_score(y, labels_lloyd)

# k-means++
kmeans_pp = KMeans(n_clusters=k_final, init='k-means++', n_init=10, random_state=42)
labels_pp = kmeans_pp.fit_predict(X_scaled)
inertia_pp = kmeans_pp.inertia_
silhouette_pp = silhouette_score(X_scaled, labels_pp)
db_pp = davies_bouldin_score(X_scaled, labels_pp)
ari_pp = adjusted_rand_score(y, labels_pp)

# Affichage des comparaisons
print("\n--- Comparaison Lloyd vs KMeans++ (k=3) ---")
print(f"Lloyd       - Inertie: {inertia_lloyd:.2f}, Silhouette: {silhouette_lloyd:.3f}, DB: {db_lloyd:.3f}, ARI: {ari_lloyd:.3f}")
print(f"KMeans++    - Inertie: {inertia_pp:.2f}, Silhouette: {silhouette_pp:.3f}, DB: {db_pp:.3f}, ARI: {ari_pp:.3f}")

# Visualisation PCA clusters (avec KMeans++)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(kmeans_pp.cluster_centers_)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_pp, cmap='viridis', alpha=0.6)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200)
plt.title("Clusters Wine (k-means++) - PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.3)
plt.show()

# Prédiction sur de nouveaux échantillons
new_samples = [
    [13.2, 2.77, 2.51, 18.5, 98.0, 2.2, 1.28, 0.26, 1.56, 5.68, 1.12, 3.48, 650],
    [12.4, 1.9, 2.2, 19.0, 100.0, 2.0, 1.2, 0.3, 1.5, 4.2, 1.0, 3.0, 700],
    [13.8, 2.4, 2.6, 16.0, 90.0, 2.6, 1.5, 0.28, 1.7, 6.1, 1.3, 3.8, 750]
]

new_samples_scaled = scaler.transform(new_samples)
predictions = kmeans_pp.predict(new_samples_scaled)

print("\nPrédictions pour de nouveaux vins :")
for i, (sample, cluster) in enumerate(zip(new_samples, predictions)):
    print(f"Vin {i+1}: {sample} => Cluster {cluster}")
