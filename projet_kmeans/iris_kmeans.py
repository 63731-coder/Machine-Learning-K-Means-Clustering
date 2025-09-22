import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA

# Chargement du dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Création d'un DataFrame
iris_df = pd.DataFrame(X, columns=feature_names)
iris_df['target'] = y

print("Aperçu des données Iris :")
print(iris_df.head())

# Étape 1 : Vérification des valeurs manquantes
print("\nValeurs manquantes dans le dataset :")
print(iris_df.isnull().sum())  # doit être tout à 0 pour ce dataset

# Étape 2 : Détection des outliers avec la méthode IQR
Q1 = iris_df[feature_names].quantile(0.25)
Q3 = iris_df[feature_names].quantile(0.75)
IQR = Q3 - Q1

out_low = Q1 - 1.5 * IQR
out_high = Q3 + 1.5 * IQR

# Détection booléenne des outliers
is_outlier = (iris_df[feature_names] < out_low) | (iris_df[feature_names] > out_high)
outlier_rows = is_outlier.any(axis=1).sum()

print(f"\nNombre de lignes contenant au moins un outlier : {outlier_rows}")

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Évaluation de plusieurs k (2 à 10)
K = range(2, 11)
silhouettes = []
db_scores = []

for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouettes.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

# Affichage des scores
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

# Affichage comparatif
print("\n--- Comparaison Lloyd vs KMeans++ (k=3) ---")
print(f"Lloyd       - Inertie: {inertia_lloyd:.2f}, Silhouette: {silhouette_lloyd:.3f}, DB: {db_lloyd:.3f}, ARI: {ari_lloyd:.3f}")
print(f"KMeans++    - Inertie: {inertia_pp:.2f}, Silhouette: {silhouette_pp:.3f}, DB: {db_pp:.3f}, ARI: {ari_pp:.3f}")

# Visualisation PCA des clusters (avec KMeans++)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(kmeans_pp.cluster_centers_)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_pp, cmap='viridis', alpha=0.6)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200)
plt.title("Clusters Iris (k-means++) - PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.3)
plt.show()

# Prédiction de nouvelles fleurs
new_samples = [
    [5.1, 3.5, 1.4, 0.2], #setosa
    [6.3, 3.3, 4.7, 1.6], #versicolor
    [7.0, 3.2, 6.0, 1.8] #virginica
]

new_samples_scaled = scaler.transform(new_samples)
predictions = kmeans_pp.predict(new_samples_scaled)

print("\nPrédictions pour de nouvelles fleurs :")
for i, (sample, cluster) in enumerate(zip(new_samples, predictions)):
    print(f"Fleur {i+1}: {sample} => Cluster {cluster}")
