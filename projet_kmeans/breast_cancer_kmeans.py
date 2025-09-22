import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA

# Chargement et préparation des données
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names

cancer_df = pd.DataFrame(X, columns=feature_names)
cancer_df['target'] = y

print("Aperçu des données Breast Cancer :")
print(cancer_df.head())

# Étape 1 : Vérification des valeurs manquantes
print("\nValeurs manquantes dans le dataset :")
print(cancer_df.isnull().sum())  # Toutes doivent être à 0

# Étape 2 : Détection des outliers avec la méthode IQR
Q1 = cancer_df[feature_names].quantile(0.25)
Q3 = cancer_df[feature_names].quantile(0.75)
IQR = Q3 - Q1

out_low = Q1 - 1.5 * IQR
out_high = Q3 + 1.5 * IQR

is_outlier = (cancer_df[feature_names] < out_low) | (cancer_df[feature_names] > out_high)
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

# Comparaison Lloyd vs KMeans++ avec k=2
k_final = 2

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
print("\n--- Comparaison Lloyd vs KMeans++ (k=2) ---")
print(f"Lloyd       - Inertie: {inertia_lloyd:.2f}, Silhouette: {silhouette_lloyd:.3f}, DB: {db_lloyd:.3f}, ARI: {ari_lloyd:.3f}")
print(f"KMeans++    - Inertie: {inertia_pp:.2f}, Silhouette: {silhouette_pp:.3f}, DB: {db_pp:.3f}, ARI: {ari_pp:.3f}")

# Visualisation PCA clusters (avec KMeans++)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(kmeans_pp.cluster_centers_)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_pp, cmap='viridis', alpha=0.6)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200)
plt.title("Clusters Breast Cancer (k-means++) - PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.3)
plt.show()

# Prédiction sur de nouvelles observations
new_samples = [
    [14.0, 20.0, 90.0, 600.0, 0.10, 0.15, 0.08, 0.05, 0.18, 0.06,
     0.40, 1.2, 3.0, 30.0, 0.007, 0.03, 0.04, 0.015, 0.02, 0.004,
     15.0, 25.0, 100.0, 700.0, 0.13, 0.22, 0.20, 0.10, 0.28, 0.07],

    [20.0, 22.0, 130.0, 1200.0, 0.15, 0.25, 0.18, 0.12, 0.20, 0.10,
     0.70, 1.8, 5.0, 50.0, 0.010, 0.04, 0.05, 0.02, 0.03, 0.006,
     25.0, 32.0, 170.0, 1800.0, 0.20, 0.35, 0.30, 0.15, 0.40, 0.12],

    [11.0, 12.0, 70.0, 370.0, 0.07, 0.08, 0.05, 0.03, 0.14, 0.05,
     0.25, 0.8, 2.0, 15.0, 0.004, 0.02, 0.02, 0.01, 0.01, 0.002,
     12.0, 15.0, 80.0, 400.0, 0.09, 0.10, 0.09, 0.04, 0.20, 0.06]
]

# Standardisation des nouveaux échantillons
new_samples_scaled = scaler.transform(new_samples)

# Prédiction avec le modèle k-means++
predictions = kmeans_pp.predict(new_samples_scaled)

print("\nPrédictions pour les nouvelles observations explicites :")
for i, cluster in enumerate(predictions):
    print(f"Échantillon {i + 1} => Cluster {cluster}")
