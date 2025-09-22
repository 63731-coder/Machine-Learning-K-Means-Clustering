import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Chargement des données
X = pd.read_csv("data/Mall_Customers.csv")

# Suppression de l'ID et conversion du genre
if 'CustomerID' in X.columns:
    X.drop('CustomerID', axis=1, inplace=True)
if 'Genre' in X.columns:
    X['Genre'] = X['Genre'].map({'Male': 0, 'Female': 1})

X = X[['Age','Annual Income (k$)']]

print("Aperçu des données Mall :")
print(X.head())
print("\nValeurs manquantes dans le dataset :")

# Étape 1 : Vérification des valeurs manquantes
print(X.isnull().sum())

# Étape 2 : Détection des outliers avec la méthode IQR
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1

out_low = Q1 - 1.5 * IQR
out_high = Q3 + 1.5 * IQR

is_outlier = (X < out_low) | (X > out_high)
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

# Comparaison Lloyd vs KMeans++ avec k=10
k_final = 3

# Lloyd
kmeans_lloyd = KMeans(n_clusters=k_final, init='random', n_init=10, random_state=42)
labels_lloyd = kmeans_lloyd.fit_predict(X_scaled)
inertia_lloyd = kmeans_lloyd.inertia_
silhouette_lloyd = silhouette_score(X_scaled, labels_lloyd)
db_lloyd = davies_bouldin_score(X_scaled, labels_lloyd)

# KMeans++
kmeans_pp = KMeans(n_clusters=k_final, init='k-means++', n_init=10, random_state=42)
labels_pp = kmeans_pp.fit_predict(X_scaled)
inertia_pp = kmeans_pp.inertia_
silhouette_pp = silhouette_score(X_scaled, labels_pp)
db_pp = davies_bouldin_score(X_scaled, labels_pp)

print("\n--- Comparaison Lloyd vs KMeans++ (k=6) ---")
print(f"Lloyd       - Inertie: {inertia_lloyd:.2f}, Silhouette: {silhouette_lloyd:.3f}, DB: {db_lloyd:.3f}")
print(f"KMeans++    - Inertie: {inertia_pp:.2f}, Silhouette: {silhouette_pp:.3f}, DB: {db_pp:.3f}")

# Visualisation PCA
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_pp, cmap='viridis', alpha=0.6)
plt.scatter(kmeans_pp.cluster_centers_[:, 0], kmeans_pp.cluster_centers_[:, 1], c='red', marker='X', s=200)
plt.title("Clusters Mall Customers (k-means++) - PCA")
plt.xlabel("Age")
plt.ylabel("Annual Income")
plt.grid(True, alpha=0.3)
plt.show()

# Prédiction de nouveaux clients
new_customers = [
    [25, 40],
    [27, 75],
    [60, 30]
]

new_customers_df = pd.DataFrame(new_customers, columns=X.columns)
new_customers_scaled = scaler.transform(new_customers_df)
predictions_customers = kmeans_pp.predict(new_customers_scaled)

print("\nPrédictions pour les nouveaux clients explicites :")
for i, (sample, cluster) in enumerate(zip(new_customers, predictions_customers)):
    print(f"Nouveau client {i+1}: {sample} => Cluster {cluster}")
