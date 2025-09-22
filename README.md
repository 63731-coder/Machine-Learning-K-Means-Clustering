# Machine Learning: K-Means Clustering

**Author:** Nicoleta Opre
**Course:** 4ALGL4A, Academic Year 2024-2025

---

## Description

This project provides a practical exploration of the **K-Means algorithm** for unsupervised clustering. The objectives are to:

* Understand the workings of K-Means and K-Means++.
* Apply K-Means on different datasets.
* Preprocess and clean data to improve clustering results.
* Analyze the quality of clustering using metrics such as inertia and silhouette score.

## Installation

To run the code, ensure you have Python 3 installed along with the following packages:

```bash
pip install numpy pandas matplotlib scikit-learn seaborn
```

## Usage

1. Load your dataset using `pandas`.
2. Preprocess data (e.g., normalization, missing value handling).
3. Initialize and fit K-Means:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
```

4. Evaluate clustering results:

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
print(f"Silhouette Score: {score}")
```

5. Visualize clusters:

```python
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
```

## Datasets

You can test K-Means on any dataset. Common choices include:

* Iris dataset
* Synthetic datasets with `make_blobs`
* Custom datasets in CSV format

## Notes

* K-Means is sensitive to the choice of initial centroids; using `k-means++` often improves convergence.
* Standardizing features can improve clustering quality.
* Always evaluate the clustering with appropriate metrics.

## References

* [Scikit-Learn KMeans Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
* [Wikipedia: K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
