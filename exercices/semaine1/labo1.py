import numpy as np

# ============================
# EXERCICES 1.3 : Création de matrices avec des formules
# ============================

# Exercice 1.a
arr1 = np.array([range(i, i - 3, -1) for i in [9, 6, 3]])
print("Exercice 1.a :")
print(arr1)
print()
"""
range(i, i - 3, -1) :
- Commence à i
- Arrête-toi à i - 3 (non inclus)
- Diminue de 1 à chaque fois
"""

# Exercice 1.b
arr2 = np.array([[i] for i in [9, 6, 3]])
print("Exercice 1.b :")
print(arr2)
print()

# Exercice 1.c
arr3 = np.array([[i, i - 1] for i in [8, 5, 2]])
print("Exercice 1.c :")
print(arr3)
print()

# Exercice 1.d
arr4 = np.array([range(i, i - 2, -1) for i in [9, 6]])
print("Exercice 1.d :")
print(arr4)
print()

# Exercice 1.e
arr5 = np.array([i for i in range(9, 2, -3)])
print("Exercice 1.e :")
print(arr5)
print()

# Exercice 2 : Créer un tableau d’entier allant de 0 à 9
arr = np.arange(10)
print("Exercice 2:", arr)

# Exercice 3
print("Exercice 3:", arr[:5])

# Exercice 4
print("Exercice 4:", arr[6:])

# Exercice 5
print("Exercice 5:", arr[4:8])

# Exercice 6
print("Exercice 6:", arr[::2])  # Pas de 2

# Exercice 7
print("Exercice 7:", arr[1::2])  # Commence à 1, pas de 2

# Exercice 8
print("Exercice 8:", arr[-3:])  # De -3 à la fin

# Exercice 9
print("Exercice 9:", arr[::-1])  # Pas de -1

# Exercice 10
print("Exercice 10:", arr[:6][::-1])  # De 0 à 5, puis inversé

# Exercice 11
print("Exercice 11:", arr[:])

# Exercice 12
arr_nan = np.array([1, 2, np.nan, 4])
print("Exercice 12:", np.isnan(arr_nan))  # True si NaN

# Exercice 13 - trouver le nombre de lignes et de colonnes d’une matrice
mat = np.array([[1, 2, 3], [4, 5, 6]])
print("Exercice 13:", "Shape:", mat.shape)  # (2 lignes, 3 colonnes)

# Exercice 14 - trouver les données manquantes dans un tableau
arr_missing = np.array([1, np.nan, 3, np.nan, 5])
missing = np.isnan(arr_missing)
print("Exercice 14:", "Indices des valeurs manquantes:", np.where(missing)[0])


# ============================
# 1.4 : Codes Matplotlib
# ============================

import numpy as np
import matplotlib.pyplot as plt

# Tracer une sinusoïde
x = np.arange(0, 3 * np.pi, 0.2)
y = np.sin(x)

print("Plot des points sur la sinusoïde :")
plt.plot(x, y)
plt.title("Courbe sin(x)")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid()
plt.show()

# Tracer sin(x) et cos(x) ensemble
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x), '-', label='sin(x)')
plt.plot(x, np.cos(x), '--', label='cos(x)')
plt.title("sin(x) et cos(x)")
plt.legend()
plt.grid()
plt.show()

# Structure des graphes pour méthode du coude et silhouette (sans données)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.title('Méthode du coude (WCSS)', fontsize=14)
plt.xlabel('Nombre de clusters (k)', fontsize=12)
plt.ylabel('WCSS (inertie)', fontsize=12)

plt.subplot(1, 2, 2)
plt.title('Indice de silhouette moyen', fontsize=14)
plt.xlabel('Nombre de clusters (k)', fontsize=12)
plt.ylabel('Indice de silhouette', fontsize=12)

plt.tight_layout()
plt.show()


# ============================
# 1.5 : Pandas
# ============================

import pandas as pd
import numpy as np

# Version de Pandas
print("Version de Pandas :", pd.__version__)

# DataFrame simple avec array NumPy
ar = np.array([[1.1, 2, 3.3, 4], [2.7, 10, 5.4, 7], [5.3, 9, 1.5, 15]])
df = pd.DataFrame(ar, index=['a1', 'a2', 'a3'], columns=['A', 'B', 'C', 'D'])
print("DataFrame avec index et colonnes :\n", df)

# DataFrame avec liste de listes
data = [['Alex', 10], ['Bob', 12]]
df2 = pd.DataFrame(data, columns=['Name', 'Age'])
print("DataFrame noms et âges :\n", df2)

# DataFrame avec dictionnaire + NaN
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily',
             'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
df3 = pd.DataFrame(exam_data)
print("DataFrame original :\n", df3)
print("Shape (lignes, colonnes) :", df3.shape)
print("Premières lignes :\n", df3.head())

# Suppression des lignes avec des NaN
df_clean = df3.dropna()
print("DataFrame sans valeurs manquantes :\n", df_clean)
