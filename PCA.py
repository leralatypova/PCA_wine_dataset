import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

means = np.mean(X, axis=0)
stds = np.std(X, axis=0, ddof=1)

X_centered = X - means
X_standardized = X_centered / stds

cov_matrix = np.cov(X_standardized, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[idx]
eigenvectors_sorted = eigenvectors[:, idx]

print("Отсортированные собственные значения:")
for i, val in enumerate(eigenvalues_sorted):
    print(f"PC{i+1}: {val:.4f}")

pc1 = eigenvectors_sorted[:, 0]
pc2 = eigenvectors_sorted[:, 1]

print(f"\nПервая главная компонента (PC1): {pc1.round(4)}")
print(f"Вторая главная компонента (PC2): {pc2.round(4)}")

X_pca = np.dot(X_standardized, eigenvectors_sorted[:, :2])

total_variance = np.sum(eigenvalues_sorted)
explained_variance_manual = eigenvalues_sorted[:2] / total_variance
cumulative_variance_manual = np.cumsum(eigenvalues_sorted) / total_variance

print(f"Объясненная дисперсия:")
print(f"PC1: {explained_variance_manual[0]:.3f}")
print(f"PC2: {explained_variance_manual[1]:.3f}")
print(f"Всего первыми двумя компонентами: {explained_variance_manual.sum():.1%}")

plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green']
markers = ['o', 's', '^']

for i, target_name in enumerate(target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                c=colors[i], marker=markers[i],
                label=target_name, alpha=0.7, s=60,
                edgecolors='black', linewidth=0.5)

plt.xlabel('Первая главная компонента (PC1)')
plt.ylabel('Вторая главная компонента (PC2)')
plt.title('PCA анализа датасета Wine', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.text(0.02, 0.98, f'Объясненная дисперсия: PC1: {explained_variance_manual[0]:.1%}, PC2: {explained_variance_manual[1]:.1%}',
         transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
         verticalalignment='top')

plt.tight_layout()
plt.show()

loadings = eigenvectors_sorted[:, :2]

print("Нагрузки для PC1 и PC2:")
loadings_df = pd.DataFrame({
    'Признак': wine.feature_names,
    'PC1': loadings[:, 0],
    'PC2': loadings[:, 1],
    'Абс_PC1': np.abs(loadings[:, 0]),
    'Абс_PC2': np.abs(loadings[:, 1])
})

print("Топ 5 признаков для PC1:")
print(loadings_df.nlargest(5, 'Абс_PC1')[['Признак', 'PC1']].to_string(index=False))

print("Топ 5 признаков для PC2:")
print(loadings_df.nlargest(5, 'Абс_PC2')[['Признак', 'PC2']].to_string(index=False))


