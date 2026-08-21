import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine()
X_full = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

group_x_features = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
                    'magnesium', 'total_phenols', 'color_intensity']

group_y_features = ['flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
                    'hue', 'od280/od315_of_diluted_wines', 'proline']

df = pd.DataFrame(X_full, columns=feature_names)
X_group = df[group_x_features].values
Y_group = df[group_y_features].values

means_X = np.mean(X_group, axis=0)
stds_X = np.std(X_group, axis=0, ddof=1)
X_standardized = (X_group - means_X) / stds_X

means_Y = np.mean(Y_group, axis=0)
stds_Y = np.std(Y_group, axis=0, ddof=1)
Y_standardized = (Y_group - means_Y) / stds_Y

def manual_cca(X, Y, n_components=1):
    n, p = X.shape

    Cxx = np.cov(X, rowvar=False)
    Cyy = np.cov(Y, rowvar=False)
    Cxy = np.cov(X, Y, rowvar=False)[:p, p:]

    Cxx_inv = np.linalg.pinv(Cxx)
    Cyy_inv = np.linalg.pinv(Cyy)

    M = Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T

    eigenvalues, eigenvectors_x = np.linalg.eig(M)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors_x = eigenvectors_x[:, idx]

    eigenvectors_y = Cyy_inv @ Cxy.T @ eigenvectors_x

    for i in range(eigenvectors_x.shape[1]):
        eigenvectors_x[:, i] = eigenvectors_x[:, i] / np.sqrt(eigenvectors_x[:, i].T @ Cxx @ eigenvectors_x[:, i])
        eigenvectors_y[:, i] = eigenvectors_y[:, i] / np.sqrt(eigenvectors_y[:, i].T @ Cyy @ eigenvectors_y[:, i])

    return eigenvalues[:n_components], eigenvectors_x[:, :n_components], eigenvectors_y[:, :n_components]


eigenvalues, x_weights, y_weights = manual_cca(X_standardized, Y_standardized, n_components=1)

print("Собственные значения:", eigenvalues)

U = X_standardized @ x_weights
V = Y_standardized @ y_weights

rho1 = np.sqrt(eigenvalues[0])
print(f"Первая каноническая корреляция ρ₁ = {rho1:.4f}")

if rho1 >= 0.7:
    strength = "очень сильная"
elif rho1 >= 0.5:
    strength = "сильная"
elif rho1 >= 0.3:
    strength = "умеренная"
else:
    strength = "слабая"

print(f"Сила связи между группами: {strength}")

plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green']

for i, target_name in enumerate(target_names):
    plt.scatter(U[y == i], V[y == i],
                c=colors[i], marker='o',
                label=target_name, alpha=0.7, s=60,
                edgecolors='black', linewidth=0.5)

plt.xlabel('U (каноническая переменная из группы X)')
plt.ylabel('V (каноническая переменная из группы Y)')
plt.title(f'Первая каноническая пара\n(ρ₁ = {rho1:.3f}, ручная реализация)', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

weights_df_x = pd.DataFrame({
    'Признак': group_x_features,
    'Вес_U': x_weights.flatten(),
    'Абс_вес_U': np.abs(x_weights.flatten())
})

weights_df_y = pd.DataFrame({
    'Признак': group_y_features,
    'Вес_V': y_weights.flatten(),
    'Абс_вес_V': np.abs(y_weights.flatten())
})

print("Канонические веса для первой пары:")

print("\nТоп-3 признака группы X (физико-химические):")
top_x = weights_df_x.nlargest(3, 'Абс_вес_U')[['Признак', 'Вес_U']]
print(top_x.to_string(index=False))

print("\nТоп-3 признака группы Y (ароматические/фенольные):")
top_y = weights_df_y.nlargest(3, 'Абс_вес_V')[['Признак', 'Вес_V']]
print(top_y.to_string(index=False))

