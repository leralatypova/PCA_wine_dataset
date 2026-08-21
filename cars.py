import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/mtcars.csv'
mtcars = pd.read_csv(url, index_col=0)
feature_names = ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec']
X = mtcars[feature_names].values


def pca_funk(X, standardize):
    if standardize:
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0, ddof=1)
        X_processed = (X - means) / stds
    else:
        X_processed = X - np.mean(X, axis=0)

    cov_matrix = np.cov(X_processed, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:, idx]

    total_variance = np.sum(eigenvalues_sorted)
    explained_variance = eigenvalues_sorted / total_variance

    print("Собственные значения:", eigenvalues_sorted.round(4))
    print("Доля объясненной дисперсии:", explained_variance.round(4))
    print(f"PC1 объясняет {explained_variance[0]:.1%} общей дисперсии")

    return {
        'eigenvalues': eigenvalues_sorted,
        'eigenvectors': eigenvectors_sorted,
        'explained_variance': explained_variance,
        'X_processed': X_processed,
        'loadings': eigenvectors_sorted[:, 0]
    }

print('Без стандартизации')
not_standrt = pca_funk(X, False)

print('Стандартизированные данные')
standrt = pca_funk(X, True)

means = np.mean(X, axis=0)
outlier = means.copy()
outlier[3] = 500
outlier[5] = 8

X_with_outlier = np.vstack([X, outlier.reshape(1, -1)])
print("Без стандартизации с выбросом")
not_std_outlier = pca_funk(X_with_outlier, False)
print("Стандартизированные с выбросом")
std_outlier = pca_funk(X_with_outlier, True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Нагрузки для первой главной компоненты',fontsize=16)


results = [
    (not_standrt, 'Сырые данные(без стандартизации)', axes[0, 0]),
    (standrt, 'Стандартизованные данные', axes[0, 1]),
    (not_std_outlier, 'С выбросом(без стандартизации)', axes[1, 0]),
    (std_outlier, 'С выбросом(со стандартизацией)', axes[1, 1])
]

for i, (result, title, ax) in enumerate(results):
    loadings = result['loadings']
    bars = ax.bar(range(len(loadings)), loadings, alpha=0.7)
    ax.set_title(title)
    ax.set_xticks(range(len(loadings)))
    ax.set_xticklabels(feature_names, rotation=45)
    ax.set_ylabel('Нагрузка PC1')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)

plt.tight_layout()
plt.show()

print("Наиболее влиятельные признаки для PC1 в каждом случае:")

cases = [
    ("Сырые данные", not_standrt, feature_names),
    ("Стандартизованные", standrt, feature_names),
    ("С выбросом (сырые)", not_std_outlier, feature_names),
    ("С выбросом (стандартиз-е)", std_outlier, feature_names)
]

for case_name, result, names in cases:
    loadings = result['loadings']
    max_idx = np.argmax(np.abs(loadings))
    max_feature = names[max_idx]
    max_loading = loadings[max_idx]
    print(f"{case_name}: {max_feature} (нагрузка: {max_loading:.3f})")

##############
print("Вторая часть")
X = mtcars[['wt', 'hp']].values
y = mtcars['mpg'].values

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

def calculate_r2(y_true, y_pred):

    ss_residual = np.sum((y_true - y_pred) ** 2)
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    return r2

R2=calculate_r2(y, y_pred)

print(f"mpg = {model.intercept_:.3f} + {model.coef_[0]:.3f}·wt + {model.coef_[1]:.3f}·hp")
print(f"Коэффициент детерминации: {R2:.4f}")

n_bootstrap = 1000
n_samples = len(mtcars)
R2_bootstrap = []

for i in range(n_bootstrap):
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_boot = X[indices]
    y_boot = y[indices]

    model_boot = LinearRegression()
    model_boot.fit(X_boot, y_boot)

    y_pred_boot = model_boot.predict(X_boot)
    R2_boot = calculate_r2(y_boot, y_pred_boot)
    R2_bootstrap.append(R2_boot)

R2_bootstrap = np.array(R2_bootstrap)

plt.figure(figsize=(12, 8))
plt.hist(R2_bootstrap, bins=30, alpha=0.7, color='skyblue', edgecolor='black',
         density=True, label='Бутстреп распределение R²')

ci_lower = np.percentile(R2_bootstrap, 2.5)
ci_upper = np.percentile(R2_bootstrap, 97.5)

print(f"Нижняя граница (2.5%): {ci_lower:.4f}")
print(f"Верхняя граница (97.5%): {ci_upper:.4f}")
print(f"Истинное значение R^2: {R2:.4f}")

plt.axvline(R2, color='red', linestyle='--', linewidth=2,
            label=f'Истинное R^2 = {R2:.3f}')
plt.axvline(ci_lower, color='orange', linestyle=':', linewidth=2,
            label=f'2.5% квантиль = {ci_lower:.3f}')
plt.axvline(ci_upper, color='orange', linestyle=':', linewidth=2,
            label=f'97.5% квантиль = {ci_upper:.3f}')

plt.xlabel('Коэффициент детерминации R^2', fontsize=12)
plt.ylabel('Плотность вероятности', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

