import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

Lambda = np.array([
    [453, 38.9, -257, -219],
    [38.9, 37, -18, -26],
    [-257, -18, 210, 77],
    [-219, -26, 77, 172]
])

n = np.random.randint(50, 101)
print(f"n={n}")

mu = np.random.multivariate_normal([0, 0, 0, 0], np.eye(4))

print(f"Математические ожидания: {mu}")

data = np.random.multivariate_normal(mu, Lambda, n)

S = np.cov(data, rowvar=False)
S_inv = inv(S)

print("Исходная матрица ковариаций:")
print(Lambda)
print("Выборочная матрица ковариаций:")
print(S)
print("Обратная матрица ковариаций¹:")
print(S_inv)

X1 = data[:, 0]
X_rest = data[:, 1:]

X_design = np.column_stack([np.ones(n), X_rest])

beta, _, _, _ = np.linalg.lstsq(X_design, X1, rcond=None)

print("Коэффициенты регрессии:")
print(f"beta0 = {beta[0]:.4f}")
print(f"beta1 = {beta[1]:.4f}")
print(f"beta2 = {beta[2]:.4f}")
print(f"beta3 = {beta[3]:.4f}")

print(f"Уравнение регрессии:")
print(f"X1 = {beta[0]:.4f} + {beta[1]:.4f}·X2 + {beta[2]:.4f}·X3 + {beta[3]:.4f}·X4")

X1_pred = X_design @ beta

print("Первые 10 прогнозных значений:")
for i in range(10):
    print(f"Наблюдение {i + 1}: X₁ = {X1[i]:.2f}, Прогноз = {X1_pred[i]:.2f}")

residuals = X1 - X1_pred
residual_variance = np.var(residuals, ddof=4)

print(f"Остаточная дисперсия: {residual_variance:.4f}")

SST = np.sum((X1 - np.mean(X1)) ** 2)
SSR = np.sum((X1_pred - np.mean(X1)) ** 2)

R_squared = SSR / SST
multiple_R = np.sqrt(R_squared)

print(f"Коэффициент детерминации: {R_squared:.4f}")
print(f"Множественный коэффициент корреляции: {multiple_R:.4f}")
print(f"Модель объясняет {R_squared*100:.2f}% вариации X1")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X1, X1_pred, alpha=0.7, color='blue', edgecolors='black')
plt.plot([X1.min(), X1.max()], [X1.min(), X1.max()], 'r--', linewidth=2)
plt.xlabel('Действительные значения X1')
plt.ylabel('Прогнозные значения X1*')
plt.title('Сравнение действительных и прогнозных значений')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(range(len(residuals)), residuals, alpha=0.7, color='red')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.xlabel('Номер наблюдения')
plt.ylabel('Остатки')
plt.title('График остатков')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

mse = np.mean(residuals ** 2)

print(f"Среднеквадратичная ошибка: {mse:.4f}")

R = np.corrcoef(data, rowvar=False)
R_inv = inv(R)

print("Матрица корреляций:")
print(R)
print("Обратная матрица корреляций:")
print(R_inv)


def partial_correlation(corr_matrix, i, j, conditioned_on):
    k = [idx for idx in range(corr_matrix.shape[0]) if idx not in [i, j] + conditioned_on]

    if len(k) == 0:
        return corr_matrix[i, j]

    P = corr_matrix[[i, j] + k, :][:, [i, j] + k]

    P_inv = inv(P)
    rho_ij = -P_inv[0, 1] / np.sqrt(P_inv[0, 0] * P_inv[1, 1])

    return rho_ij


variables = ['X1', 'X2', 'X3', 'X4']
print("Частные корреляции X1 с другими переменными:")

for j in range(1, 4):
    conditioned_on = [k for k in range(1, 4) if k != j]
    partial_corr = partial_correlation(R, 0, j, conditioned_on)
    print(f"r(X1, X{j + 1} | остальные) = {partial_corr:.4f}")

print("Матрица частных корреляций:")
partial_corr_matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(i + 1, 4):
        conditioned_on = [k for k in range(4) if k != i and k != j]
        partial_corr_matrix[i, j] = partial_correlation(R, i, j, conditioned_on)
        partial_corr_matrix[j, i] = partial_corr_matrix[i, j]

print(partial_corr_matrix)

print("Сравнение парных и частных корреляций:")
print("Переменные | Парная корр. | Частная корр. | Разница")

for i in range(4):
    for j in range(i + 1, 4):
        pair_corr = R[i, j]
        conditioned_on = [k for k in range(4) if k != i and k != j]
        part_corr = partial_correlation(R, i, j, conditioned_on)
        diff = abs(pair_corr) - abs(part_corr)

        correlation_type = ""

        if diff > 0:
            correlation_type = "ложная"
        else:
            correlation_type = "скрытая"

        print(
            f"r({variables[i]},{variables[j]}) | {pair_corr:11.4f} | {part_corr:13.4f} | {diff:7.4f} {correlation_type}")