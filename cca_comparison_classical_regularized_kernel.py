import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from scipy.linalg import eigh, inv, sqrtm
from sklearn.metrics.pairwise import pairwise_kernels, euclidean_distances


url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/mtcars.csv'
mtcars = pd.read_csv(url, index_col=0)

X_features = ['cyl', 'disp', 'hp']
Y_features = ['mpg', 'wt', 'qsec']

X = mtcars[X_features].values
Y = mtcars[Y_features].values

X_std = (X-np.mean(X, axis=0))/ np.std(X, axis=0, ddof=1)
Y_std = (Y-np.mean(Y, axis=0))/ np.std(Y, axis=0, ddof=1)

cca = CCA(n_components=1)
cca.fit(X_std, Y_std)

X_c, Y_c = cca.transform(X_std, Y_std)

rho1_classic = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
a1_classic = cca.x_weights_[:, 0]
b1_classic = cca.y_weights_[:, 0]

print(f"Классический CCA:")
print(f"Первая каноническая корреляция: {rho1_classic:.4f}")
print("Веса для X (a1):", [f"{w:.4f}" for w in a1_classic])
print("Веса для Y (b1):", [f"{w:.4f}" for w in b1_classic])

def rcca_manual(X, Y, lambda_x, lambda_y):
    n, p = X.shape
    _, q = Y.shape

    Sigma_xx = X.T @ X / n + lambda_x * np.eye(p)
    Sigma_yy = Y.T @ Y / n + lambda_y * np.eye(q)
    Sigma_xy = X.T @ Y / n

    Sigma_xx_inv_sqrt = inv(sqrtm(Sigma_xx))
    Sigma_yy_inv_sqrt = inv(sqrtm(Sigma_yy))

    M = Sigma_xx_inv_sqrt @ Sigma_xy @ Sigma_yy_inv_sqrt

    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    canonical_corrs = S
    a_vectors = Sigma_xx_inv_sqrt @ U
    b_vectors = Sigma_yy_inv_sqrt @ Vt.T

    return canonical_corrs, a_vectors, b_vectors

reg_params = [(0.01, 0.01), (0.1, 0.1), (1.0, 1.0)]
results = []

for lambda_x, lambda_y in reg_params:
    canonical_corrs, a_vectors, b_vectors = rcca_manual(X_std, Y_std, lambda_x, lambda_y)

    if canonical_corrs is not None:
        rho1_rcca = canonical_corrs[0]
        a1_rcca = a_vectors[:, 0]
        b1_rcca = b_vectors[:, 0]

        results.append({
            'lambda_x': lambda_x,
            'lambda_y': lambda_y,
            'rho1': rho1_rcca,
            'a1': a1_rcca,
            'b1': b1_rcca
        })

        print(f"\nПараметры:  {lambda_x}, {lambda_y}")
        print(f"Первая каноническая корреляция: {rho1_rcca:.4f}")
        print("Веса для X (a1):", [f"{w:.4f}" for w in a1_rcca])
        print("Веса для Y (b1):", [f"{w:.4f}" for w in b1_rcca])

print("\nОбщая таблица:")
for result in results:
    change_rho = result['rho1'] - rho1_classic
    print(f"{'RCCA':<25} {result['lambda_x']:<8.2f} {result['lambda_y']:<8.2f} "
          f"{result['rho1']:<10.4f} {change_rho:>+8.4f}")



if results:
    print("\nАнализ компромисса стабильность-корреляция:")

    for i, result in enumerate(results):
        change_a = np.linalg.norm(result['a1'] - a1_classic)
        change_b = np.linalg.norm(result['b1'] - b1_classic)
        total_change = change_a + change_b

        correlation_loss = rho1_classic - result['rho1']

        print(f"\nlambda = {result['lambda_x']}:")
        print(f"  Потеря корреляции: {correlation_loss:.4f}")
        print(f"  Изменение весов: {total_change:.4f}")
        print(f"  Относительная потеря: {correlation_loss / rho1_classic:.1%}")


print('========================================================================================================================================')

X_nonlin = np.column_stack([X, mtcars['hp'].values ** 2])
Y_nonlin = np.column_stack([Y, 1 / (mtcars['mpg'].values)])

X_std_nonlin = (X_nonlin-np.mean(X_nonlin, axis=0))/ np.std(X_nonlin, axis=0, ddof=1)
Y_std_nonlin = (Y_nonlin-np.mean(Y_nonlin, axis=0))/ np.std(Y_nonlin, axis=0, ddof=1)
def compute_sigma_median(X):
    n = X.shape[0]
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            distances.append(np.linalg.norm(X[i] - X[j]))
    return np.median(distances)

def inv_sqrt_matrix(M):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    S_inv_sqrt = 1.0 / np.sqrt(S + 1e-10)
    return U @ np.diag(S_inv_sqrt) @ Vt

cca_lin = CCA(n_components=1)
cca_lin.fit(X_std_nonlin, Y_std_nonlin)
X_c_lin, Y_c_lin = cca_lin.transform(X_std_nonlin, Y_std_nonlin)
rho_lin = np.corrcoef(X_c_lin[:, 0], Y_c_lin[:, 0])[0, 1]
print(f"Первая каноническая корреляция(стандартный CCA) = {rho_lin:.4f}")

def rbf_kernel(X, sigma):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = np.linalg.norm(X[i] - X[j])
            K[i, j] = np.exp(-dist**2 / (2 * sigma**2))
    return K

def kernel_cca(Kx, Ky, lam):
    n = Kx.shape[0]

    one_n = np.ones((n, n)) / n
    Kx_centered = Kx - one_n @ Kx - Kx @ one_n + one_n @ Kx @ one_n
    Ky_centered = Ky - one_n @ Ky - Ky @ one_n + one_n @ Ky @ one_n

    Kx_reg = Kx_centered + lam * np.eye(n)
    Ky_reg = Ky_centered + lam * np.eye(n)

    Kx_inv_sqrt = inv_sqrt_matrix(Kx_reg)
    Ky_inv_sqrt = inv_sqrt_matrix(Ky_reg)

    M = Kx_inv_sqrt @ Kx_centered @ Ky_centered @ Ky_inv_sqrt

    U, D, Vt = np.linalg.svd(M)

    alpha = Kx_inv_sqrt @ U[:, 0]
    beta = Ky_inv_sqrt @ Vt.T[:, 0]

    f = Kx_centered @ alpha
    g = Ky_centered @ beta

    rho_ker = np.corrcoef(f, g)[0, 1]
    return min(abs(rho_ker), 1.0), f, g

sigma_x = compute_sigma_median(X_std_nonlin)
sigma_y = compute_sigma_median(Y_std_nonlin)

Kx = rbf_kernel(X_std_nonlin, sigma_x)
Ky = rbf_kernel(Y_std_nonlin, sigma_y)


rho_ker, f_ker, g_ker = kernel_cca(Kx, Ky, 0.1)
print(f"\nsigma_X = {sigma_x:.4f}, sigma_Y = {sigma_y:.4f}")
print(f"Первая каноническая корреляция rho_ker = {rho_ker:.4f}")

print(f"Линейный CCA:          rho_lin = {rho_lin:.4f}")
print(f"Kernel CCA:            rho_ker = {rho_ker:.4f}")
print(f"Разница:                         {rho_ker - rho_lin:+.4f}")

if rho_ker > rho_lin:
    print(f"Kernel CCA улучшил корреляцию на {(rho_ker - rho_lin)*100:.1f}%")
else:
    print(f"Kernel CCA ухудшил корреляцию на {(rho_lin - rho_ker)*100:.1f}%")

print(f"Линейный CCA (без нелинейных признаков): rho_lin = {rho1_classic:.4f}")

sigma_x_orig = compute_sigma_median(X_std)
sigma_y_orig = compute_sigma_median(Y_std)
print(f"\nsigma_X_orig = {sigma_x_orig:.4f}, sigma_Y_orig = {sigma_y_orig:.4f}")

Kx_orig = rbf_kernel(X_std, sigma_x_orig)
Ky_orig = rbf_kernel(Y_std, sigma_y_orig)


rho_ker_orig, f_orig, g_orig = kernel_cca(Kx_orig, Ky_orig, 0.1)
print(f"rho_ker = {rho_ker_orig:.4f}")

print(f"Kernel CCA (без нелинейных признаков): rho_ker = {rho_ker_orig:.4f}")

print("\nС нелинейными признаками:")
print(f"  Линейный CCA: rho_lin = {rho_lin:.4f}")
print(f"  Kernel CCA:   rho_ker = {rho_ker:.4f}")
print(f"  Выигрыш Kernel CCA: {rho_ker - rho_lin:+.4f}")

print("\nБез нелинейных признаков :")
print(f"  Линейный CCA: rho_lin = {rho1_classic:.4f}")
print(f"  Kernel CCA:   rho_ker = {rho_ker_orig:.4f}")
print(f"  Выигрыш Kernel CCA: {rho_ker_orig - rho1_classic:+.4f}")



