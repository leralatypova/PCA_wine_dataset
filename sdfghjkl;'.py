import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2


Sigma = np.array([[2.25, 1.0],
                  [1.0, 1.0]])

rho = Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])

print(f"Коэффициент корреляции: {rho:.4f}")

eigenvals, eigenvecs = np.linalg.eig(Sigma)

lambda1, lambda2 = eigenvals
u1, u2 = eigenvecs.T

print("Уравнения для главных компонент:")
print(f"PC1 = {u1[0]:.4f}·xi1 + {u1[1]:.4f}·xi2")
print(f"PC2 = {u2[0]:.4f}·xi1 + {u2[1]:.4f}·xi2")


def plot_ellipse(eigvals, eigvecs, ax, n_std=1.0, **kwargs):
    mean = np.zeros(2)

    vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
    theta = np.arctan2(vy, vx)

    width, height = 2 * n_std * np.sqrt(eigvals)

    ellipse = plt.matplotlib.patches.Ellipse(
        xy=mean, width=width, height=height,
        angle=np.degrees(theta), **kwargs
    )
    ax.add_patch(ellipse)

    return width / 2, height / 2, theta


fig, ax = plt.subplots(figsize=(10, 8))

semi_major, semi_minor, theta = plot_ellipse(eigenvals, eigenvecs, ax, n_std=1.0,
                                             edgecolor='red', facecolor='none',
                                             linewidth=2, label='Эллипс рассеяния')

origin = np.zeros(2)
ax.quiver(*origin, *u1 * np.sqrt(lambda1), color='blue', scale=1,
          scale_units='xy', angles='xy', width=0.01, label='Главная ось 1')
ax.quiver(*origin, *u2 * np.sqrt(lambda2), color='green', scale=1,
          scale_units='xy', angles='xy', width=0.01, label='Главная ось 2')

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax.set_xlabel('xi1')
ax.set_ylabel('xi2')
ax.set_title('Эллипс рассеяния с главными осями')
ax.legend()
plt.show()

n_samples = 10000

X_normal =  np.random.multivariate_normal([0, 0], Sigma, n_samples)

Sigma_inv = np.linalg.inv(Sigma)
mahalanobis_sq_normal = np.sum(X_normal @ Sigma_inv * X_normal, axis=1)

inside_normal = np.sum(mahalanobis_sq_normal <= 1.0)
prop_inside_normal = inside_normal / n_samples
expected_prop = chi2.cdf(1.0, df=2)

print("Нормальное распределение:")
print(f"Ожидаемая доля внутри эллипса: {expected_prop:.4f}")
print(f"Доля внутри эллипса: {prop_inside_normal:.4f}")
print(f"Разница: {prop_inside_normal - expected_prop:.4f}")
print()

n_uniform = np.random.uniform(-np.sqrt(3), np.sqrt(3), (n_samples, 2))

L = np.linalg.cholesky(Sigma)
X_nonnormal = n_uniform @ L.T

mahalanobis_sq_nonnormal = np.sum(X_nonnormal @ Sigma_inv * X_nonnormal, axis=1)

inside_nonnormal = np.sum(mahalanobis_sq_nonnormal <= 1.0)
prop_inside_nonnormal = inside_nonnormal / n_samples

print("Ненормальное распределение (равномерное):")
print(f"Ожидаемая доля внутри эллипса: {expected_prop:.4f}")
print(f"Наблюдаемая доля внутри эллипса: {prop_inside_nonnormal:.4f}")
print(f"Разница: {prop_inside_nonnormal - expected_prop:.4f}")
print()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(X_normal[:, 0], X_normal[:, 1], alpha=0.3, s=1, color='blue', label='Точки')
inside_mask_normal = mahalanobis_sq_normal <= 1.0
ax1.scatter(X_normal[inside_mask_normal, 0], X_normal[inside_mask_normal, 1],
            alpha=0.6, s=2, color='red', label='Внутри эллипса')
plot_ellipse(eigenvals, eigenvecs, ax1, n_std=1.0, edgecolor='red', facecolor='none', linewidth=2)
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title(f'Нормальное распределение\n({prop_inside_normal * 100:.1f}% внутри эллипса)')
ax1.legend()

ax2.scatter(X_nonnormal[:, 0], X_nonnormal[:, 1], alpha=0.3, s=1, color='blue', label='Точки')
inside_mask_nonnormal = mahalanobis_sq_nonnormal <= 1.0
ax2.scatter(X_nonnormal[inside_mask_nonnormal, 0], X_nonnormal[inside_mask_nonnormal, 1],
            alpha=0.6, s=2, color='red', label='Внутри эллипса')
plot_ellipse(eigenvals, eigenvecs, ax2, n_std=1.0, edgecolor='red', facecolor='none', linewidth=2)
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_title(f'Равномерное распределение\n({prop_inside_nonnormal * 100:.1f}% внутри эллипса)')
ax2.legend()

plt.tight_layout()
plt.show()
