import numpy as np
from sklearn.metrics import balanced_accuracy_score

mu = np.array([2.0, -1.0])
Sigma = np.array([[4.0, 1.8], [1.8, 1.0]])
L = np.linalg.cholesky(Sigma)

inliers = np.random.standard_normal((900, 2)) @ L.T + mu
outliers = np.random.uniform(mu - 8, mu + 8, (100, 2))

X = np.vstack([inliers, outliers])
y = np.r_[np.zeros(900), np.ones(100)]

score = np.abs((X - X.mean(axis=0)) / X.std(axis=0)).max(axis=1)
y_pred = (score > np.quantile(score, 0.9)).astype(int)

print(balanced_accuracy_score(y, y_pred))