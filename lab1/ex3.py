import numpy as np
from pyod.utils.data import generate_data
from sklearn.metrics import balanced_accuracy_score

X, y = generate_data(n_train=1000, n_features=1, contamination=0.1,
                     train_only=True)

z = np.abs((X - X.mean()) / X.std()).ravel()
y_pred = (z > np.quantile(z, 0.9)).astype(int)

print(balanced_accuracy_score(y, y_pred))