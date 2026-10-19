import matplotlib.pyplot as plt
from pyod.utils.data import generate_data

X_train, X_test, y_train, y_test = generate_data(
    n_train = 400, 
    n_test=100, 
    n_features=2, 
    contamination=0.1)

X_inliers = X_train[y_train == 0]
X_outliers = X_train[y_train == 1]

def add_plot (X_inliers, X_outliers, inlier_color = 'pink', outlier_color = 'green'):
    plt.scatter(X_inliers[:,0], X_inliers[:,1], label = 'Inliers', color = inlier_color, s = 40)
    plt.scatter(X_outliers[:,0], X_outliers[:,1], label = 'Outliers', color = outlier_color, s = 50, marker = '^')
    plt.legend()
    plt.show()

print (add_plot(X_inliers,X_outliers))