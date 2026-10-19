import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from sklearn.metrics import confusion_matrix, roc_curve

X_train, X_test, y_train, y_test = generate_data(
    n_train=400, n_test=100, n_features=2, contamination=0.1)

def run(c):
    mdl = KNN(contamination=c)
    mdl.fit(X_train)
    for name, y, pred in [("train", y_train, mdl.labels_),
                          ("test", y_test, mdl.predict(X_test))]:
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        print(c, name, tn, fp, fn, tp, (tp / (tp + fn) + tn / (tn + fp)) / 2)
    return mdl.decision_function(X_test)

fpr, tpr, _ = roc_curve(y_test, run(0.1))
plt.plot(fpr, tpr)
plt.show()

for c in [0.05, 0.2, 0.3]:
    run(c)