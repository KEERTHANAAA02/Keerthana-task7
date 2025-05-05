import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

df = pd.read_csv("D:/INTERNSHIP/breast-cancer.csv")  

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

features = ['radius_mean', 'texture_mean']
X = df[features].values
y = df['diagnosis'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap = ListedColormap(['purple', 'yellow'])

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=cmap)
    plt.title(title)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.show()

linear_svm = SVC(kernel='linear', C=1)
linear_svm.fit(X_train, y_train)
plot_decision_boundary(linear_svm, X_train, y_train, "Linear SVM Decision Boundary")

rbf_svm = SVC(kernel='rbf', C=1, gamma=0.5)
rbf_svm.fit(X_train, y_train)
plot_decision_boundary(rbf_svm, X_train, y_train, "RBF SVM Decision Boundary")

print("Linear SVM Test Accuracy:", accuracy_score(y_test, linear_svm.predict(X_test)))
print("RBF SVM Test Accuracy:", accuracy_score(y_test, rbf_svm.predict(X_test)))

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 0.5, 1],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters from GridSearch:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

best_model = grid.best_estimator_
cv_scores = cross_val_score(best_model, X, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

plot_decision_boundary(best_model, X_train, y_train, "Best SVM after Tuning")
