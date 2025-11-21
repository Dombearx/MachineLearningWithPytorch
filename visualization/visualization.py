import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, xlabel="sepal length", ylabel="petal length"):
    plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="setosa")
    plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="versicolor")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper left")

def plot_data_universtal(X, y, xlabel="feature 1", ylabel="feature 2", X_test=None, y_test=None):
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = plt.cm.RdYlBu

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f"class {cl}",
                    edgecolor="black")
        
    if X_test is not None and y_test is not None:
        for idx, cl in enumerate(np.unique(y_test)):
            plt.scatter(X_test[y_test == cl, 0],
                        X_test[y_test == cl, 1],
                        alpha=0.8,
                        c=colors[idx],
                        marker=markers[idx],
                        label=f"Test set",
                        edgecolor="black")
            plt.scatter(X_test[y_test == cl, 0],
                        X_test[y_test == cl, 1],
                        c="none",
                        edgecolor="black",
                        alpha=1.0,
                        linewidth=1,
                        marker='o',
                        s=100)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper left")

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = plt.cm.RdYlBu

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)