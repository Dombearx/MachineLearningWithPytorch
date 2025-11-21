from typing import Any


import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        random_state = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.weights = random_state.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias = np.float64(0.0)
        self.errors = []

        for _ in range(self.n_iterations):
            errors = 0
            for x_i, y_i in zip(X, y):
                update = self.learning_rate * (y_i - self.predict(x_i))
                self.weights += update * x_i
                self.bias += update
                errors += int(update != 0.0)
            self.errors.append(errors)

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.net_input(X) >= 0.0, 1, 0)
